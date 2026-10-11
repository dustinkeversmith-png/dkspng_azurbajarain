#pragma once

#include "stevejob/messaging/sjc1_container.hpp"
#include "stevejob/messaging/totr_engine.hpp"
#include "stevejob/models/architectures/bottleneck_ae.hpp"
#include "stevejob/models/architectures/mlp_regressor.hpp"
#include "stevejob/storage/azurite_client.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <pybind11/pybind11.h>

#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace stevejob::bridge {

struct TournamentCandidate {
    std::string name;
    std::string weights;
    std::string bias;
};

class EngineGateway {
public:
    template <typename Model>
    static py::dict seed_model(Model& model,
                               const std::string& target_blob,
                               const std::string& journal_blob,
                               std::size_t journal_capacity) {
        std::lock_guard<std::mutex> lock(gate());
        const stevejob::models::RegressionScore score = model.evaluate();
        return store_seed(model.network(), score, model.get_directives(), target_blob, journal_blob, journal_capacity);
    }

    static py::dict seed_kind(const std::string& kind,
                              const std::string& target_blob,
                              const std::string& journal_blob,
                              std::size_t journal_capacity) {
        if (kind == "bottleneck_ae") {
            stevejob::models::BottleneckAutoencoder model;
            return seed_model(model, target_blob, journal_blob, journal_capacity);
        }
        stevejob::models::MlpRegressor model;
        return seed_model(model, target_blob, journal_blob, journal_capacity);
    }

    static py::dict read_meta(const std::string& target_blob, const std::string& journal_blob) {
        const Loaded loaded = load(target_blob, journal_blob);
        const stevejob::models::RegressionScore score = score_loaded(loaded.model);
        const stevejob::models::ModelDirective directives = directives_for(loaded.model.architecture);
        py::dict receipt;
        receipt["status"] = "ready";
        receipt["layers"] = slice_names(loaded.model);
        receipt["loss"] = score.loss;
        receipt["max_residual"] = score.max_residual;
        receipt["checksum"] = loaded.checksum;
        receipt["architecture"] = loaded.model.architecture;
        receipt["directives"] = directive_dict(directives);
        receipt["telemetry"] = telemetry_dict(loaded.model);
        return receipt;
    }

    static py::dict read_slice(const std::string& target_blob,
                               const std::string& journal_blob,
                               const std::string& slice_id) {
        const Loaded loaded = load(target_blob, journal_blob);
        const stevejob::models::SliceRef slice = loaded.model.find_slice(slice_id);
        const stevejob::models::RegressionScore score = score_loaded(loaded.model);
        py::dict receipt;
        receipt["status"] = "ready";
        receipt["layer_name"] = slice.slice_id;
        receipt["in_dim"] = static_cast<int>(slice.rows);
        receipt["out_dim"] = static_cast<int>(slice.cols);
        receipt["totr_w"] = loaded.model.slice_token(slice);
        receipt["biases"] = loaded.model.slice_bias(slice);
        receipt["loss"] = score.loss;
        receipt["max_residual"] = score.max_residual;
        receipt["checksum"] = loaded.checksum;
        receipt["gradient_hint"] = gradient_hint(loaded.model, slice_id);
        receipt["telemetry"] = telemetry_dict(loaded.model);
        receipt["directives"] = directive_dict(directives_for(loaded.model.architecture));
        return receipt;
    }

    static py::dict commit_tournament(const std::string& slice_id,
                                      const std::vector<TournamentCandidate>& candidates,
                                      const std::string& target_blob,
                                      const std::string& journal_blob,
                                      std::size_t journal_capacity) {
        std::lock_guard<std::mutex> lock(gate());
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = bare_blob(target_blob, "models/");
        const std::string journal_name = bare_blob(journal_blob, "evals/");
        const std::string current = client.download_blob("models", stored_name);
        stevejob::models::NeuralModel baseline =
            stevejob::models::NeuralModel::import_document(stevejob::messaging::Sjc1Container::unpack(current));
        stevejob::storage::SlidingJournal journal(journal_capacity == 0 ? 1 : journal_capacity);
        load_journal(client, journal_name, journal);
        const double previous_loss = journal.size() > 0 ? journal.history().back().loss
                                                        : std::numeric_limits<double>::infinity();
        double best_loss = std::numeric_limits<double>::infinity();
        std::string winner = "none";
        stevejob::models::NeuralModel best_model = baseline;
        stevejob::models::RegressionScore best_score{};
        bool any_scored = false;
        bool any_divergent = false;
        bool any_unchanged = false;
        const stevejob::models::SliceRef active_slice = baseline.find_slice(slice_id);
        const auto active_block = stevejob::messaging::TotrEngine::decode_matrix(
            baseline.slice_token(active_slice), active_slice.rows, active_slice.cols);
        const auto active_bias = stevejob::messaging::TotrEngine::decode_vector(
            baseline.slice_bias(active_slice), active_slice.cols);
        py::list report;
        for (const TournamentCandidate& candidate : candidates) {
            stevejob::models::NeuralModel trial = baseline;
            py::dict row;
            row["name"] = candidate.name;
            try {
                const stevejob::models::SliceRef slice = trial.find_slice(slice_id);
                const auto block = stevejob::messaging::TotrEngine::decode_matrix(candidate.weights, slice.rows, slice.cols);
                const auto bias = stevejob::messaging::TotrEngine::decode_vector(candidate.bias, slice.cols);
                if (block == active_block && bias == active_bias) {
                    row["status"] = "unchanged";
                    any_unchanged = true;
                    report.append(row);
                    continue;
                }
                trial.patch_slice(slice_id, block, bias);
                const stevejob::models::RegressionScore score = score_loaded(trial);
                row["loss"] = score.loss;
                row["status"] = score.loss > 100.0 ? "rejected_divergent" : "scored";
                if (score.loss > 100.0) {
                    any_divergent = true;
                } else if (score.loss < best_loss) {
                    best_loss = score.loss;
                    best_score = score;
                    best_model = std::move(trial);
                    winner = candidate.name;
                    any_scored = true;
                }
            } catch (const std::invalid_argument&) {
                row["loss"] = 101.0;
                row["status"] = "rejected_divergent";
                any_divergent = true;
            }
            report.append(row);
        }
        constexpr double kMinCommitDrop = 1e-4;
        const bool kept = any_scored && (previous_loss - best_loss > kMinCommitDrop);
        std::string checksum = client.sha256_hex16(current);
        if (kept) {
            const std::string packed = stevejob::messaging::Sjc1Container::pack(best_model.export_document());
            client.upload_blob("models", stored_name, packed, "application/octet-stream");
            checksum = client.sha256_hex16(packed);
            const double delta = previous_loss < std::numeric_limits<double>::max() ? best_loss - previous_loss : 0.0;
            std::map<std::string, double> metrics{{"delta", delta}, {"sparsity", telemetry_value(best_model, "sparsity")}};
            std::map<std::string, std::string> meta{{"winner", winner}, {"slice", slice_id}};
            journal.add_record(best_score.loss, best_score.max_residual, best_score.std_residual, checksum, metrics, meta);
            client.upload_blob("evals", journal_name, journal.serialize(), "application/json");
        }
        py::dict receipt;
        if (!any_scored) {
            receipt["status"] = any_divergent && !any_unchanged ? "rejected_divergent" : "rejected";
        } else if (kept) {
            receipt["status"] = "committed_natively";
        } else {
            receipt["status"] = "rejected";
        }
        receipt["winner"] = winner;
        receipt["loss"] = any_scored ? best_loss : (any_divergent && !any_unchanged ? 101.0 : previous_loss);
        receipt["delta"] = any_scored && previous_loss < std::numeric_limits<double>::max() ? best_loss - previous_loss : 0.0;
        receipt["checksum"] = checksum;
        receipt["candidates"] = report;
        return receipt;
    }

private:
    struct Loaded {
        stevejob::models::NeuralModel model;
        std::string checksum;
    };

    static std::mutex& gate() {
        static std::mutex mutex;
        return mutex;
    }

    static std::string bare_blob(std::string name, const char* prefix) {
        const std::string marker(prefix);
        if (name.rfind(marker, 0) == 0) {
            name = name.substr(marker.size());
        }
        return name;
    }

    static py::dict store_seed(stevejob::models::NeuralModel& model,
                               const stevejob::models::RegressionScore& score,
                               const stevejob::models::ModelDirective& directives,
                               const std::string& target_blob,
                               const std::string& journal_blob,
                               std::size_t journal_capacity) {
        const std::string packed = stevejob::messaging::Sjc1Container::pack(model.export_document());
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = bare_blob(target_blob, "models/");
        const std::string journal_name = bare_blob(journal_blob, "evals/");
        client.upload_blob("models", stored_name, packed, "application/octet-stream");
        const std::string checksum = client.sha256_hex16(packed);
        stevejob::storage::SlidingJournal journal(journal_capacity == 0 ? 1 : journal_capacity);
        journal.add_record(score.loss, score.max_residual, score.std_residual, checksum, {{"delta", 0.0}}, {{"winner", "seed"}});
        client.upload_blob("evals", journal_name, journal.serialize(), "application/json");
        py::dict receipt;
        receipt["status"] = "seeded";
        receipt["loss"] = score.loss;
        receipt["max_residual"] = score.max_residual;
        receipt["checksum"] = checksum;
        receipt["layers"] = slice_names(model);
        receipt["architecture"] = model.architecture;
        receipt["directives"] = directive_dict(directives);
        return receipt;
    }

    static Loaded load(const std::string& target_blob, const std::string& journal_blob) {
        (void)journal_blob;
        stevejob::storage::AzuriteClient client;
        const std::string bytes = client.download_blob("models", bare_blob(target_blob, "models/"));
        Loaded loaded;
        loaded.model = stevejob::models::NeuralModel::import_document(stevejob::messaging::Sjc1Container::unpack(bytes));
        loaded.checksum = client.sha256_hex16(bytes);
        return loaded;
    }

    static void load_journal(stevejob::storage::AzuriteClient& client,
                             const std::string& journal_name,
                             stevejob::storage::SlidingJournal& journal) {
        try {
            const std::string text = client.download_blob("evals", journal_name);
            if (text.find("\"task\": \"regression\"") != std::string::npos) {
                journal.deserialize(text);
            }
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("HTTP 404") == std::string::npos) {
                throw;
            }
        }
    }

    static stevejob::models::RegressionScore score_loaded(const stevejob::models::NeuralModel& model) {
        if (model.architecture == "bottleneck_ae") {
            stevejob::models::BottleneckAutoencoder scorer;
            return scorer.score_network(model);
        }
        return model.evaluate_mse();
    }

    static stevejob::models::ModelDirective directives_for(const std::string& architecture) {
        if (architecture == "bottleneck_ae") {
            return stevejob::models::BottleneckAutoencoder{}.get_directives();
        }
        return stevejob::models::MlpRegressor{}.get_directives();
    }

    static std::string gradient_hint(stevejob::models::NeuralModel model, const std::string& slice_id) {
        if (model.architecture == "bottleneck_ae") {
            stevejob::models::BottleneckAutoencoder arch;
            arch.adopt(std::move(model));
            return arch.compute_slice_gradient(slice_id);
        }
        stevejob::models::MlpRegressor arch;
        arch.adopt(std::move(model));
        return arch.compute_slice_gradient(slice_id);
    }

    static py::list slice_names(const stevejob::models::NeuralModel& model) {
        py::list names;
        for (const stevejob::models::SliceRef& slice : model.get_slice_roster()) {
            names.append(slice.slice_id);
        }
        return names;
    }

    static py::dict directive_dict(const stevejob::models::ModelDirective& directives) {
        py::dict out;
        out["task"] = directives.task;
        out["model_type"] = directives.model_type;
        out["rules"] = directives.rules;
        return out;
    }

    static py::dict telemetry_dict(const stevejob::models::NeuralModel& model) {
        stevejob::models::MlpRegressor view;
        view.adopt(stevejob::models::NeuralModel(model));
        py::dict out;
        for (const auto& item : view.derive_training_telemetry()) {
            out[item.first.c_str()] = item.second;
        }
        return out;
    }

    static double telemetry_value(const stevejob::models::NeuralModel& model, const char* key) {
        stevejob::models::MlpRegressor view;
        view.adopt(stevejob::models::NeuralModel(model));
        const std::map<std::string, double> telemetry = view.derive_training_telemetry();
        const auto found = telemetry.find(key);
        return found == telemetry.end() ? 0.0 : found->second;
    }
};

}  // namespace stevejob::bridge
