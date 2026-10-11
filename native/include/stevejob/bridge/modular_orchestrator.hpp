#pragma once

#include "stevejob/messaging/sjc1_container.hpp"
#include "stevejob/messaging/totr_engine.hpp"
#include "stevejob/models/neural_model.hpp"
#include "stevejob/storage/azurite_client.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <pybind11/pybind11.h>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace stevejob::bridge {

class ModularOrchestrator {
public:
    static py::dict seed_model(stevejob::models::NeuralModel& model,
                               const std::string& target_blob,
                               const std::string& journal_blob) {
        std::lock_guard<std::mutex> lock(gate());
        const stevejob::models::RegressionScore score = model.evaluate_mse();
        const std::string packed = stevejob::messaging::Sjc1Container::pack(model.export_document());
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = bare_blob(target_blob, "models/");
        const std::string journal_name = bare_blob(journal_blob, "evals/");
        client.upload_blob("models", stored_name, packed, "application/octet-stream");
        const std::string checksum = client.sha256_hex16(packed);
        stevejob::storage::SlidingJournal journal(5);
        journal.add_entry(score.loss, score.max_residual, score.std_residual, checksum);
        client.upload_blob("evals", journal_name, journal.serialize(), "application/json");
        py::dict receipt;
        receipt["status"] = "seeded";
        receipt["loss"] = score.loss;
        receipt["max_residual"] = score.max_residual;
        receipt["checksum"] = checksum;
        receipt["layers"] = slice_names(model);
        return receipt;
    }

    static py::dict seed_shapes(const std::vector<std::vector<int>>& shapes,
                                const std::string& target_blob,
                                const std::string& journal_blob) {
        stevejob::models::NeuralModel model("stack");
        for (std::size_t index = 0; index < shapes.size(); ++index) {
            if (shapes[index].size() != 2 || shapes[index][0] < 1 || shapes[index][1] < 1) {
                throw std::invalid_argument("each layer shape is [rows, cols]");
            }
            const auto rows = static_cast<std::size_t>(shapes[index][0]);
            const auto cols = static_cast<std::size_t>(shapes[index][1]);
            const stevejob::models::ActivationType activation = index + 1 == shapes.size()
                                                                    ? stevejob::models::ActivationType::Linear
                                                                    : stevejob::models::ActivationType::ReLU;
            model.add_layer("layer" + std::to_string(index + 1), rows, cols, activation);
        }
        model.set_loss(stevejob::models::LossType::MeanSquaredError);
        model.initialize_default_weights();
        return seed_model(model, target_blob, journal_blob);
    }

    static py::dict read_meta(const std::string& checkpoint_blob, const std::string& journal_blob) {
        const Loaded loaded = load(checkpoint_blob, journal_blob);
        const stevejob::models::RegressionScore score = loaded.model.evaluate_mse();
        py::dict receipt;
        receipt["status"] = "ready";
        receipt["layers"] = slice_names(loaded.model);
        receipt["loss"] = score.loss;
        receipt["max_residual"] = score.max_residual;
        receipt["checksum"] = loaded.checksum;
        return receipt;
    }

    static py::dict read_slice(const std::string& checkpoint_blob,
                               const std::string& journal_blob,
                               const std::string& slice_id) {
        const Loaded loaded = load(checkpoint_blob, journal_blob);
        const stevejob::models::SliceRef slice = loaded.model.find_slice(slice_id);
        const stevejob::models::RegressionScore score = loaded.model.evaluate_mse();
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
        return receipt;
    }

    static py::dict commit_slice(const std::string& slice_id,
                                 const std::string& patch_totr_w,
                                 const std::string& patch_b,
                                 const std::string& target_blob,
                                 const std::string& journal_blob) {
        std::lock_guard<std::mutex> lock(gate());
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = bare_blob(target_blob, "models/");
        const std::string journal_name = bare_blob(journal_blob, "evals/");
        const std::string current = client.download_blob("models", stored_name);
        stevejob::models::NeuralModel model =
            stevejob::models::NeuralModel::import_document(stevejob::messaging::Sjc1Container::unpack(current));
        stevejob::storage::SlidingJournal journal = load_journal(client, journal_name);
        const double previous_loss = journal.size() > 0 ? journal.history().back().loss
                                                        : std::numeric_limits<double>::infinity();
        bool scored = false;
        bool divergent = false;
        stevejob::models::RegressionScore score{};
        try {
            const stevejob::models::SliceRef slice = model.find_slice(slice_id);
            const auto block = stevejob::messaging::TotrEngine::decode_matrix(patch_totr_w, slice.rows, slice.cols);
            const auto bias = stevejob::messaging::TotrEngine::decode_vector(patch_b, slice.cols);
            model.patch_slice(slice_id, block, bias);
            score = model.evaluate_mse();
            scored = true;
        } catch (const std::invalid_argument&) {
            divergent = true;
        } catch (const std::out_of_range&) {
            divergent = true;
        }
        double loss = 101.0;
        double max_residual = 0.0;
        double std_residual = 0.0;
        if (scored) {
            loss = score.loss;
            max_residual = score.max_residual;
            std_residual = score.std_residual;
            divergent = loss > 100.0;
        }
        const bool kept = scored && !divergent && !(previous_loss <= loss);
        std::string checksum;
        std::size_t bytes = 0;
        if (kept) {
            const std::string packed = stevejob::messaging::Sjc1Container::pack(model.export_document());
            client.upload_blob("models", stored_name, packed, "application/octet-stream");
            bytes = packed.size();
            checksum = client.sha256_hex16(packed);
            journal.add_entry(score.loss, score.max_residual, score.std_residual, checksum);
            client.upload_blob("evals", journal_name, journal.serialize(), "application/json");
        } else {
            bytes = current.size();
            checksum = client.sha256_hex16(current);
        }
        py::dict receipt;
        if (divergent) {
            receipt["status"] = "rejected_divergent";
        } else if (kept) {
            receipt["status"] = "committed_natively";
        } else {
            receipt["status"] = "rejected";
        }
        receipt["loss"] = loss;
        receipt["max_residual"] = max_residual;
        receipt["std_residual"] = std_residual;
        receipt["checksum"] = checksum;
        receipt["bytes"] = bytes;
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

    static py::list slice_names(const stevejob::models::NeuralModel& model) {
        py::list names;
        for (const stevejob::models::SliceRef& slice : model.get_slice_roster()) {
            names.append(slice.slice_id);
        }
        return names;
    }

    static Loaded load(const std::string& checkpoint_blob, const std::string& journal_blob) {
        (void)journal_blob;
        stevejob::storage::AzuriteClient client;
        const std::string bytes = client.download_blob("models", bare_blob(checkpoint_blob, "models/"));
        Loaded loaded;
        loaded.model = stevejob::models::NeuralModel::import_document(stevejob::messaging::Sjc1Container::unpack(bytes));
        loaded.checksum = client.sha256_hex16(bytes);
        return loaded;
    }

    static stevejob::storage::SlidingJournal load_journal(stevejob::storage::AzuriteClient& client,
                                                          const std::string& journal_name) {
        stevejob::storage::SlidingJournal journal(5);
        std::string text;
        try {
            text = client.download_blob("evals", journal_name);
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("HTTP 404") == std::string::npos) {
                throw;
            }
        }
        if (!text.empty() && text.find("\"task\": \"regression\"") != std::string::npos) {
            journal.deserialize(text);
        }
        return journal;
    }
};

}  // namespace stevejob::bridge
