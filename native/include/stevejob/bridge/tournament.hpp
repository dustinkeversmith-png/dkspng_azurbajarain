#pragma once

#include "stevejob/messaging/sjc1_scheme.hpp"
#include "stevejob/messaging/totr_engine.hpp"
#include "stevejob/models/architecture_catalog.hpp"
#include "stevejob/models/model_checkpoint.hpp"
#include "stevejob/storage/azurite_client.hpp"
#include "stevejob/storage/blob_path.hpp"

#include <pybind11/pybind11.h>

#include <limits>
#include <stdexcept>
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

class Tournament {
public:
    static py::dict commit(const std::string& slice_id,
                           const std::vector<TournamentCandidate>& candidates,
                           const std::string& target_blob,
                           const std::string& journal_blob,
                           std::size_t journal_capacity) {
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = stevejob::storage::BlobPath::bare(target_blob, "models/");
        const std::string journal_name = stevejob::storage::BlobPath::bare(journal_blob, "evals/");
        const std::string current = client.download_blob("models", stored_name);
        stevejob::models::NeuralNetwork baseline =
            stevejob::models::ArchitectureCatalog::prepare(stevejob::messaging::SJC1Compressor::instance().unpack(current));
        stevejob::storage::SlidingJournal journal(journal_capacity == 0 ? 1 : journal_capacity);
        stevejob::models::ModelCheckpoint::read_journal(client, journal_name, journal);

        Board board;
        board.previous_loss = journal.committed_loss();
        board.best_model = baseline;
        const stevejob::models::SliceRef active_slice = baseline.find_slice(slice_id);
        const std::string active_token = baseline.slice_token(active_slice);
        const std::string active_bias = baseline.slice_bias(active_slice);
        const bool dead_probe = probe_is_dead(baseline, slice_id);
        for (const TournamentCandidate& candidate : candidates) {
            board.consider(baseline, candidate, slice_id, active_token, active_bias, active_slice.cols, dead_probe);
        }
        return board.publish(client, journal, stored_name, journal_name, current, slice_id, active_token);
    }

    static py::dict score_layers(const std::vector<std::string>& weight_tokens,
                                 const std::vector<std::string>& bias_tokens,
                                 const std::vector<std::vector<int>>& shapes) {
        const stevejob::models::ModelEvaluation score = evaluate(assemble(weight_tokens, bias_tokens, shapes));
        py::dict receipt;
        receipt["status"] = "scored";
        write_metrics(receipt, score);
        return receipt;
    }

    static py::dict commit_layers(const std::vector<std::string>& weight_tokens,
                                  const std::vector<std::string>& bias_tokens,
                                  const std::vector<std::vector<int>>& shapes,
                                  const std::string& target_blob,
                                  const std::string& journal_blob,
                                  std::size_t journal_capacity) {
        bool scored = false;
        bool divergent = false;
        stevejob::models::NeuralNetwork candidate;
        stevejob::models::ModelEvaluation score{};
        try {
            candidate = assemble(weight_tokens, bias_tokens, shapes);
            score = evaluate(candidate);
            scored = true;
            divergent = score.primary_loss > 100.0;
        } catch (const std::invalid_argument&) {
            divergent = true;
        }
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = stevejob::storage::BlobPath::bare(target_blob, "models/");
        const std::string journal_name = stevejob::storage::BlobPath::bare(journal_blob, "evals/");
        stevejob::storage::SlidingJournal journal(journal_capacity == 0 ? 1 : journal_capacity);
        stevejob::models::ModelCheckpoint::read_journal(client, journal_name, journal);
        const double previous_loss = journal.committed_loss();
        const bool kept = scored && !divergent && (previous_loss - score.primary_loss > 1e-4);
        std::size_t bytes = 0;
        std::string checksum;
        if (kept) {
            const std::string packed = stevejob::messaging::SJC1Compressor::instance().pack(candidate);
            client.upload_blob("models", stored_name, packed, "application/octet-stream");
            bytes = packed.size();
            checksum = client.sha256_hex16(packed);
            const double delta = previous_loss < std::numeric_limits<double>::max() ? score.primary_loss - previous_loss : 0.0;
            stevejob::models::TrialNote note;
            note.status = "committed_natively";
            note.loss = score.primary_loss;
            note.delta = delta;
            note.checksum = checksum;
            note.score = &score;
            const stevejob::storage::JournalRecord record = stevejob::models::ArchitectureCatalog::visit(
                stevejob::models::NeuralNetwork(candidate),
                [&](auto& model) { return model.trial_record(note); });
            journal.append_record(record);
            client.upload_blob("evals", journal_name, journal.serialize(), "application/json");
        } else {
            try {
                const std::string current = client.download_blob("models", stored_name);
                bytes = current.size();
                checksum = client.sha256_hex16(current);
            } catch (const std::runtime_error& error) {
                if (std::string(error.what()).find("HTTP 404") == std::string::npos) {
                    throw;
                }
                if (journal.size() > 0) {
                    checksum = journal.history().back().checksum;
                }
            }
        }
        py::dict receipt;
        if (divergent) {
            receipt["status"] = "rejected_divergent";
        } else if (kept) {
            receipt["status"] = "committed_natively";
        } else {
            receipt["status"] = "rejected";
        }
        if (scored) {
            write_metrics(receipt, score);
        } else {
            receipt["loss"] = 101.0;
        }
        if (previous_loss < std::numeric_limits<double>::max()) {
            receipt["previous_loss"] = previous_loss;
        }
        receipt["blob"] = "models/" + stored_name;
        receipt["bytes"] = bytes;
        receipt["checksum"] = checksum;
        return receipt;
    }

    static py::dict read_checkpoint(const std::string& checkpoint_blob, const std::string& journal_blob) {
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = stevejob::storage::BlobPath::bare(checkpoint_blob, "models/");
        const std::string journal_name = stevejob::storage::BlobPath::bare(journal_blob, "evals/");
        const std::string model_bytes = client.download_blob("models", stored_name);
        const std::string journal_text = client.download_blob("evals", journal_name);
        stevejob::models::NeuralNetwork network =
            stevejob::models::ArchitectureCatalog::prepare(
                stevejob::messaging::SJC1Compressor::instance().unpack(model_bytes));
        stevejob::storage::SlidingJournal journal(5);
        journal.deserialize(journal_text);
        if (journal.size() == 0) {
            throw std::invalid_argument("journal has no entries");
        }
        const stevejob::storage::JournalRecord& latest = journal.history().back();
        py::dict result;
        result["status"] = "ready";
        result["bytes"] = model_bytes.size();
        result["journal_bytes"] = journal_text.size();
        result["checksum"] = client.sha256_hex16(model_bytes);
        result["loss"] = latest.loss;
        result["journal_checksum"] = latest.checksum;
        for (const stevejob::storage::MetricField& field : latest.metrics) {
            if (std::holds_alternative<double>(field.value)) {
                result[py::str(field.name)] = std::get<double>(field.value);
            }
        }
        stevejob::models::ArchitectureCatalog::visit(std::move(network), [&](auto& model) {
            const std::vector<stevejob::models::LayerSpec>& layers = model.network().layers;
            for (std::size_t index = 0; index < layers.size(); ++index) {
                const stevejob::models::LayerSpec& layer = layers[index];
                const stevejob::models::SliceRef slice{layer.name, layer.name, index, 0, 0, layer.in_dim, layer.out_dim};
                const std::string prefix = "l" + std::to_string(index + 1);
                result[py::str(prefix + "_w")] = model.get_scheme_for_slice(slice).serialize_matrix(layer.weights);
                result[py::str(prefix + "_b")] =
                    model.get_bias_scheme(slice).serialize_vector(layer.biases, 0, layer.biases.size());
            }
            return 0;
        });
        return result;
    }

    static std::string winner_step(const std::string& token, const std::string& active) {
        if (token.empty()) {
            return "";
        }
        const std::size_t pos = token.find("g:");
        std::string gamma = "?";
        if (pos != std::string::npos) {
            const std::size_t begin = pos + 2;
            const std::size_t end = token.find('|', begin);
            gamma = token.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        }
        int distance = 0;
        try {
            distance = stevejob::messaging::TotrEngine::integer_distance(token, active);
        } catch (const std::invalid_argument&) {
            distance = 0;
        }
        return "g=" + gamma + " d=" + std::to_string(distance);
    }

private:
    struct Judgement {
        bool unchanged = false;
        stevejob::models::NeuralNetwork trial;
        stevejob::models::ModelEvaluation score;
    };

    struct Board {
        double previous_loss = std::numeric_limits<double>::infinity();
        double best_loss = std::numeric_limits<double>::infinity();
        std::string winner = "none";
        std::string winner_token;
        stevejob::models::NeuralNetwork best_model;
        stevejob::models::ModelEvaluation best_score{};
        bool any_scored = false;
        bool any_divergent = false;
        bool any_unchanged = false;
        bool any_zero = false;
        py::list report;

        void consider(const stevejob::models::NeuralNetwork& baseline,
                      const TournamentCandidate& candidate,
                      const std::string& slice_id,
                      const std::string& active_token,
                      const std::string& active_bias,
                      std::size_t bias_count,
                      bool dead_probe) {
            py::dict row;
            row["name"] = candidate.name;
            int distance = 0;
            try {
                distance = stevejob::messaging::TotrEngine::integer_distance(candidate.weights, active_token);
            } catch (const std::invalid_argument&) {
                mark_divergent(row, 0);
                return;
            }
            row["distance"] = distance;
            if (distance < 2 && !(dead_probe && bias_raised(candidate.bias, active_bias, bias_count))) {
                row["status"] = "rejected_zero_mutation";
                any_zero = true;
                report.append(row);
                return;
            }
            try {
                const Judgement judged = judge(baseline, slice_id, candidate.weights, candidate.bias, active_token, active_bias);
                if (judged.unchanged) {
                    row["status"] = "unchanged";
                    any_unchanged = true;
                    report.append(row);
                    return;
                }
                row["loss"] = judged.score.primary_loss;
                row["status"] = judged.score.primary_loss > 100.0 ? "rejected_divergent" : "scored";
                if (judged.score.primary_loss > 100.0) {
                    any_divergent = true;
                } else if (judged.score.primary_loss < best_loss) {
                    best_loss = judged.score.primary_loss;
                    best_score = judged.score;
                    best_model = judged.trial;
                    winner = candidate.name;
                    winner_token = candidate.weights;
                    any_scored = true;
                }
            } catch (const std::invalid_argument&) {
                mark_divergent(row, distance);
                return;
            }
            report.append(row);
        }

        py::dict publish(stevejob::storage::AzuriteClient& client,
                         stevejob::storage::SlidingJournal& journal,
                         const std::string& stored_name,
                         const std::string& journal_name,
                         const std::string& current,
                         const std::string& slice_id,
                         const std::string& active_token) const {
            constexpr double kMinCommitDrop = 1e-4;
            const bool kept = any_scored && (previous_loss - best_loss > kMinCommitDrop);
            std::string checksum = client.sha256_hex16(current);
            const std::string status = outcome(kept);
            if (kept) {
                const std::string packed = stevejob::messaging::SJC1Compressor::instance().pack(best_model);
                client.upload_blob("models", stored_name, packed, "application/octet-stream");
                checksum = client.sha256_hex16(packed);
            }
            const double delta = any_scored && previous_loss < std::numeric_limits<double>::max() ? best_loss - previous_loss : 0.0;
            const double published_loss = any_scored ? best_loss : (any_divergent && !any_unchanged ? 101.0 : previous_loss);
            if (status != "rejected_zero_mutation") {
                journal.append_record(trial_record(status, published_loss, delta, checksum, slice_id, winner_token, active_token));
                client.upload_blob("evals", journal_name, journal.serialize(), "application/json");
            }
            py::dict receipt;
            receipt["status"] = status;
            receipt["winner"] = winner;
            receipt["winner_step"] = Tournament::winner_step(winner_token, active_token);
            receipt["loss"] = published_loss;
            receipt["delta"] = delta;
            receipt["checksum"] = checksum;
            receipt["candidates"] = report;
            return receipt;
        }

        void mark_divergent(py::dict row, int distance) {
            row["loss"] = 101.0;
            row["status"] = "rejected_divergent";
            row["distance"] = distance;
            any_divergent = true;
            report.append(row);
        }

        std::string outcome(bool kept) const {
            if (!any_scored && any_zero && !any_divergent) {
                return "rejected_zero_mutation";
            }
            if (!any_scored) {
                return any_divergent && !any_unchanged ? "rejected_divergent" : "rejected";
            }
            return kept ? "committed_natively" : "rejected";
        }

        stevejob::storage::JournalRecord trial_record(const std::string& status,
                                                      double published_loss,
                                                      double delta,
                                                      const std::string& checksum,
                                                      const std::string& slice_id,
                                                      const std::string& token,
                                                      const std::string& active_token) const {
            stevejob::models::TrialNote note;
            note.status = status;
            note.loss = published_loss;
            note.delta = delta;
            note.checksum = checksum;
            note.slice_id = slice_id;
            note.winner = winner;
            note.winner_step = Tournament::winner_step(token, active_token);
            note.record_winner = true;
            note.record_step = true;
            note.record_slice = true;
            if (status == "committed_natively") {
                note.score = &best_score;
            }
            return stevejob::models::ArchitectureCatalog::visit(stevejob::models::NeuralNetwork(best_model), [&](auto& model) {
                return model.trial_record(note);
            });
        }
    };

    static void write_metrics(py::dict& receipt, const stevejob::models::ModelEvaluation& score) {
        receipt["loss"] = score.primary_loss;
        for (const stevejob::core::DescribedField& field : score.metrics) {
            if (field.key == "loss" || !std::holds_alternative<double>(field.value)) {
                continue;
            }
            receipt[py::str(field.key)] = std::get<double>(field.value);
        }
    }

    static stevejob::models::NeuralNetwork assemble(const std::vector<std::string>& weight_tokens,
                                                  const std::vector<std::string>& bias_tokens,
                                                  const std::vector<std::vector<int>>& shapes) {
        if (weight_tokens.size() != bias_tokens.size() || weight_tokens.size() != shapes.size() || weight_tokens.empty()) {
            throw std::invalid_argument("each layer needs a weight token, a bias token, and a shape");
        }
        return stevejob::models::ArchitectureCatalog::visit(stevejob::models::NeuralNetwork::from_shapes(shapes), [&](auto& model) {
            const std::vector<stevejob::models::SliceRef> roster = model.network().get_slice_roster();
            if (roster.size() != weight_tokens.size()) {
                throw std::invalid_argument("layer tokens do not match the model slices");
            }
            for (std::size_t index = 0; index < weight_tokens.size(); ++index) {
                model.apply_slice_patch(roster[index], weight_tokens[index], bias_tokens[index]);
            }
            return stevejob::models::NeuralNetwork(model.network());
        });
    }

    static stevejob::models::ModelEvaluation evaluate(const stevejob::models::NeuralNetwork& network) {
        return stevejob::models::ArchitectureCatalog::visit(stevejob::models::NeuralNetwork(network), [&](auto& model) {
            return model.evaluate_model();
        });
    }

    static bool probe_is_dead(const stevejob::models::NeuralNetwork& network, const std::string& slice_id) {
        const std::string hint = stevejob::models::ArchitectureCatalog::visit(
            stevejob::models::NeuralNetwork(network),
            [&](auto& model) { return model.derive_gradient_probe(slice_id).to_string(); });
        return hint.find('+') == std::string::npos && hint.find('-') == std::string::npos;
    }

    static Judgement judge(const stevejob::models::NeuralNetwork& baseline,
                           const std::string& slice_id,
                           const std::string& weights,
                           const std::string& bias,
                           const std::string& active_token,
                           const std::string& active_bias) {
        return stevejob::models::ArchitectureCatalog::visit(stevejob::models::NeuralNetwork(baseline), [&](auto& model) {
            const stevejob::models::SliceRef slice = model.network().find_slice(slice_id);
            const stevejob::messaging::IEncodingScheme& weight_scheme = model.get_scheme_for_slice(slice);
            const stevejob::messaging::IEncodingScheme& bias_scheme = model.get_bias_scheme(slice);
            const auto block = weight_scheme.deserialize_matrix(weights, slice.rows, slice.cols);
            const auto bias_values = bias_scheme.deserialize_vector(bias, slice.cols);
            const auto active_block = weight_scheme.deserialize_matrix(active_token, slice.rows, slice.cols);
            const auto active_bias_values = bias_scheme.deserialize_vector(active_bias, slice.cols);
            Judgement judged;
            if (block == active_block && bias_values == active_bias_values) {
                judged.unchanged = true;
                return judged;
            }
            model.apply_slice_patch(slice, weights, bias);
            judged.trial = model.network();
            judged.score = model.score_patch(judged.trial);
            return judged;
        });
    }

    static bool bias_raised(const std::string& updated, const std::string& active, std::size_t count) {
        try {
            const auto next = stevejob::messaging::BiasDeltaScheme::instance().deserialize_vector(updated, count);
            const auto previous = stevejob::messaging::BiasDeltaScheme::instance().deserialize_vector(active, count);
            for (std::size_t index = 0; index < count; ++index) {
                if (next[index] >= previous[index] + 0.15) {
                    return true;
                }
            }
        } catch (const std::invalid_argument&) {
            return false;
        }
        return false;
    }
};

}  // namespace stevejob::bridge
