#pragma once

#include "stevejob/messaging/sjc1_scheme.hpp"
#include "stevejob/models/architecture_catalog.hpp"
#include "stevejob/models/model_template.hpp"
#include "stevejob/storage/azurite_client.hpp"
#include "stevejob/storage/blob_path.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace stevejob::models {

struct LoadedCheckpoint {
    NeuralNetwork model;
    std::string checksum;
};

struct ModelView {
    ModelEvaluation score;
    ModelDirective directives;
    py::dict telemetry;
    py::dict candidate_scheme;
    py::list parameters;
    py::dict probe;
};

class ModelCheckpoint {
public:
    template <typename Model>
    static py::dict store(Model& model,
                          const std::string& target_blob,
                          const std::string& journal_blob,
                          std::size_t journal_capacity) {
        const ModelEvaluation score = model.evaluate_model();
        const std::string packed = model.container_scheme().pack(model.network());
        stevejob::storage::AzuriteClient client;
        const std::string stored_name = stevejob::storage::BlobPath::bare(target_blob, "models/");
        const std::string journal_name = stevejob::storage::BlobPath::bare(journal_blob, "evals/");
        client.upload_blob("models", stored_name, packed, "application/octet-stream");
        const std::string checksum = client.sha256_hex16(packed);
        model.bind_journal(journal_capacity);
        model.append_seed(score, checksum);
        client.upload_blob("evals", journal_name, model.journal().serialize(), "application/json");
        return render_checkpoint(model, model.checkpoint_document(score, checksum));
    }

    static LoadedCheckpoint open(const std::string& target_blob) {
        stevejob::storage::AzuriteClient client;
        const std::string bytes = client.download_blob("models", stevejob::storage::BlobPath::bare(target_blob, "models/"));
        LoadedCheckpoint loaded;
        loaded.model = ArchitectureCatalog::prepare(stevejob::messaging::SJC1Compressor::instance().unpack(bytes));
        loaded.checksum = client.sha256_hex16(bytes);
        return loaded;
    }

    static stevejob::storage::SlidingJournal open_journal(const std::string& journal_blob, std::size_t capacity) {
        stevejob::storage::SlidingJournal journal(capacity == 0 ? 1 : capacity);
        stevejob::storage::AzuriteClient client;
        read_journal(client, stevejob::storage::BlobPath::bare(journal_blob, "evals/"), journal);
        return journal;
    }

    static void read_journal(stevejob::storage::AzuriteClient& client,
                             const std::string& journal_name,
                             stevejob::storage::SlidingJournal& journal) {
        try {
            const std::string text = client.download_blob("evals", journal_name);
            if (text.find("\"history\"") != std::string::npos) {
                journal.deserialize(text);
            }
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("HTTP 404") == std::string::npos) {
                throw;
            }
        }
    }

    static ModelView inspect(const NeuralNetwork& network, const std::string& slice_id = "") {
        return ArchitectureCatalog::visit(NeuralNetwork(network), [&](auto& model) {
            ModelView view;
            view.score = model.evaluate_model();
            view.directives = model.get_directives();
            view.telemetry = telemetry_dict(model.derive_training_telemetry());
            view.candidate_scheme = scheme_dict(model.candidate_scheme());
            if (!slice_id.empty()) {
                const SliceRef slice = model.network().find_slice(slice_id);
                view.parameters = parameter_list(model, slice);
                const stevejob::core::DescribedField probe = model.derive_gradient_probe(slice_id);
                view.probe["key"] = probe.key;
                view.probe["token"] = probe.to_string();
                view.probe["format"] = probe.format_description;
                view.probe["rule"] = probe.actionable_rule;
            }
            return view;
        });
    }

    static py::dict ready_receipt(const LoadedCheckpoint& loaded, const ModelView& view) {
        py::dict receipt;
        receipt["status"] = "ready";
        receipt["layers"] = slice_names(loaded.model);
        write_metrics(receipt, view.score);
        receipt["checksum"] = loaded.checksum;
        receipt["architecture"] = loaded.model.architecture;
        receipt["directives"] = directive_dict(view.directives);
        receipt["telemetry"] = view.telemetry;
        receipt["candidate_scheme"] = view.candidate_scheme;
        return receipt;
    }

    static py::dict slice_receipt(const LoadedCheckpoint& loaded, const std::string& slice_id) {
        const SliceRef slice = loaded.model.find_slice(slice_id);
        const ModelView view = inspect(loaded.model, slice_id);
        py::dict receipt = ready_receipt(loaded, view);
        receipt["layer_name"] = slice.slice_id;
        receipt["in_dim"] = static_cast<int>(slice.rows);
        receipt["out_dim"] = static_cast<int>(slice.cols);
        receipt["parameters"] = view.parameters;
        receipt["probe"] = view.probe;
        return receipt;
    }

private:
    template <typename Model>
    static py::dict render_checkpoint(const Model& model, const CheckpointDocument& document) {
        py::dict receipt;
        receipt["status"] = document.status;
        for (const stevejob::storage::MetricField& field : document.fields) {
            if (std::holds_alternative<double>(field.value)) {
                receipt[py::str(field.name)] = std::get<double>(field.value);
            } else if (std::holds_alternative<std::int64_t>(field.value)) {
                receipt[py::str(field.name)] = std::get<std::int64_t>(field.value);
            } else {
                receipt[py::str(field.name)] = std::get<std::string>(field.value);
            }
        }
        if (document.include_layers) {
            receipt["layers"] = slice_names(model.network());
        }
        if (document.include_directives) {
            receipt["directives"] = directive_dict(model.get_directives());
        }
        return receipt;
    }

    static py::list slice_names(const NeuralNetwork& model) {
        py::list names;
        for (const SliceRef& slice : model.get_slice_roster()) {
            names.append(slice.slice_id);
        }
        return names;
    }

    static py::dict directive_dict(const ModelDirective& directives) {
        py::dict out;
        out["task"] = directives.task;
        out["model_type"] = directives.model_type;
        out["rules"] = directives.rules;
        return out;
    }

    static void write_metrics(py::dict& receipt, const ModelEvaluation& score) {
        receipt["loss"] = score.primary_loss;
        for (const stevejob::core::DescribedField& field : score.metrics) {
            if (field.key == "loss" || !std::holds_alternative<double>(field.value)) {
                continue;
            }
            receipt[field.key.c_str()] = std::get<double>(field.value);
        }
    }

    static py::dict scheme_dict(const CandidateScheme& scheme) {
        py::dict out;
        py::list slots;
        for (const std::string& slot : scheme.slots) {
            slots.append(slot);
        }
        py::list fields;
        for (const CandidateField& field : scheme.fields) {
            py::dict row;
            row["key"] = field.key;
            row["role"] = field.role;
            fields.append(row);
        }
        py::list groups;
        for (const std::string& group : scheme.groups) {
            groups.append(group);
        }
        out["slots"] = slots;
        out["fields"] = fields;
        out["groups"] = groups;
        return out;
    }

    template <typename Model>
    static py::list parameter_list(Model& model, const SliceRef& slice) {
        py::list rows;
        for (const CandidateField& field : model.candidate_scheme().fields) {
            py::dict row;
            row["key"] = field.key;
            row["role"] = field.role;
            if (field.role == "weights") {
                row["scheme"] = model.get_scheme_for_slice(slice).get_descriptor().scheme_name;
                row["token"] = model.serialize_slice(slice);
            } else if (field.role == "bias") {
                row["scheme"] = model.get_bias_scheme(slice).get_descriptor().scheme_name;
                row["token"] = model.serialize_bias(slice);
            } else {
                continue;
            }
            rows.append(row);
        }
        return rows;
    }

    static py::dict telemetry_dict(const std::vector<TelemetrySignal>& signals) {
        py::dict out;
        for (const TelemetrySignal& signal : signals) {
            if (std::holds_alternative<double>(signal.field.value)) {
                out[signal.field.key.c_str()] = std::get<double>(signal.field.value);
            }
        }
        return out;
    }
};

}  // namespace stevejob::models
