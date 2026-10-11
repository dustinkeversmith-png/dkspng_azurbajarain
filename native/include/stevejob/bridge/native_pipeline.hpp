#pragma once

#include "stevejob/messaging/compressor.hpp"
#include "stevejob/messaging/totr_engine.hpp"
#include "stevejob/models/regression_mlp_model.hpp"
#include "stevejob/storage/json_text.hpp"
#include "stevejob/storage/native_blob_client.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <pybind11/pybind11.h>

#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

namespace py = pybind11;

namespace stevejob::bridge {

inline const stevejob::messaging::LayerRecord& require_layer(const stevejob::messaging::ModelData& model,
                                                             const char* name) {
    for (const stevejob::messaging::LayerRecord& layer : model.layers) {
        if (layer.name == name) {
            return layer;
        }
    }
    throw std::invalid_argument(std::string(name) + " is missing from the checkpoint");
}

inline double journal_loss(const std::string& payload) {
    const stevejob::storage::JsonText parsed = stevejob::storage::JsonText::parse(payload);
    if (parsed.kind != stevejob::storage::JsonText::Kind::Object) {
        throw std::invalid_argument("journal must be a JSON object");
    }
    for (const auto& item : parsed.object) {
        if (item.first == "mean_squared_error" && item.second.kind == stevejob::storage::JsonText::Kind::Number) {
            return item.second.number;
        }
    }
    throw std::invalid_argument("journal is missing mean_squared_error");
}

inline py::object json_to_py(const stevejob::storage::JsonText& value) {
    using Kind = stevejob::storage::JsonText::Kind;
    if (value.kind == Kind::Null) {
        return py::none();
    }
    if (value.kind == Kind::Bool) {
        return py::bool_(value.boolean);
    }
    if (value.kind == Kind::Number) {
        if (value.number_is_int) {
            return py::int_(value.integer);
        }
        return py::float_(value.number);
    }
    if (value.kind == Kind::String) {
        return py::str(value.text);
    }
    if (value.kind == Kind::Array) {
        py::list items;
        for (const stevejob::storage::JsonText& item : value.array) {
            items.append(json_to_py(item));
        }
        return items;
    }
    py::dict mapping;
    for (const auto& item : value.object) {
        mapping[py::str(item.first)] = json_to_py(item.second);
    }
    return mapping;
}

inline void append_json_string(std::string& out, const std::string& text) {
    out.push_back('"');
    for (unsigned char character : text) {
        switch (character) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (character < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(character));
                }
        }
    }
    out.push_back('"');
}

inline void dump_json_into(std::string& out, py::handle value, int indent);

inline void dump_json_into(std::string& out, py::handle value, int indent) {
    const std::string pad(static_cast<std::size_t>(indent) * 2, ' ');
    const std::string nested(static_cast<std::size_t>(indent + 1) * 2, ' ');
    if (value.is_none()) {
        out += "null";
        return;
    }
    if (py::isinstance<py::bool_>(value)) {
        out += value.cast<bool>() ? "true" : "false";
        return;
    }
    if (py::isinstance<py::int_>(value)) {
        out += std::to_string(value.cast<long long>());
        return;
    }
    if (py::isinstance<py::float_>(value)) {
        std::ostringstream number;
        number.precision(17);
        number << value.cast<double>();
        out += number.str();
        return;
    }
    if (py::isinstance<py::str>(value)) {
        append_json_string(out, value.cast<std::string>());
        return;
    }
    if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
        py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
        if (py::len(sequence) == 0) {
            out += "[]";
            return;
        }
        out += "[\n";
        bool first = true;
        for (py::handle item : sequence) {
            if (!first) {
                out += ",\n";
            }
            first = false;
            out += nested;
            dump_json_into(out, item, indent + 1);
        }
        out += "\n";
        out += pad;
        out += "]";
        return;
    }
    if (py::isinstance<py::dict>(value)) {
        py::dict mapping = py::reinterpret_borrow<py::dict>(value);
        if (py::len(mapping) == 0) {
            out += "{}";
            return;
        }
        out += "{\n";
        bool first = true;
        for (auto item : mapping) {
            if (!first) {
                out += ",\n";
            }
            first = false;
            out += nested;
            append_json_string(out, py::cast<std::string>(py::str(item.first)));
            out += ": ";
            dump_json_into(out, py::reinterpret_borrow<py::object>(item.second), indent + 1);
        }
        out += "\n";
        out += pad;
        out += "}";
        return;
    }
    throw std::invalid_argument("JSON payload must contain objects, arrays, strings, and numbers");
}

inline py::object storage_json_load(const std::string& raw) {
    return json_to_py(stevejob::storage::JsonText::parse(raw));
}

inline std::string storage_json_dump(py::handle value) {
    if (!py::isinstance<py::dict>(value)) {
        throw std::invalid_argument("JSON payload must be an object");
    }
    std::string out;
    dump_json_into(out, value, 0);
    out.push_back('\n');
    return out;
}

inline std::string content_checksum(const std::string& payload) {
    return stevejob::storage::NativeBlobClient("UseDevelopmentStorage=true").sha256_hex16(payload);
}

inline std::string blob_exchange(const std::string& connection,
                                 const std::string& method,
                                 const std::string& path,
                                 const std::vector<stevejob::storage::QueryItem>& query,
                                 const std::string& body,
                                 const std::string& content_type) {
    return stevejob::storage::NativeBlobClient(connection).exchange(method, path, query, body, content_type);
}

inline py::dict read_state_native(const std::string& checkpoint_blob, const std::string& journal_blob) {
    std::string l1_w;
    std::string l2_w;
    std::string l1_b;
    std::string l2_b;
    std::string checksum;
    std::string journal_checksum;
    std::size_t bytes = 0;
    std::size_t journal_bytes = 0;
    double mse = 0.0;
    double max_residual = 0.0;
    double std_residual = 0.0;
    {
        py::gil_scoped_release release;
        stevejob::storage::NativeBlobClient client;
        const std::string model_bytes = client.download_blob("models", checkpoint_blob);
        const std::string journal = client.download_blob("evals", journal_blob);
        const stevejob::messaging::DecodeResult decoded = stevejob::messaging::TensorCompressor::decode(model_bytes);
        const stevejob::messaging::LayerRecord& first = require_layer(decoded.model, "layer1");
        const stevejob::messaging::LayerRecord& second = require_layer(decoded.model, "layer2");
        l1_w = stevejob::messaging::TotrEngine::encode_matrix(first.weights, 1);
        l2_w = stevejob::messaging::TotrEngine::encode_matrix(second.weights, 1);
        l1_b = stevejob::messaging::TotrEngine::encode_vector(first.biases);
        l2_b = stevejob::messaging::TotrEngine::encode_vector(second.biases);
        bytes = model_bytes.size();
        journal_bytes = journal.size();
        checksum = client.sha256_hex16(model_bytes);
        stevejob::storage::SlidingJournal parsed(5);
        parsed.deserialize(journal);
        if (parsed.size() == 0) {
            throw std::invalid_argument("journal has no entries");
        }
        const stevejob::storage::JournalEntry& latest = parsed.history().back();
        mse = latest.loss;
        max_residual = latest.max_residual;
        std_residual = latest.std_residual;
        journal_checksum = latest.checksum;
    }
    py::dict result;
    result["status"] = "ready";
    result["l1_w"] = l1_w;
    result["l2_w"] = l2_w;
    result["l1_b"] = l1_b;
    result["l2_b"] = l2_b;
    result["bytes"] = bytes;
    result["journal_bytes"] = journal_bytes;
    result["checksum"] = checksum;
    result["loss"] = mse;
    result["max_residual"] = max_residual;
    result["std_residual"] = std_residual;
    result["journal_checksum"] = journal_checksum;
    return result;
}

inline py::dict score_totr_candidate(const std::string& l1_w,
                                      const std::string& l2_w,
                                      const std::string& l1_b,
                                      const std::string& l2_b) {
    double loss = 0.0;
    double max_residual = 0.0;
    double std_residual = 0.0;
    {
        py::gil_scoped_release release;
        const auto first_weights = stevejob::messaging::TotrEngine::decode_matrix(l1_w, 2, 4);
        const auto second_weights = stevejob::messaging::TotrEngine::decode_matrix(l2_w, 4, 1);
        const auto first_bias = stevejob::messaging::TotrEngine::decode_vector(l1_b, 4);
        const auto second_bias = stevejob::messaging::TotrEngine::decode_vector(l2_b, 1);
        stevejob::models::RegressionMLPModel model(first_weights, first_bias, second_weights, second_bias);
        const stevejob::models::RegressionScore score = model.evaluate_mse(50);
        loss = score.loss;
        max_residual = score.max_residual;
        std_residual = score.std_residual;
    }
    py::dict receipt;
    receipt["status"] = "scored";
    receipt["loss"] = loss;
    receipt["max_residual"] = max_residual;
    receipt["std_residual"] = std_residual;
    return receipt;
}

inline py::dict commit_winner_native(const std::string& winning_comp_l1_w,
                                     const std::string& winning_comp_l2_w,
                                     const std::string& winning_l1_b,
                                     const std::string& winning_l2_b,
                                     const std::string& target_blob,
                                     const std::string& journal_blob = "regression_journal.json") {
    static std::mutex commit_gate;
    double loss = 0.0;
    double max_residual = 0.0;
    double std_residual = 0.0;
    double previous_loss = std::numeric_limits<double>::infinity();
    std::size_t bytes = 0;
    std::string checksum;
    std::string stored_name;
    bool kept = false;
    bool divergent = false;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(commit_gate);
        bool scored = false;
        stevejob::models::RegressionScore score{};
        std::vector<std::vector<double>> first_weights;
        std::vector<double> first_bias;
        std::vector<std::vector<double>> second_weights;
        std::vector<double> second_bias;
        try {
            first_weights = stevejob::messaging::TotrEngine::decode_matrix(winning_comp_l1_w, 2, 4);
            second_weights = stevejob::messaging::TotrEngine::decode_matrix(winning_comp_l2_w, 4, 1);
            first_bias = stevejob::messaging::TotrEngine::decode_vector(winning_l1_b, 4);
            second_bias = stevejob::messaging::TotrEngine::decode_vector(winning_l2_b, 1);
            stevejob::models::RegressionMLPModel model(first_weights, first_bias, second_weights, second_bias);
            score = model.evaluate_mse(50);
            scored = true;
        } catch (const std::invalid_argument&) {
            divergent = true;
        }
        stevejob::storage::NativeBlobClient client;
        stored_name = target_blob;
        const std::string model_prefix = "models/";
        if (stored_name.rfind(model_prefix, 0) == 0) {
            stored_name = stored_name.substr(model_prefix.size());
        }
        std::string journal_name = journal_blob;
        const std::string eval_prefix = "evals/";
        if (journal_name.rfind(eval_prefix, 0) == 0) {
            journal_name = journal_name.substr(eval_prefix.size());
        }
        std::string existing;
        try {
            existing = client.download_blob("evals", journal_name);
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("HTTP 404") == std::string::npos) {
                throw;
            }
        }
        stevejob::storage::SlidingJournal journal(5);
        const bool regression_journal = existing.find("\"task\": \"regression\"") != std::string::npos;
        if (regression_journal && !existing.empty()) {
            journal.deserialize(existing);
        }
        if (journal.size() > 0) {
            previous_loss = journal.history().back().loss;
        }
        if (scored) {
            loss = score.loss;
            max_residual = score.max_residual;
            std_residual = score.std_residual;
            divergent = loss > 100.0;
        } else {
            loss = 101.0;
        }
        kept = scored && !divergent && !(previous_loss <= loss);
        if (kept) {
            stevejob::messaging::ModelData document;
            document.layers.push_back(stevejob::messaging::LayerRecord{"layer1", first_weights, first_bias});
            document.layers.push_back(stevejob::messaging::LayerRecord{"layer2", second_weights, second_bias});
            const std::string packed = stevejob::messaging::TensorCompressor::pack(document);
            client.upload_blob("models", stored_name, packed, "application/octet-stream");
            bytes = packed.size();
            checksum = client.sha256_hex16(packed);
            journal.add_entry(score.loss, score.max_residual, score.std_residual, checksum);
            client.upload_blob("evals", journal_name, journal.serialize(), "application/json");
        } else {
            try {
                const std::string current = client.download_blob("models", stored_name);
                bytes = current.size();
                checksum = client.sha256_hex16(current);
            } catch (const std::runtime_error&) {
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
    receipt["loss"] = loss;
    receipt["max_residual"] = max_residual;
    receipt["std_residual"] = std_residual;
    if (previous_loss < std::numeric_limits<double>::max()) {
        receipt["previous_loss"] = previous_loss;
    }
    receipt["blob"] = "models/" + stored_name;
    receipt["bytes"] = bytes;
    receipt["checksum"] = checksum;
    return receipt;
}

}  // namespace stevejob::bridge
