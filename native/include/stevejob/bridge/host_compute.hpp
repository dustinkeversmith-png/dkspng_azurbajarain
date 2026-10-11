#pragma once

#include "stevejob/math/linear_algebra.hpp"
#include "stevejob/math/tensor_stats.hpp"
#include "stevejob/messaging/compressor.hpp"
#include "stevejob/models/ring_mlp_model.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace stevejob::bridge {

inline void fail(const std::string& message) {
    throw std::invalid_argument(message);
}

inline py::object value_or(const py::dict& mapping, const char* key, py::object fallback) {
    if (!mapping.contains(key)) {
        return fallback;
    }
    return py::reinterpret_borrow<py::object>(mapping[key]);
}

inline py::dict as_dict(py::handle value, const std::string& message) {
    if (!py::isinstance<py::dict>(value)) {
        fail(message);
    }
    return py::reinterpret_borrow<py::dict>(value);
}

inline bool is_sequence(py::handle value) {
    return py::isinstance<py::sequence>(value) && !py::isinstance<py::str>(value) &&
           !py::isinstance<py::bytes>(value);
}

inline py::list coerce_vector(py::handle value, const std::string& label, bool allow_none) {
    if (value.is_none()) {
        if (allow_none) {
            return py::list();
        }
        fail(label + " must be a list");
    }
    if (!py::isinstance<py::list>(value)) {
        fail(label + " must be a list");
    }
    py::list out;
    for (py::handle item : value) {
        out.append(py::cast<double>(item));
    }
    return out;
}

inline py::list coerce_matrix(py::handle value, const std::string& label, bool allow_none) {
    if (value.is_none()) {
        if (allow_none) {
            return py::list();
        }
        fail(label + " must be a matrix");
    }
    if (!py::isinstance<py::list>(value)) {
        fail(label + " must be a matrix");
    }
    py::list matrix;
    for (py::handle row : value) {
        if (!is_sequence(row)) {
            fail(label + " must be a matrix");
        }
        py::list out_row;
        for (py::handle item : row) {
            out_row.append(py::cast<double>(item));
        }
        matrix.append(out_row);
    }
    return matrix;
}

inline py::object field_or(py::handle layer, const char* primary, const char* fallback) {
    py::dict object = as_dict(layer, "layer must be an object");
    if (object.contains(primary)) {
        return object[primary];
    }
    if (fallback != nullptr && object.contains(fallback)) {
        return object[fallback];
    }
    return py::none();
}

inline std::vector<double> collect_parameters(py::handle layers_value, bool enforce, std::size_t max_parameters) {
    py::dict layers = as_dict(layers_value, "checkpoint must be an object");
    std::vector<double> values;
    for (auto item : layers) {
        const std::string name = py::cast<std::string>(py::str(item.first));
        if (!py::isinstance<py::dict>(item.second)) {
            fail("layer " + name + " must be an object with weights");
        }
        py::dict layer = py::reinterpret_borrow<py::dict>(item.second);
        py::object weights = value_or(layer, "weights", py::list());
        if (!py::isinstance<py::list>(weights)) {
            fail("layer " + name + " weights must be a matrix");
        }
        for (py::handle row : weights) {
            if (!is_sequence(row)) {
                fail("layer " + name + " weights must be a matrix");
            }
            for (py::handle entry : row) {
                values.push_back(py::cast<double>(entry));
            }
        }
        py::object biases = value_or(layer, "biases", py::list());
        if (!py::isinstance<py::list>(biases)) {
            fail("layer " + name + " biases must be a list");
        }
        for (py::handle entry : biases) {
            values.push_back(py::cast<double>(entry));
        }
    }
    if (enforce && values.empty()) {
        fail("checkpoint has no parameters");
    }
    if (enforce && values.size() > max_parameters) {
        fail("checkpoint has " + std::to_string(values.size()) +
             " parameters; this run allows at most " + std::to_string(max_parameters));
    }
    return values;
}

inline py::dict checkpoint_summary(py::handle layers, std::size_t max_parameters) {
    const std::vector<double> values = collect_parameters(layers, true, max_parameters);
    double total = 0.0;
    double low = values[0];
    double high = values[0];
    for (double value : values) {
        total += value;
        if (value < low) {
            low = value;
        }
        if (value > high) {
            high = value;
        }
    }
    const double count = static_cast<double>(values.size());
    py::dict summary;
    summary["parameter_count"] = values.size();
    summary["weight_mean"] = total / count;
    summary["weight_min"] = low;
    summary["weight_max"] = high;
    return summary;
}

inline py::dict layer_diagnostics(py::handle layers_value) {
    py::dict layers = as_dict(layers_value, "checkpoint must be an object");
    py::dict diagnostics;
    for (auto item : layers) {
        const std::string name = py::cast<std::string>(py::str(item.first));
        py::dict layer = as_dict(item.second, "layer " + name + " must be an object with weights");
        py::list matrix = coerce_matrix(value_or(layer, "weights", py::none()), name, true);
        stevejob::math::TensorStats stats;
        stats.compute(py::cast<std::vector<std::vector<double>>>(matrix));
        py::dict row;
        row["mean"] = stats.mean();
        row["variance"] = stats.variance();
        row["max"] = stats.max_val();
        row["min"] = stats.min_val();
        row["total_elements"] = stats.count();
        diagnostics[py::str(name)] = row;
    }
    return diagnostics;
}

inline py::dict update_checkpoint(py::handle layers_value,
                                  py::handle gradients_value,
                                  double learning_rate,
                                  double weight_decay,
                                  double dampening) {
    py::dict layers = as_dict(layers_value, "checkpoint must be an object");
    py::dict gradients = as_dict(gradients_value, "gradients must be an object keyed by layer");
    py::dict updated;
    for (auto item : layers) {
        const std::string name = py::cast<std::string>(py::str(item.first));
        py::dict layer = as_dict(item.second, "layer " + name + " must be an object with weights");
        py::object raw_grads = value_or(gradients, name.c_str(), py::dict());
        if (raw_grads.is_none()) {
            raw_grads = py::dict();
        }
        if (!py::isinstance<py::dict>(raw_grads)) {
            fail("gradients for " + name + " must be an object");
        }
        py::dict layer_grads = py::reinterpret_borrow<py::dict>(raw_grads);
        py::list weights = coerce_matrix(value_or(layer, "weights", py::none()), name + ".weights", true);
        py::list grad_weights = coerce_matrix(
            value_or(layer_grads, "weights", py::none()), name + ".gradient weights", true);
        auto new_weights = stevejob::math::apply_gradient_step(
            py::cast<std::vector<std::vector<double>>>(weights),
            py::cast<std::vector<std::vector<double>>>(grad_weights),
            learning_rate,
            weight_decay,
            dampening);
        py::list biases = coerce_vector(value_or(layer, "biases", py::none()), name + ".biases", true);
        py::list grad_biases = coerce_vector(
            value_or(layer_grads, "biases", py::none()), name + ".gradient biases", true);
        py::object new_biases = py::list();
        if (py::len(biases) > 0) {
            auto stepped = stevejob::math::apply_gradient_step(
                {py::cast<std::vector<double>>(biases)},
                {py::cast<std::vector<double>>(grad_biases)},
                learning_rate,
                weight_decay,
                dampening);
            new_biases = py::cast(stepped.empty() ? std::vector<double>() : stepped[0]);
        }
        py::dict stored;
        stored["weights"] = new_weights;
        stored["biases"] = new_biases;
        updated[py::str(name)] = stored;
    }
    return updated;
}

inline py::dict normalize_ring_candidate(py::handle weights_value) {
    py::dict weights = as_dict(weights_value, "candidate_weights must be an object with layer1 and layer2");
    const std::pair<const char*, std::tuple<int, int, int>> expected[] = {
        {"layer1", {2, 4, 4}},
        {"layer2", {4, 2, 2}},
    };
    py::dict normalized;
    for (const auto& spec : expected) {
        const char* name = spec.first;
        const int rows = std::get<0>(spec.second);
        const int cols = std::get<1>(spec.second);
        const int bias_count = std::get<2>(spec.second);
        if (!weights.contains(name)) {
            fail(std::string(name) + " is missing");
        }
        py::dict layer = as_dict(weights[name], std::string(name) + " is missing");
        py::list matrix = coerce_matrix(field_or(layer, "weights", "w"), std::string(name) + " weights", false);
        py::list bias = coerce_vector(field_or(layer, "biases", "b"), std::string(name) + " biases", false);
        bool rectangular = py::len(matrix) == static_cast<std::size_t>(rows);
        if (rectangular) {
            for (py::handle row : matrix) {
                if (py::len(row) != static_cast<std::size_t>(cols)) {
                    rectangular = false;
                }
            }
        }
        if (!rectangular) {
            fail(std::string(name) + " weights must be " + std::to_string(rows) + "x" + std::to_string(cols));
        }
        if (py::len(bias) != static_cast<std::size_t>(bias_count)) {
            fail(std::string(name) + " biases must have length " + std::to_string(bias_count));
        }
        py::dict stored;
        stored["weights"] = matrix;
        stored["biases"] = bias;
        normalized[name] = stored;
    }
    return normalized;
}

inline py::dict score_ring(py::handle weights_value, int seed, int samples) {
    py::dict weights = normalize_ring_candidate(weights_value);
    py::dict first = weights["layer1"].cast<py::dict>();
    py::dict second = weights["layer2"].cast<py::dict>();
    stevejob::models::RingMLPModel model(
        py::cast<std::vector<std::vector<double>>>(first["weights"]),
        py::cast<std::vector<double>>(first["biases"]),
        py::cast<std::vector<std::vector<double>>>(second["weights"]),
        py::cast<std::vector<double>>(second["biases"]));
    stevejob::models::RingScore score = model.evaluate_ring(seed, samples);
    py::dict measured;
    measured["loss"] = score.loss;
    measured["mean_radius"] = score.mean_radius;
    measured["min_radius"] = score.min_radius;
    measured["max_radius"] = score.max_radius;
    measured["std_radius"] = score.std_radius;
    py::object round = py::module_::import("builtins").attr("round");
    py::dict rounded;
    for (auto item : measured) {
        rounded[item.first] = round(item.second, 4);
    }
    return rounded;
}

inline py::dict stage_ring_candidates(py::handle raw, int max_candidates) {
    if (!py::isinstance<py::list>(raw) || py::len(raw) < 2 || py::len(raw) > static_cast<std::size_t>(max_candidates)) {
        fail("candidates must contain 2 or 3 hypotheses");
    }
    py::list candidates;
    std::size_t index = 0;
    for (py::handle item : raw) {
        if (!py::isinstance<py::dict>(item)) {
            fail("each candidate must be an object");
        }
        py::dict entry = py::reinterpret_borrow<py::dict>(item);
        py::object raw_weights = py::none();
        if (entry.contains("candidate_weights")) {
            raw_weights = entry["candidate_weights"];
        } else if (entry.contains("weights")) {
            raw_weights = entry["weights"];
        }
        py::dict weights = normalize_ring_candidate(raw_weights);
        std::string candidate_id = "candidate_" + std::to_string(index);
        if (entry.contains("candidate_id")) {
            candidate_id = py::cast<std::string>(py::str(entry["candidate_id"]));
        }
        py::dict stored;
        stored["candidate_id"] = candidate_id;
        stored["candidate_weights"] = weights;
        candidates.append(stored);
        ++index;
    }
    py::dict result;
    result["count"] = candidates.size();
    result["candidates"] = candidates;
    return result;
}

inline py::dict trial_row(py::handle entry) {
    if (!py::isinstance<py::dict>(entry)) {
        return py::dict();
    }
    py::dict item = py::reinterpret_borrow<py::dict>(entry);
    py::dict row = item;
    if (item.contains("result") && py::isinstance<py::dict>(item["result"])) {
        row = item["result"].cast<py::dict>();
    }
    if (!row.contains("loss") || !row.contains("weights")) {
        return py::dict();
    }
    return row;
}

inline py::dict select_best_hypothesis(py::handle raw) {
    if (!py::isinstance<py::list>(raw)) {
        fail("trials must be the aggregated score list");
    }
    py::list scored;
    for (py::handle entry : raw) {
        py::dict row = trial_row(entry);
        if (row.contains("loss")) {
            scored.append(row);
        }
    }
    if (py::len(scored) == 0) {
        fail("no scored hypotheses");
    }
    py::dict best = scored[0].cast<py::dict>();
    for (py::handle row_value : scored) {
        py::dict row = row_value.cast<py::dict>();
        if (py::cast<double>(row["loss"]) < py::cast<double>(best["loss"])) {
            best = row;
        }
    }
    py::list trials;
    for (py::handle row_value : scored) {
        py::dict row = row_value.cast<py::dict>();
        py::dict trial;
        if (row.contains("candidate_id")) {
            trial["candidate_id"] = row["candidate_id"];
        } else {
            trial["candidate_id"] = py::none();
        }
        trial["loss"] = row["loss"];
        trial["mean_radius"] = row["mean_radius"];
        trial["min_radius"] = value_or(row, "min_radius", py::none());
        trial["max_radius"] = value_or(row, "max_radius", py::none());
        trial["weights"] = row["weights"];
        trials.append(trial);
    }
    py::dict result;
    result["best_id"] = value_or(best, "candidate_id", py::none());
    result["best_loss"] = best["loss"];
    result["best_mean_radius"] = best["mean_radius"];
    result["best_weights"] = best["weights"];
    result["parameter_count"] = 22;
    result["trials"] = trials;
    return result;
}

inline py::dict receipt_stats(py::handle layers, py::handle outputs_value) {
    const std::vector<double> values = collect_parameters(layers, false, 0);
    py::dict summary;
    summary["parameter_count"] = values.size();
    summary["output_count"] = 0;
    if (!values.empty()) {
        double total = 0.0;
        double low = values[0];
        double high = values[0];
        for (double value : values) {
            total += value;
            if (value < low) {
                low = value;
            }
            if (value > high) {
                high = value;
            }
        }
        summary["weight_mean"] = total / static_cast<double>(values.size());
        summary["weight_min"] = low;
        summary["weight_max"] = high;
    }
    std::size_t output_count = 0;
    double output_total = 0.0;
    if (!outputs_value.is_none() && py::isinstance<py::list>(outputs_value)) {
        for (py::handle row : outputs_value) {
            if (!is_sequence(row)) {
                continue;
            }
            for (py::handle entry : row) {
                output_total += py::cast<double>(entry);
                ++output_count;
            }
        }
    }
    summary["output_count"] = output_count;
    if (output_count > 0) {
        summary["output_mean"] = output_total / static_cast<double>(output_count);
    }
    return summary;
}

inline std::size_t output_count(py::handle outputs_value) {
    return py::cast<std::size_t>(receipt_stats(py::dict(), outputs_value)["output_count"]);
}

inline py::dict eval_summary(py::handle payload_value, std::size_t max_residuals, std::size_t max_anomalies) {
    py::dict payload = as_dict(payload_value, "evaluation must be an object");
    py::object residuals = value_or(payload, "residuals", py::list());
    py::object anomalies = value_or(payload, "anomalies", py::list());
    if (!py::isinstance<py::list>(residuals) || !py::isinstance<py::list>(anomalies)) {
        fail("evaluation residuals and anomalies must be lists");
    }
    if (py::len(residuals) > max_residuals) {
        fail("evaluation has " + std::to_string(py::len(residuals)) + " residuals; the cap is " +
             std::to_string(max_residuals));
    }
    if (py::len(anomalies) > max_anomalies) {
        fail("evaluation has " + std::to_string(py::len(anomalies)) + " anomalies; the cap is " +
             std::to_string(max_anomalies));
    }
    py::object mse = value_or(payload, "mean_squared_error", py::none());
    if (py::isinstance<py::bool_>(mse) ||
        !(py::isinstance<py::int_>(mse) || py::isinstance<py::float_>(mse))) {
        fail("mean_squared_error must be a number");
    }
    py::list residual_values = coerce_vector(residuals, "residuals", false);
    py::list anomaly_values;
    for (py::handle item : anomalies) {
        py::str text = py::str(item);
        anomaly_values.append(text.attr("__getitem__")(py::slice(0, 80, 1)));
    }
    py::dict summary;
    summary["mean_squared_error"] = py::cast<double>(mse);
    summary["residual_count"] = py::len(residual_values);
    summary["residuals"] = residual_values;
    summary["anomalies"] = anomaly_values;
    return summary;
}

inline void validate_tiny_inputs(py::handle inputs, std::size_t max_rows, std::size_t max_cols) {
    if (!py::isinstance<py::list>(inputs) || py::len(inputs) == 0 || py::len(inputs) > max_rows) {
        fail("inputs must be 1 to 8 samples with at most 8 features");
    }
    for (py::handle row : inputs) {
        if (!is_sequence(row) || py::len(row) > max_cols) {
            fail("inputs must be 1 to 8 samples with at most 8 features");
        }
    }
}

inline py::tuple single_layer(py::handle checkpoint_value) {
    py::dict checkpoint = as_dict(checkpoint_value, "checkpoint must be an object");
    if (py::len(checkpoint) != 1) {
        fail("packed evaluation currently runs one layer");
    }
    py::dict layer;
    for (auto item : checkpoint) {
        layer = as_dict(item.second, "layer must be an object");
    }
    return py::make_tuple(
        coerce_matrix(value_or(layer, "weights", py::none()), "weights", true),
        coerce_vector(value_or(layer, "biases", py::none()), "biases", true));
}

inline py::bytes pack_checkpoint(py::handle layers_value, py::handle outputs_value) {
    py::dict layers = as_dict(layers_value, "checkpoint must be an object");
    stevejob::messaging::ModelData model;
    for (auto item : layers) {
        stevejob::messaging::LayerRecord record;
        record.name = py::cast<std::string>(py::str(item.first));
        py::dict layer = as_dict(item.second, "layer " + record.name + " must be an object");
        py::object weights = value_or(layer, "weights", py::list());
        py::object biases = value_or(layer, "biases", py::list());
        if (weights.is_none()) {
            weights = py::list();
        }
        if (biases.is_none()) {
            biases = py::list();
        }
        record.weights = py::cast<std::vector<std::vector<double>>>(weights);
        record.biases = py::cast<std::vector<double>>(biases);
        model.layers.push_back(std::move(record));
    }
    if (!outputs_value.is_none()) {
        model.outputs = py::cast<std::vector<std::vector<double>>>(outputs_value);
    }
    return py::bytes(stevejob::messaging::TensorCompressor::pack(model));
}

inline py::dict omit(py::dict mapping, py::list keys) {
    std::vector<std::string> skipped;
    skipped.reserve(py::len(keys));
    for (py::handle key : keys) {
        skipped.push_back(py::cast<std::string>(py::str(key)));
    }
    py::dict out;
    for (auto item : mapping) {
        const std::string name = py::cast<std::string>(py::str(item.first));
        bool drop = false;
        for (const std::string& key : skipped) {
            if (key == name) {
                drop = true;
            }
        }
        if (!drop) {
            out[item.first] = item.second;
        }
    }
    return out;
}

inline py::dict parse_connection_parts(const std::string& raw) {
    py::dict parts;
    std::size_t start = 0;
    while (start < raw.size()) {
        const std::size_t end = raw.find(';', start) == std::string::npos ? raw.size() : raw.find(';', start);
        const std::size_t eq = raw.find('=', start);
        if (eq != std::string::npos && eq < end && eq > start) {
            parts[py::str(raw.substr(start, eq - start))] = py::str(raw.substr(eq + 1, end - (eq + 1)));
        }
        if (end == raw.size()) {
            break;
        }
        start = end + 1;
    }
    return parts;
}

inline std::string canonical_ms_headers(py::dict headers) {
    std::vector<std::string> names;
    for (auto item : headers) {
        names.push_back(py::cast<std::string>(py::str(item.first)));
    }
    std::sort(names.begin(), names.end());
    std::string canonical;
    for (const std::string& name : names) {
        std::string lower = name;
        for (char& character : lower) {
            if (character >= 'A' && character <= 'Z') {
                character = static_cast<char>(character - 'A' + 'a');
            }
        }
        if (lower.rfind("x-ms-", 0) != 0) {
            continue;
        }
        canonical += lower;
        canonical.push_back(':');
        canonical += py::cast<std::string>(py::str(headers[py::str(name)]));
        canonical.push_back('\n');
    }
    return canonical;
}

inline std::string canonical_resource(const std::string& account,
                                      const std::string& resource_path,
                                      py::dict query) {
    std::string resource = "/" + account + resource_path;
    std::vector<std::string> names;
    for (auto item : query) {
        names.push_back(py::cast<std::string>(py::str(item.first)));
    }
    std::sort(names.begin(), names.end());
    for (const std::string& name : names) {
        std::string lower = name;
        for (char& character : lower) {
            if (character >= 'A' && character <= 'Z') {
                character = static_cast<char>(character - 'A' + 'a');
            }
        }
        resource.push_back('\n');
        resource += lower;
        resource.push_back(':');
        resource += py::cast<std::string>(py::str(query[py::str(name)]));
    }
    return resource;
}

inline std::string query_string(py::dict query) {
    py::object quote = py::module_::import("urllib.parse").attr("quote");
    std::string text;
    bool first = true;
    for (auto item : query) {
        if (!first) {
            text.push_back('&');
        }
        first = false;
        const std::string key = py::cast<std::string>(py::str(item.first));
        const std::string value = py::cast<std::string>(py::str(item.second));
        text += py::cast<std::string>(quote(key));
        text.push_back('=');
        text += py::cast<std::string>(quote(value));
    }
    return text;
}

}  // namespace stevejob::bridge
