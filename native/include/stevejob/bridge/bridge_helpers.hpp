#pragma once

#include "stevejob/math/linear_algebra.hpp"
#include "stevejob/math/tensor_stats.hpp"
#include "stevejob/models/architecture_catalog.hpp"
#include "stevejob/models/neural_network.hpp"
#include "stevejob/models/checkpoint_registry.hpp"
#include "stevejob/storage/json_text.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace stevejob::bridge {

class ValueCodec {
public:
    static void fail(const std::string& message) { throw std::invalid_argument(message); }

    static py::object value_or(const py::dict& mapping, const std::string& key, py::object fallback) {
        if (!mapping.contains(key.c_str())) {
            return fallback;
        }
        return py::reinterpret_borrow<py::object>(mapping[key.c_str()]);
    }

    static py::dict as_dict(py::handle value, const std::string& message) {
        if (!py::isinstance<py::dict>(value)) {
            fail(message);
        }
        return py::reinterpret_borrow<py::dict>(value);
    }

    static bool is_sequence(py::handle value) {
        return py::isinstance<py::sequence>(value) && !py::isinstance<py::str>(value) && !py::isinstance<py::bytes>(value);
    }

    static py::list coerce_vector(py::handle value, const std::string& label, bool allow_none) {
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

    static py::list coerce_matrix(py::handle value, const std::string& label, bool allow_none) {
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

    static py::dict omit(py::dict mapping, py::list keys) {
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
};

class FieldWriter {
public:
    static py::dict from_fields(const std::vector<stevejob::models::FieldValue>& fields) {
        py::dict out;
        for (const stevejob::models::FieldValue& field : fields) {
            if (!field.present) {
                continue;
            }
            const char* key = field.key.c_str();
            if (field.kind == stevejob::models::ValueKind::Count) {
                out[key] = field.count;
            } else if (field.kind == stevejob::models::ValueKind::NumberList) {
                out[key] = field.numbers;
            } else if (field.kind == stevejob::models::ValueKind::TextList) {
                out[key] = field.texts;
            } else {
                out[key] = field.number;
            }
        }
        return out;
    }
};

class SchemaLookup {
public:
    static std::string architecture_from_object(py::handle value) {
        const char* tag = stevejob::models::CheckpointRegistry::model_tag();
        if (py::isinstance<py::dict>(value)) {
            py::dict object = py::reinterpret_borrow<py::dict>(value);
            if (object.contains(tag) && py::isinstance<py::str>(object[tag])) {
                return py::cast<std::string>(object[tag]);
            }
        }
        return "";
    }

    static stevejob::models::SchemaMap from_object(py::handle value) {
        stevejob::models::CheckpointRegistry& registry = stevejob::models::CheckpointRegistry::instance();
        const std::string arch = architecture_from_object(value);
        if (!arch.empty()) {
            if (registry.contains(arch)) {
                return registry.find(arch);
            }
            if (arch == "mlp_regressor" || arch == "bottleneck_ae" || arch == "demo_scale") {
                return stevejob::models::ArchitectureCatalog::make(
                    arch, [](const auto& model) { return model.checkpoint_schema(); });
            }
        }
        return registry.fallback();
    }
};

class CheckpointBridge {
public:
    static std::vector<double> collect_parameters(py::handle layers_value, bool enforce, std::size_t max_parameters) {
        const stevejob::models::SchemaMap schema = SchemaLookup::from_object(layers_value);
        py::dict layers = ValueCodec::as_dict(layers_value, "checkpoint must be an object");
        std::vector<double> values;
        for (auto item : layers) {
            const std::string name = py::cast<std::string>(py::str(item.first));
            if (name == stevejob::models::CheckpointRegistry::model_tag()) {
                continue;
            }
            if (!py::isinstance<py::dict>(item.second)) {
                ValueCodec::fail("layer " + name + " must be an object");
            }
            py::dict layer = py::reinterpret_borrow<py::dict>(item.second);
            for (const stevejob::models::ParameterSlot& slot : schema.slots()) {
                append_slot(values, layer, name, slot);
            }
        }
        if (enforce && values.empty()) {
            ValueCodec::fail("checkpoint has no parameters");
        }
        if (enforce && values.size() > max_parameters) {
            ValueCodec::fail("checkpoint has " + std::to_string(values.size()) + " parameters; this run allows at most " +
                             std::to_string(max_parameters));
        }
        return values;
    }

    static py::dict checkpoint_summary(py::handle layers, std::size_t max_parameters) {
        const stevejob::models::SchemaMap schema = SchemaLookup::from_object(layers);
        return FieldWriter::from_fields(schema.compile_summary(collect_parameters(layers, true, max_parameters)));
    }

    static py::dict layer_diagnostics(py::handle layers_value) {
        const stevejob::models::SchemaMap schema = SchemaLookup::from_object(layers_value);
        const std::string arch = SchemaLookup::architecture_from_object(layers_value);
        py::dict layers = ValueCodec::as_dict(layers_value, "checkpoint must be an object");
        py::dict diagnostics;
        for (auto item : layers) {
            const std::string name = py::cast<std::string>(py::str(item.first));
            if (name == stevejob::models::CheckpointRegistry::model_tag()) {
                continue;
            }
            py::dict layer = ValueCodec::as_dict(item.second, "layer " + name + " must be an object");
            const std::vector<std::vector<double>> matrix = matrix_slot(schema, layer, name);
            const stevejob::math::TensorStats stats = stevejob::models::ArchitectureCatalog::make(
                arch.empty() ? "mlp_regressor" : arch,
                [&](const auto& model) { return model.compute_layer_stats(matrix); });
            diagnostics[py::str(name)] = FieldWriter::from_fields(
                schema.compile_diagnostics(stats.mean(), stats.variance(), stats.min_val(), stats.max_val(), stats.count()));
        }
        return diagnostics;
    }

    static py::dict update_checkpoint(py::handle layers_value,
                                      py::handle gradients_value,
                                      double learning_rate,
                                      double weight_decay,
                                      double dampening) {
        const stevejob::models::SchemaMap schema = SchemaLookup::from_object(layers_value);
        py::dict layers = ValueCodec::as_dict(layers_value, "checkpoint must be an object");
        py::dict gradients = ValueCodec::as_dict(gradients_value, "gradients must be an object keyed by layer");
        py::dict updated;
        for (auto item : layers) {
            const std::string name = py::cast<std::string>(py::str(item.first));
            if (name == stevejob::models::CheckpointRegistry::model_tag()) {
                updated[item.first] = item.second;
                continue;
            }
            py::dict layer = ValueCodec::as_dict(item.second, "layer " + name + " must be an object");
            py::dict layer_grads = gradient_layer(gradients, name);
            py::dict stored;
            for (const stevejob::models::ParameterSlot& slot : schema.slots()) {
                stored[slot.key.c_str()] = step_slot(layer, layer_grads, name, slot, learning_rate, weight_decay, dampening);
            }
            updated[py::str(name)] = stored;
        }
        return updated;
    }

    static py::dict receipt_stats(py::handle layers, py::handle outputs_value) {
        const stevejob::models::SchemaMap schema = SchemaLookup::from_object(layers);
        const stevejob::models::OutputSample outputs = rollup_outputs(outputs_value);
        return FieldWriter::from_fields(schema.compile_receipt(collect_parameters(layers, false, 0), outputs));
    }

    static std::size_t output_count(py::handle outputs_value) { return rollup_outputs(outputs_value).count; }

    static void validate_tiny_inputs(py::handle inputs, std::size_t max_rows, std::size_t max_cols) {
        const std::string message = "inputs must be 1 to " + std::to_string(max_rows) + " samples with at most " +
                                    std::to_string(max_cols) + " features";
        if (!py::isinstance<py::list>(inputs) || py::len(inputs) == 0 || py::len(inputs) > max_rows) {
            ValueCodec::fail(message);
        }
        for (py::handle row : inputs) {
            if (!ValueCodec::is_sequence(row) || py::len(row) > max_cols) {
                ValueCodec::fail(message);
            }
        }
    }

    static py::tuple single_layer(py::handle checkpoint_value) {
        const stevejob::models::SchemaMap schema = SchemaLookup::from_object(checkpoint_value);
        py::dict checkpoint = ValueCodec::as_dict(checkpoint_value, "checkpoint must be an object");
        std::size_t layers = 0;
        py::dict layer;
        for (auto item : checkpoint) {
            const std::string name = py::cast<std::string>(py::str(item.first));
            if (name == stevejob::models::CheckpointRegistry::model_tag()) {
                continue;
            }
            layer = ValueCodec::as_dict(item.second, "layer must be an object");
            ++layers;
        }
        if (layers != 1) {
            ValueCodec::fail("packed evaluation currently runs one layer");
        }
        py::tuple result(schema.slots().size());
        std::size_t index = 0;
        for (const stevejob::models::ParameterSlot& slot : schema.slots()) {
            result[index++] = read_slot(layer, slot.key, slot.shape);
        }
        return result;
    }

    static py::bytes pack_checkpoint(py::handle layers_value, py::handle outputs_value) {
        const stevejob::models::SchemaMap schema = SchemaLookup::from_object(layers_value);
        const std::string arch = SchemaLookup::architecture_from_object(layers_value);
        py::dict layers = ValueCodec::as_dict(layers_value, "checkpoint must be an object");
        stevejob::models::NeuralNetwork model("checkpoint");
        if (!arch.empty()) {
            model.architecture = arch;
        }
        for (auto item : layers) {
            const std::string name = py::cast<std::string>(py::str(item.first));
            if (name == stevejob::models::CheckpointRegistry::model_tag()) {
                continue;
            }
            py::dict layer = ValueCodec::as_dict(item.second, "layer " + name + " must be an object");
            std::vector<std::vector<double>> weights;
            std::vector<double> biases;
            for (const stevejob::models::ParameterSlot& slot : schema.slots()) {
                py::object value = ValueCodec::value_or(layer, slot.key, py::list());
                if (value.is_none()) {
                    value = py::list();
                }
                if (slot.shape == stevejob::models::SlotShape::Matrix) {
                    weights = py::cast<std::vector<std::vector<double>>>(value);
                } else {
                    biases = py::cast<std::vector<double>>(value);
                }
            }
            const std::size_t rows = weights.size();
            const std::size_t cols = biases.size();
            model.add_layer(name, rows, cols);
            if (!weights.empty()) {
                model.layers.back().weights = std::move(weights);
            }
            if (!biases.empty()) {
                model.layers.back().biases = std::move(biases);
            }
        }
        if (!outputs_value.is_none()) {
            model.outputs = py::cast<std::vector<std::vector<double>>>(outputs_value);
        }
        return py::bytes(stevejob::models::ArchitectureCatalog::make(
            arch.empty() ? "mlp_regressor" : arch,
            [&](const auto& arch_model) { return arch_model.container_scheme().pack(model); }));
    }

    static double parse_dampening(py::handle args_or_value, const std::string& architecture = "") {
        return stevejob::models::ArchitectureCatalog::make(
            architecture.empty() ? "mlp_regressor" : architecture,
            [&](const auto& model) -> double {
                py::object raw = py::reinterpret_borrow<py::object>(args_or_value);
                if (py::isinstance<py::dict>(raw)) {
                    py::dict mapping = py::reinterpret_borrow<py::dict>(raw);
                    if (mapping.contains("dampening") && !mapping["dampening"].is_none()) {
                        raw = py::reinterpret_borrow<py::object>(mapping["dampening"]);
                    } else if (mapping.contains("dampening_text") && !mapping["dampening_text"].is_none()) {
                        raw = py::reinterpret_borrow<py::object>(mapping["dampening_text"]);
                    } else {
                        return 1.0;
                    }
                }
                if (raw.is_none()) {
                    return 1.0;
                }
                if (!py::isinstance<py::bool_>(raw) && (py::isinstance<py::int_>(raw) || py::isinstance<py::float_>(raw))) {
                    return model.validate_dampening(py::cast<double>(raw));
                }
                if (py::isinstance<py::str>(raw)) {
                    return model.parse_dampening_text(py::cast<std::string>(raw));
                }
                ValueCodec::fail("dampening must be a number");
                return 1.0;
            });
    }

    static double parse_learning_rate(py::handle args_or_value, const std::string& architecture = "") {
        return stevejob::models::ArchitectureCatalog::make(
            architecture.empty() ? "mlp_regressor" : architecture,
            [&](const auto& model) -> double {
                py::object raw = py::reinterpret_borrow<py::object>(args_or_value);
                if (py::isinstance<py::dict>(raw)) {
                    py::dict mapping = py::reinterpret_borrow<py::dict>(raw);
                    raw = ValueCodec::value_or(mapping, "learning_rate", py::none());
                }
                if (raw.is_none()) {
                    return model.validate_learning_rate(model.config().default_learning_rate);
                }
                return model.validate_learning_rate(py::cast<double>(raw));
            });
    }

    static double parse_weight_decay(py::handle args_or_value, const std::string& architecture = "") {
        return stevejob::models::ArchitectureCatalog::make(
            architecture.empty() ? "mlp_regressor" : architecture,
            [&](const auto& model) -> double {
                py::object raw = py::reinterpret_borrow<py::object>(args_or_value);
                if (py::isinstance<py::dict>(raw)) {
                    py::dict mapping = py::reinterpret_borrow<py::dict>(raw);
                    raw = ValueCodec::value_or(mapping, "weight_decay", py::none());
                }
                if (raw.is_none()) {
                    return model.validate_weight_decay(model.config().default_weight_decay);
                }
                return model.validate_weight_decay(py::cast<double>(raw));
            });
    }

    static py::dict parse_gradients(py::handle args_or_value,
                                    py::handle checkpoint_value = py::none(),
                                    const std::string& architecture = "") {
        const std::string arch = !architecture.empty() ? architecture : SchemaLookup::architecture_from_object(checkpoint_value);
        return stevejob::models::ArchitectureCatalog::make(
            arch.empty() ? "mlp_regressor" : arch,
            [&](const auto& model) -> py::dict {
                py::object raw = py::reinterpret_borrow<py::object>(args_or_value);
                if (py::isinstance<py::dict>(raw)) {
                    py::dict mapping = py::reinterpret_borrow<py::dict>(raw);
                    if (mapping.contains("gradients") && !mapping["gradients"].is_none()) {
                        raw = py::reinterpret_borrow<py::object>(mapping["gradients"]);
                    } else if (mapping.contains("gradients_text") && py::isinstance<py::str>(mapping["gradients_text"])) {
                        raw = py::reinterpret_borrow<py::object>(mapping["gradients_text"]);
                    }
                }
                py::dict parsed;
                if (py::isinstance<py::str>(raw)) {
                    const std::string text = py::cast<std::string>(raw);
                    if (!text.empty()) {
                        const stevejob::storage::JsonText json = model.parse_gradients_text(text);
                        parsed = json_to_py_dict(json);
                    }
                } else if (py::isinstance<py::dict>(raw)) {
                    parsed = py::reinterpret_borrow<py::dict>(raw);
                } else {
                    ValueCodec::fail("gradients must be an object keyed by layer");
                }
                if (py::len(parsed) == 0) {
                    ValueCodec::fail("gradients must not be empty");
                }
                py::dict normalized;
                py::dict ckpt = py::isinstance<py::dict>(checkpoint_value)
                                    ? py::reinterpret_borrow<py::dict>(checkpoint_value)
                                    : py::dict();
                for (auto item : parsed) {
                    const std::string layer_name = py::cast<std::string>(py::str(item.first));
                    if (!py::isinstance<py::dict>(item.second)) {
                        ValueCodec::fail("gradients for " + layer_name + " must be an object");
                    }
                    py::dict layer_grad = py::reinterpret_borrow<py::dict>(item.second);
                    py::dict out_layer;
                    std::size_t rows = 1;
                    std::size_t cols = 1;
                    if (ckpt.contains(layer_name.c_str()) && py::isinstance<py::dict>(ckpt[layer_name.c_str()])) {
                        py::dict ckpt_layer = py::reinterpret_borrow<py::dict>(ckpt[layer_name.c_str()]);
                        if (ckpt_layer.contains("weights") && py::isinstance<py::list>(ckpt_layer["weights"])) {
                            py::list w = py::reinterpret_borrow<py::list>(ckpt_layer["weights"]);
                            rows = py::len(w);
                            if (rows > 0 && ValueCodec::is_sequence(w[0])) {
                                cols = py::len(w[0]);
                            }
                        }
                        if (ckpt_layer.contains("biases") && py::isinstance<py::list>(ckpt_layer["biases"])) {
                            cols = py::len(ckpt_layer["biases"]);
                        }
                    }
                    const stevejob::models::SliceRef slice{layer_name, layer_name, 0, 0, 0, rows, cols};
                    for (auto g_item : layer_grad) {
                        const std::string slot_key = py::cast<std::string>(py::str(g_item.first));
                        if (slot_key == "weights" && py::isinstance<py::str>(g_item.second)) {
                            out_layer["weights"] = py::cast(
                                model.get_scheme_for_slice(slice).deserialize_matrix(
                                    py::cast<std::string>(g_item.second), rows, cols));
                        } else if (slot_key == "biases" && py::isinstance<py::str>(g_item.second)) {
                            out_layer["biases"] = py::cast(
                                model.get_bias_scheme(slice).deserialize_vector(
                                    py::cast<std::string>(g_item.second), cols));
                        } else {
                            out_layer[g_item.first] = g_item.second;
                        }
                    }
                    normalized[py::str(layer_name)] = out_layer;
                }
                return normalized;
            });
    }

    static py::list evaluate_checkpoint_inputs(py::handle checkpoint_value, py::handle inputs_value) {
        const std::string arch = SchemaLookup::architecture_from_object(checkpoint_value);
        py::list coerced = ValueCodec::coerce_matrix(inputs_value, "inputs", false);
        const auto inputs = py::cast<std::vector<std::vector<double>>>(coerced);
        py::tuple layer = single_layer(checkpoint_value);
        const auto weights = py::cast<std::vector<std::vector<double>>>(layer[0]);
        const auto biases = py::cast<std::vector<double>>(layer[1]);
        const auto outputs = stevejob::models::ArchitectureCatalog::make(
            arch.empty() ? "mlp_regressor" : arch,
            [&](const auto& model) { return model.evaluate_layer_inputs(inputs, weights, biases); });
        return py::cast(outputs);
    }

private:
    static void append_slot(std::vector<double>& values,
                            py::dict layer,
                            const std::string& name,
                            const stevejob::models::ParameterSlot& slot) {
        py::object raw = ValueCodec::value_or(layer, slot.key, py::list());
        if (slot.shape == stevejob::models::SlotShape::Matrix) {
            if (!py::isinstance<py::list>(raw)) {
                ValueCodec::fail("layer " + name + " " + slot.key + " must be a matrix");
            }
            for (py::handle row : raw) {
                if (!ValueCodec::is_sequence(row)) {
                    ValueCodec::fail("layer " + name + " " + slot.key + " must be a matrix");
                }
                for (py::handle entry : row) {
                    values.push_back(py::cast<double>(entry));
                }
            }
            return;
        }
        if (!py::isinstance<py::list>(raw)) {
            ValueCodec::fail("layer " + name + " " + slot.key + " must be a list");
        }
        for (py::handle entry : raw) {
            values.push_back(py::cast<double>(entry));
        }
    }

    static std::vector<std::vector<double>> matrix_slot(const stevejob::models::SchemaMap& schema,
                                                        py::dict layer,
                                                        const std::string& name) {
        for (const stevejob::models::ParameterSlot& slot : schema.slots()) {
            if (slot.shape != stevejob::models::SlotShape::Matrix) {
                continue;
            }
            py::list matrix = ValueCodec::coerce_matrix(ValueCodec::value_or(layer, slot.key, py::none()), name, true);
            return py::cast<std::vector<std::vector<double>>>(matrix);
        }
        return {};
    }

    static py::object read_slot(py::dict layer, const std::string& key, stevejob::models::SlotShape shape) {
        if (shape == stevejob::models::SlotShape::Matrix) {
            return ValueCodec::coerce_matrix(ValueCodec::value_or(layer, key, py::none()), key, true);
        }
        return ValueCodec::coerce_vector(ValueCodec::value_or(layer, key, py::none()), key, true);
    }

    static py::object step_slot(py::dict layer,
                                py::dict gradients,
                                const std::string& name,
                                const stevejob::models::ParameterSlot& slot,
                                double learning_rate,
                                double weight_decay,
                                double dampening) {
        if (slot.shape == stevejob::models::SlotShape::Matrix) {
            py::list weights = ValueCodec::coerce_matrix(ValueCodec::value_or(layer, slot.key, py::none()), name + "." + slot.key, true);
            py::list grad_weights = ValueCodec::coerce_matrix(
                ValueCodec::value_or(gradients, slot.key, py::none()), name + ".gradient " + slot.key, true);
            return py::cast(stevejob::math::apply_gradient_step(py::cast<std::vector<std::vector<double>>>(weights),
                                                                py::cast<std::vector<std::vector<double>>>(grad_weights),
                                                                learning_rate,
                                                                weight_decay,
                                                                dampening));
        }
        py::list biases = ValueCodec::coerce_vector(ValueCodec::value_or(layer, slot.key, py::none()), name + "." + slot.key, true);
        py::list grad_biases = ValueCodec::coerce_vector(
            ValueCodec::value_or(gradients, slot.key, py::none()), name + ".gradient " + slot.key, true);
        if (py::len(biases) == 0) {
            return py::list();
        }
        auto stepped = stevejob::math::apply_gradient_step({py::cast<std::vector<double>>(biases)},
                                                           {py::cast<std::vector<double>>(grad_biases)},
                                                           learning_rate,
                                                           weight_decay,
                                                           dampening);
        return py::cast(stepped.empty() ? std::vector<double>() : stepped[0]);
    }

    static stevejob::models::OutputSample rollup_outputs(py::handle outputs_value) {
        stevejob::models::OutputSample rollup;
        if (outputs_value.is_none() || !py::isinstance<py::list>(outputs_value)) {
            return rollup;
        }
        for (py::handle row : outputs_value) {
            if (!ValueCodec::is_sequence(row)) {
                continue;
            }
            for (py::handle entry : row) {
                rollup.total += py::cast<double>(entry);
                ++rollup.count;
            }
        }
        return rollup;
    }

    static py::object json_to_py_object(const stevejob::storage::JsonText& node) {
        switch (node.kind) {
            case stevejob::storage::JsonText::Kind::Null:
                return py::none();
            case stevejob::storage::JsonText::Kind::Bool:
                return py::bool_(node.boolean);
            case stevejob::storage::JsonText::Kind::Number:
                if (node.number_is_int) {
                    return py::int_(node.integer);
                }
                return py::float_(node.number);
            case stevejob::storage::JsonText::Kind::String:
                return py::str(node.text);
            case stevejob::storage::JsonText::Kind::Array: {
                py::list out;
                for (const auto& item : node.array) {
                    out.append(json_to_py_object(item));
                }
                return out;
            }
            case stevejob::storage::JsonText::Kind::Object:
            default:
                return json_to_py_dict(node);
        }
    }

    static py::dict json_to_py_dict(const stevejob::storage::JsonText& node) {
        if (node.kind != stevejob::storage::JsonText::Kind::Object) {
            ValueCodec::fail("expected JSON object");
        }
        py::dict out;
        for (const auto& pair : node.object) {
            out[py::str(pair.first)] = json_to_py_object(pair.second);
        }
        return out;
    }

    static py::dict gradient_layer(py::dict gradients, const std::string& name) {
        py::object raw_grads = ValueCodec::value_or(gradients, name, py::dict());
        if (raw_grads.is_none()) {
            raw_grads = py::dict();
        }
        if (!py::isinstance<py::dict>(raw_grads)) {
            ValueCodec::fail("gradients for " + name + " must be an object");
        }
        return py::reinterpret_borrow<py::dict>(raw_grads);
    }
};

class EvaluationBridge {
public:
    static py::dict summarize(py::handle payload_value, std::size_t max_residuals, std::size_t max_anomalies) {
        const stevejob::models::SchemaMap& schema = SchemaLookup::from_object(payload_value);
        py::dict payload = ValueCodec::as_dict(payload_value, "evaluation must be an object");
        const std::map<std::string, std::size_t> caps{{"residuals", max_residuals}, {"anomalies", max_anomalies}};
        stevejob::models::EvalSample sample;
        for (const stevejob::models::EvalInput& input : schema.eval_inputs()) {
            read_input(payload, input, caps, sample);
        }
        return FieldWriter::from_fields(schema.compile_eval(sample));
    }

private:
    static void read_input(py::dict payload,
                           const stevejob::models::EvalInput& input,
                           const std::map<std::string, std::size_t>& caps,
                           stevejob::models::EvalSample& sample) {
        py::object value = ValueCodec::value_or(payload, input.key, py::none());
        if (input.kind == stevejob::models::ValueKind::Number) {
            if (py::isinstance<py::bool_>(value) || !(py::isinstance<py::int_>(value) || py::isinstance<py::float_>(value))) {
                ValueCodec::fail(input.key + " must be a number");
            }
            sample.numbers[input.key] = py::cast<double>(value);
            return;
        }
        if (!py::isinstance<py::list>(value)) {
            ValueCodec::fail(input.key + " must be a list");
        }
        const auto cap = caps.find(input.cap);
        if (cap != caps.end() && py::len(value) > cap->second) {
            ValueCodec::fail("evaluation has " + std::to_string(py::len(value)) + " " + input.key + "; the cap is " +
                             std::to_string(cap->second));
        }
        if (input.kind == stevejob::models::ValueKind::NumberList) {
            sample.number_lists[input.key] = py::cast<std::vector<double>>(ValueCodec::coerce_vector(value, input.key, false));
            return;
        }
        std::vector<std::string> texts;
        for (py::handle item : value) {
            const std::string text = py::cast<std::string>(py::str(item));
            texts.push_back(input.clip > 0 && static_cast<int>(text.size()) > input.clip ? text.substr(0, static_cast<std::size_t>(input.clip))
                                                                                          : text);
        }
        sample.text_lists[input.key] = std::move(texts);
    }
};

class SharedKeyText {
public:
    static py::dict parse_connection_parts(const std::string& raw) {
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

    static std::string canonical_ms_headers(py::dict headers) {
        std::vector<std::string> names;
        for (auto item : headers) {
            names.push_back(py::cast<std::string>(py::str(item.first)));
        }
        std::sort(names.begin(), names.end());
        std::string canonical;
        for (const std::string& name : names) {
            const std::string lower = lowered(name);
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

    static std::string canonical_resource(const std::string& account, const std::string& resource_path, py::dict query) {
        std::string resource = "/" + account + resource_path;
        std::vector<std::string> names;
        for (auto item : query) {
            names.push_back(py::cast<std::string>(py::str(item.first)));
        }
        std::sort(names.begin(), names.end());
        for (const std::string& name : names) {
            resource.push_back('\n');
            resource += lowered(name);
            resource.push_back(':');
            resource += py::cast<std::string>(py::str(query[py::str(name)]));
        }
        return resource;
    }

    static std::string query_string(py::dict query) {
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

private:
    static std::string lowered(std::string text) {
        for (char& character : text) {
            if (character >= 'A' && character <= 'Z') {
                character = static_cast<char>(character - 'A' + 'a');
            }
        }
        return text;
    }
};

}  // namespace stevejob::bridge
