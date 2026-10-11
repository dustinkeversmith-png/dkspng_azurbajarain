#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stevejob/bridge/host_compute.hpp"
#include "stevejob/bridge/engine_gateway.hpp"
#include "stevejob/bridge/modular_orchestrator.hpp"
#include "stevejob/bridge/native_pipeline.hpp"
#include "stevejob/models/neural_model.hpp"
#include "stevejob/math/linear_algebra.hpp"
#include "stevejob/math/tensor_stats.hpp"
#include "stevejob/messaging/compressor.hpp"
#include "stevejob/models/regression_mlp_model.hpp"
#include "stevejob/models/ring_mlp_model.hpp"

namespace py = pybind11;

namespace {

stevejob::messaging::ModelData model_from_parts(
    const std::vector<std::string>& names,
    const std::vector<std::vector<std::vector<double>>>& weights,
    const std::vector<std::vector<double>>& biases,
    const std::vector<std::vector<double>>& outputs) {
    if (names.size() != weights.size() || names.size() != biases.size()) {
        throw std::invalid_argument("layer names, weights, and biases must match");
    }
    stevejob::messaging::ModelData model;
    model.layers.reserve(names.size());
    for (std::size_t index = 0; index < names.size(); ++index) {
        stevejob::messaging::LayerRecord layer;
        layer.name = names[index];
        layer.weights = weights[index];
        layer.biases = biases[index];
        model.layers.push_back(std::move(layer));
    }
    model.outputs = outputs;
    return model;
}

py::dict document_from_decode(const stevejob::messaging::DecodeResult& decoded) {
    py::dict layers;
    for (const stevejob::messaging::LayerRecord& layer : decoded.model.layers) {
        py::dict entry;
        entry["weights"] = layer.weights;
        entry["biases"] = layer.biases;
        layers[py::str(layer.name)] = entry;
    }
    py::dict document;
    document["layers"] = layers;
    document["outputs"] = decoded.model.outputs;
    document["compressed"] = decoded.compressed;
    document["container_checksum"] = stevejob::core::to_hex(decoded.checksum);
    return document;
}

py::dict score_dict(const stevejob::models::RingScore& score) {
    py::dict out;
    out["loss"] = score.loss;
    out["mean_radius"] = score.mean_radius;
    out["min_radius"] = score.min_radius;
    out["max_radius"] = score.max_radius;
    out["std_radius"] = score.std_radius;
    return out;
}

}  // namespace

PYBIND11_MODULE(tensor_engine, module) {
    module.doc() = "Native C++ tensor steps for a tiny perceptron workflow";

    py::class_<stevejob::math::TensorStats>(module, "TensorStats")
        .def(py::init<>())
        .def("compute", &stevejob::math::TensorStats::compute)
        .def_property_readonly("mean", &stevejob::math::TensorStats::mean)
        .def_property_readonly("variance", &stevejob::math::TensorStats::variance)
        .def_property_readonly("max_val", &stevejob::math::TensorStats::max_val)
        .def_property_readonly("min_val", &stevejob::math::TensorStats::min_val)
        .def_property_readonly("count", &stevejob::math::TensorStats::count);

    py::class_<stevejob::models::RingScore>(module, "RingScore")
        .def_readonly("loss", &stevejob::models::RingScore::loss)
        .def_readonly("mean_radius", &stevejob::models::RingScore::mean_radius)
        .def_readonly("min_radius", &stevejob::models::RingScore::min_radius)
        .def_readonly("max_radius", &stevejob::models::RingScore::max_radius)
        .def_readonly("std_radius", &stevejob::models::RingScore::std_radius);

    py::class_<stevejob::models::RingMLPModel>(module, "RingMLPModel")
        .def(py::init<std::vector<std::vector<double>>,
                      std::vector<double>,
                      std::vector<std::vector<double>>,
                      std::vector<double>>(),
             py::arg("first_weights"),
             py::arg("first_bias"),
             py::arg("second_weights"),
             py::arg("second_bias"))
        .def("forward", &stevejob::models::RingMLPModel::forward)
        .def("evaluate_ring",
             [](const stevejob::models::RingMLPModel& model, int seed, int samples) {
                 stevejob::models::RingScore score;
                 {
                     py::gil_scoped_release release;
                     score = model.evaluate_ring(seed, samples);
                 }
                 return score_dict(score);
             },
             py::arg("seed") = 42,
             py::arg("samples") = 100)
        .def(
            "apply_gradients",
            [](stevejob::models::RingMLPModel& model,
               const std::vector<std::vector<double>>& layer1_weights,
               const std::vector<double>& layer1_biases,
               const std::vector<std::vector<double>>& layer2_weights,
               const std::vector<double>& layer2_biases,
               double learning_rate,
               double decay) {
                stevejob::models::Gradients grads{
                    layer1_weights, layer1_biases, layer2_weights, layer2_biases};
                py::gil_scoped_release release;
                model.apply_gradients(grads, learning_rate, decay);
            },
            py::arg("layer1_weights"),
            py::arg("layer1_biases"),
            py::arg("layer2_weights"),
            py::arg("layer2_biases"),
            py::arg("learning_rate"),
            py::arg("decay"));

    py::class_<stevejob::messaging::TensorCompressor>(module, "TensorCompressor")
        .def_static(
            "pack",
            [](const std::vector<std::string>& names,
               const std::vector<std::vector<std::vector<double>>>& weights,
               const std::vector<std::vector<double>>& biases,
               const std::vector<std::vector<double>>& outputs) {
                std::string packed;
                {
                    py::gil_scoped_release release;
                    packed = stevejob::messaging::TensorCompressor::pack(
                        model_from_parts(names, weights, biases, outputs));
                }
                return py::bytes(packed);
            },
            py::arg("names"),
            py::arg("weights"),
            py::arg("biases"),
            py::arg("outputs"))
        .def_static(
            "unpack",
            [](const py::bytes& packed) {
                const std::string bytes = packed;
                return document_from_decode(stevejob::messaging::TensorCompressor::decode(bytes));
            },
            py::arg("blob"));

    module.def(
        "compute_layer_stats",
        [](const std::vector<std::vector<double>>& matrix) {
            stevejob::math::TensorStats stats;
            {
                py::gil_scoped_release release;
                stats.compute(matrix);
            }
            return stats;
        },
        py::arg("matrix"),
        "Mean, population variance, and min/max for one layer matrix");
    module.def(
        "apply_gradient_step",
        [](const std::vector<std::vector<double>>& weights,
           const std::vector<std::vector<double>>& gradients,
           double learning_rate,
           double weight_decay,
           double dampening) {
            py::gil_scoped_release release;
            return stevejob::math::apply_gradient_step(
                weights, gradients, learning_rate, weight_decay, dampening);
        },
        py::arg("weights"),
        py::arg("gradients"),
        py::arg("learning_rate"),
        py::arg("weight_decay") = 0.0001,
        py::arg("dampening") = 1.0,
        "Applies w = w - lr * dampening * (grad + decay * w)");
    module.def(
        "linear_outputs",
        [](const std::vector<std::vector<double>>& inputs,
           const std::vector<std::vector<double>>& weights,
           const std::vector<double>& biases) {
            py::gil_scoped_release release;
            return stevejob::math::linear_outputs(inputs, weights, biases);
        },
        py::arg("inputs"),
        py::arg("weights"),
        py::arg("biases"),
        "y = bias + weights · inputs for one layer");
    module.def(
        "mlp_forward",
        [](const std::vector<std::vector<double>>& noise,
           const std::vector<std::vector<double>>& first_weights,
           const std::vector<double>& first_bias,
           const std::vector<std::vector<double>>& second_weights,
           const std::vector<double>& second_bias) {
            stevejob::models::RingMLPModel model(
                first_weights, first_bias, second_weights, second_bias);
            py::gil_scoped_release release;
            return model.forward(noise);
        },
        py::arg("noise"),
        py::arg("first_weights"),
        py::arg("first_bias"),
        py::arg("second_weights"),
        py::arg("second_bias"),
        "2-4-2 tanh MLP. Weight matrices are shaped (in, out).");
    module.def(
        "evaluate_ring",
        [](const std::vector<std::vector<double>>& first_weights,
           const std::vector<double>& first_bias,
           const std::vector<std::vector<double>>& second_weights,
           const std::vector<double>& second_bias,
           int seed,
           int samples) {
            stevejob::models::RingMLPModel model(
                first_weights, first_bias, second_weights, second_bias);
            stevejob::models::RingScore score;
            {
                py::gil_scoped_release release;
                score = model.evaluate_ring(seed, samples);
            }
            return score_dict(score);
        },
        py::arg("first_weights"),
        py::arg("first_bias"),
        py::arg("second_weights"),
        py::arg("second_bias"),
        py::arg("seed") = 42,
        py::arg("samples") = 100,
        "Score a 2-4-2 generator against a radius-1 ring. No gradients.");
    module.def("coerce_matrix", &stevejob::bridge::coerce_matrix, py::arg("value"), py::arg("label"), py::arg("allow_none") = false);
    module.def("coerce_vector", &stevejob::bridge::coerce_vector, py::arg("value"), py::arg("label"), py::arg("allow_none") = false);
    module.def("parameter_count", [](py::handle layers, std::size_t max_parameters) {
        return stevejob::bridge::collect_parameters(layers, true, max_parameters).size();
    }, py::arg("layers"), py::arg("max_parameters"));
    module.def("checkpoint_summary", &stevejob::bridge::checkpoint_summary, py::arg("layers"), py::arg("max_parameters"));
    module.def("layer_diagnostics", &stevejob::bridge::layer_diagnostics, py::arg("layers"));
    module.def("update_checkpoint", &stevejob::bridge::update_checkpoint,
               py::arg("layers"), py::arg("gradients"), py::arg("learning_rate"), py::arg("weight_decay"), py::arg("dampening"));
    module.def("normalize_ring_candidate", &stevejob::bridge::normalize_ring_candidate, py::arg("weights"));
    module.def("score_ring", &stevejob::bridge::score_ring, py::arg("weights"), py::arg("seed") = 42, py::arg("samples") = 100);
    module.def("stage_ring_candidates", &stevejob::bridge::stage_ring_candidates, py::arg("candidates"), py::arg("max_candidates"));
    module.def("select_best_hypothesis", &stevejob::bridge::select_best_hypothesis, py::arg("trials"));
    module.def("receipt_stats", &stevejob::bridge::receipt_stats, py::arg("layers"), py::arg("outputs"));
    module.def("output_count", &stevejob::bridge::output_count, py::arg("outputs"));
    module.def("eval_summary", &stevejob::bridge::eval_summary, py::arg("payload"), py::arg("max_residuals"), py::arg("max_anomalies"));
    module.def("validate_tiny_inputs", &stevejob::bridge::validate_tiny_inputs, py::arg("inputs"), py::arg("max_rows"), py::arg("max_cols"));
    module.def("single_layer", &stevejob::bridge::single_layer, py::arg("checkpoint"));
    module.def("pack_checkpoint", &stevejob::bridge::pack_checkpoint, py::arg("layers"), py::arg("outputs"));
    module.def("omit", &stevejob::bridge::omit, py::arg("mapping"), py::arg("keys"));
    module.def("parse_connection_parts", &stevejob::bridge::parse_connection_parts, py::arg("raw"));
    module.def("canonical_ms_headers", &stevejob::bridge::canonical_ms_headers, py::arg("headers"));
    module.def("canonical_resource", &stevejob::bridge::canonical_resource, py::arg("account"), py::arg("resource_path"), py::arg("query"));
    module.def("query_string", &stevejob::bridge::query_string, py::arg("query"));
    module.def(
        "blob_exchange",
        [](const std::string& connection,
           const std::string& method,
           const std::string& path,
           py::dict query,
           py::bytes body,
           const std::string& content_type) {
            std::vector<stevejob::storage::QueryItem> items;
            for (auto item : query) {
                items.push_back(stevejob::storage::QueryItem{
                    py::cast<std::string>(py::str(item.first)),
                    py::cast<std::string>(py::str(item.second)),
                });
            }
            const char* data = PyBytes_AsString(body.ptr());
            const auto size = static_cast<std::size_t>(PyBytes_Size(body.ptr()));
            std::string payload(data == nullptr ? "" : data, data == nullptr ? 0 : size);
            std::string response;
            {
                py::gil_scoped_release release;
                response = stevejob::bridge::blob_exchange(connection, method, path, items, payload, content_type);
            }
            return py::bytes(response);
        },
        py::arg("connection"),
        py::arg("method"),
        py::arg("path"),
        py::arg("query"),
        py::arg("body"),
        py::arg("content_type"),
        "Signed Azurite request. Python passes the path and receives the response bytes.");
    module.def(
        "storage_json_load",
        [](py::bytes raw) {
            const char* data = PyBytes_AsString(raw.ptr());
            const auto size = static_cast<std::size_t>(PyBytes_Size(raw.ptr()));
            return stevejob::bridge::storage_json_load(std::string(data == nullptr ? "" : data, data == nullptr ? 0 : size));
        },
        py::arg("raw"));
    module.def(
        "storage_json_dump",
        [](py::handle value) { return py::bytes(stevejob::bridge::storage_json_dump(value)); },
        py::arg("value"));
    module.def(
        "content_checksum",
        [](py::bytes raw) {
            const char* data = PyBytes_AsString(raw.ptr());
            const auto size = static_cast<std::size_t>(PyBytes_Size(raw.ptr()));
            return stevejob::bridge::content_checksum(std::string(data == nullptr ? "" : data, data == nullptr ? 0 : size));
        },
        py::arg("payload"));
    py::class_<stevejob::models::RegressionScore>(module, "RegressionScore")
        .def_readonly("loss", &stevejob::models::RegressionScore::loss)
        .def_readonly("max_residual", &stevejob::models::RegressionScore::max_residual)
        .def_readonly("std_residual", &stevejob::models::RegressionScore::std_residual);
    py::class_<stevejob::models::RegressionMLPModel>(module, "RegressionMLPModel")
        .def(py::init<std::vector<std::vector<double>>,
                      std::vector<double>,
                      std::vector<std::vector<double>>,
                      std::vector<double>>(),
             py::arg("first_weights"),
             py::arg("first_bias"),
             py::arg("second_weights"),
             py::arg("second_bias"))
        .def("forward", &stevejob::models::RegressionMLPModel::forward)
        .def("evaluate_mse",
             [](const stevejob::models::RegressionMLPModel& model, int sample_count) {
                 stevejob::models::RegressionScore score;
                 {
                     py::gil_scoped_release release;
                     score = model.evaluate_mse(sample_count);
                 }
                 py::dict out;
                 out["loss"] = score.loss;
                 out["max_residual"] = score.max_residual;
                 out["std_residual"] = score.std_residual;
                 return out;
             },
             py::arg("sample_count") = 50);
    py::class_<stevejob::messaging::TotrEngine>(module, "TotrEngine")
        .def_static("encode_matrix", &stevejob::messaging::TotrEngine::encode_matrix, py::arg("matrix"), py::arg("rank") = 1)
        .def_static("decode_matrix", &stevejob::messaging::TotrEngine::decode_matrix, py::arg("encoded"), py::arg("rows"), py::arg("cols"))
        .def_static("encode_vector", &stevejob::messaging::TotrEngine::encode_vector, py::arg("values"))
        .def_static("decode_vector", &stevejob::messaging::TotrEngine::decode_vector, py::arg("text"), py::arg("count"));
    py::class_<stevejob::storage::SlidingJournal>(module, "SlidingJournal")
        .def(py::init<std::size_t>(), py::arg("max_history") = 5)
        .def("add_entry", &stevejob::storage::SlidingJournal::add_entry)
        .def("set_capacity", &stevejob::storage::SlidingJournal::set_capacity)
        .def("serialize", &stevejob::storage::SlidingJournal::serialize)
        .def("deserialize", &stevejob::storage::SlidingJournal::deserialize)
        .def("size", &stevejob::storage::SlidingJournal::size);
    module.def("read_state_native",
               &stevejob::bridge::read_state_native,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               "Download a checkpoint and journal in C++ and return TOTR text plus a receipt.");
    module.def("read_totr_state_native",
               &stevejob::bridge::read_state_native,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               "Download a checkpoint and sliding journal in C++ and return TOTR text plus a receipt.");
    module.def("score_totr_candidate",
               &stevejob::bridge::score_totr_candidate,
               py::arg("l1_w"),
               py::arg("l2_w"),
               py::arg("l1_b"),
               py::arg("l2_b"),
               "Score one 2-4-1 TOTR candidate in memory. Does not upload.");
    module.def("commit_winner_native",
               &stevejob::bridge::commit_winner_native,
               py::arg("winning_comp_l1_w"),
               py::arg("winning_comp_l2_w"),
               py::arg("winning_l1_b"),
               py::arg("winning_l2_b"),
               py::arg("target_blob"),
               py::arg("journal_blob") = "regression_journal.json",
               "Score one 2-4-1 TOTR candidate. Upload it only when its MSE is lower.");
    module.def("seed_stack_native",
               &stevejob::bridge::ModularOrchestrator::seed_shapes,
               py::arg("shapes"),
               py::arg("target_blob"),
               py::arg("journal_blob"),
               "Build a NeuralModel from layer shapes, score it, and store the SJC1 checkpoint.");
    module.def("read_model_meta",
               &stevejob::bridge::ModularOrchestrator::read_meta,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               "Return layer-slice names and the network loss. No weight matrices.");
    module.def("read_layer_slice",
               &stevejob::bridge::ModularOrchestrator::read_slice,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               py::arg("slice_id"),
               "Return one TOTR slice, its shape, and a short bias string.");
    module.def("commit_slice_native",
               &stevejob::bridge::ModularOrchestrator::commit_slice,
               py::arg("slice_id"),
               py::arg("totr_w"),
               py::arg("bias_str"),
               py::arg("target_blob"),
               py::arg("journal_blob"),
               "Patch one slice, score the full stack, and store it only when loss drops.");

    py::enum_<stevejob::models::ActivationType>(module, "ActivationType")
        .value("Linear", stevejob::models::ActivationType::Linear)
        .value("ReLU", stevejob::models::ActivationType::ReLU)
        .value("Tanh", stevejob::models::ActivationType::Tanh)
        .value("Sigmoid", stevejob::models::ActivationType::Sigmoid)
        .export_values();
    py::enum_<stevejob::models::LossType>(module, "LossType")
        .value("MeanSquaredError", stevejob::models::LossType::MeanSquaredError)
        .value("CrossEntropy", stevejob::models::LossType::CrossEntropy)
        .value("MeanAbsoluteError", stevejob::models::LossType::MeanAbsoluteError)
        .export_values();
    py::class_<stevejob::models::NeuralModel>(module, "NeuralModel")
        .def(py::init<std::string>(), py::arg("model_name") = "Model")
        .def("add_layer",
             &stevejob::models::NeuralModel::add_layer,
             py::arg("name"),
             py::arg("in_dim"),
             py::arg("out_dim"),
             py::arg("activation") = stevejob::models::ActivationType::ReLU,
             py::return_value_policy::reference)
        .def("set_loss",
             &stevejob::models::NeuralModel::set_loss,
             py::arg("loss_type"),
             py::return_value_policy::reference)
        .def("initialize_default_weights", &stevejob::models::NeuralModel::initialize_default_weights)
        .def("seed_to_storage",
             [](stevejob::models::NeuralModel& model, const std::string& target_blob, const std::string& journal_blob) {
                 return stevejob::bridge::ModularOrchestrator::seed_model(model, target_blob, journal_blob);
             },
             py::arg("target_blob"),
             py::arg("journal_blob"));

    py::class_<stevejob::models::MlpRegressor>(module, "MlpRegressor")
        .def(py::init<>())
        .def("seed_to_storage",
             [](stevejob::models::MlpRegressor& model,
                const std::string& target_blob,
                const std::string& journal_blob,
                std::size_t journal_capacity) {
                 return stevejob::bridge::EngineGateway::seed_model(model, target_blob, journal_blob, journal_capacity);
             },
             py::arg("target_blob"),
             py::arg("journal_blob"),
             py::arg("journal_capacity") = 10);
    py::class_<stevejob::models::BottleneckAutoencoder>(module, "BottleneckAutoencoder")
        .def(py::init<>())
        .def("seed_to_storage",
             [](stevejob::models::BottleneckAutoencoder& model,
                const std::string& target_blob,
                const std::string& journal_blob,
                std::size_t journal_capacity) {
                 return stevejob::bridge::EngineGateway::seed_model(model, target_blob, journal_blob, journal_capacity);
             },
             py::arg("target_blob"),
             py::arg("journal_blob"),
             py::arg("journal_capacity") = 10);
    module.def("read_architecture_meta",
               &stevejob::bridge::EngineGateway::read_meta,
               py::arg("target_blob"),
               py::arg("journal_blob"));
    module.def("read_architecture_slice",
               &stevejob::bridge::EngineGateway::read_slice,
               py::arg("target_blob"),
               py::arg("journal_blob"),
               py::arg("slice_id"));
    module.def(
        "commit_tournament",
        [](const std::string& slice_id,
           py::list candidates,
           const std::string& target_blob,
           const std::string& journal_blob,
           std::size_t journal_capacity) {
            std::vector<stevejob::bridge::TournamentCandidate> parsed;
            for (py::handle item : candidates) {
                const py::dict row = item.cast<py::dict>();
                parsed.push_back(stevejob::bridge::TournamentCandidate{
                    row["name"].cast<std::string>(),
                    row["w"].cast<std::string>(),
                    row["b"].cast<std::string>(),
                });
            }
            return stevejob::bridge::EngineGateway::commit_tournament(
                slice_id, parsed, target_blob, journal_blob, journal_capacity);
        },
        py::arg("slice_id"),
        py::arg("candidates"),
        py::arg("target_blob"),
        py::arg("journal_blob"),
        py::arg("journal_capacity") = 10);
}
