#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "stevejob/bridge/bridge_helpers.hpp"
#include "stevejob/bridge/engine_gateway.hpp"
#include "stevejob/bridge/native_pipeline.hpp"
#include "stevejob/compiler/candidate_reader.hpp"
#include "stevejob/models/neural_network.hpp"
#include "stevejob/math/linear_algebra.hpp"
#include "stevejob/math/tensor_stats.hpp"
#include "stevejob/messaging/sjc1_scheme.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tensor_engine, module) {
    module.doc() = "Native C++ tensor steps for a tiny perceptron workflow";

    py::module_ math = module.def_submodule("math", "Layer algebra and matrix statistics");
    py::class_<stevejob::math::TensorStats>(math, "TensorStats")
        .def(py::init<>())
        .def("compute", &stevejob::math::TensorStats::compute)
        .def_static("from_matrix", &stevejob::math::TensorStats::from_matrix, py::arg("matrix"))
        .def_property_readonly("mean", &stevejob::math::TensorStats::mean)
        .def_property_readonly("variance", &stevejob::math::TensorStats::variance)
        .def_property_readonly("max_val", &stevejob::math::TensorStats::max_val)
        .def_property_readonly("min_val", &stevejob::math::TensorStats::min_val)
        .def_property_readonly("count", &stevejob::math::TensorStats::count);
    math.def("compute_layer_stats",
             &stevejob::math::TensorStats::from_matrix,
             py::arg("matrix"),
             "Mean, population variance, and min/max for one layer matrix");
    math.def("apply_gradient_step",
             &stevejob::math::apply_gradient_step,
             py::arg("weights"),
             py::arg("gradients"),
             py::arg("learning_rate"),
             py::arg("weight_decay") = 0.0001,
             py::arg("dampening") = 1.0,
             "Applies w = w - lr * dampening * (grad + decay * w)");
    math.def("linear_outputs",
             &stevejob::math::linear_outputs,
             py::arg("inputs"),
             py::arg("weights"),
             py::arg("biases"),
             "y = bias + weights · inputs for one layer");

    py::class_<stevejob::messaging::SJC1Compressor>(module, "SJC1Compressor")
        .def_static(
            "pack",
            [](const std::vector<std::string>& names,
               const std::vector<std::vector<std::vector<double>>>& weights,
               const std::vector<std::vector<double>>& biases,
               const std::vector<std::vector<double>>& outputs) {
                stevejob::models::NeuralNetwork model("checkpoint");
                if (names.size() != weights.size() || names.size() != biases.size()) {
                    throw std::invalid_argument("layer names, weights, and biases must match");
                }
                for (std::size_t index = 0; index < names.size(); ++index) {
                    const std::size_t rows = weights[index].size();
                    const std::size_t cols = biases[index].size();
                    const bool last = index + 1 == names.size();
                    model.add_layer(names[index], rows, cols,
                                    last ? stevejob::models::Activations::linear() : stevejob::models::Activations::relu());
                    model.layers.back().weights = weights[index];
                    model.layers.back().biases = biases[index];
                }
                model.outputs = outputs;
                std::string packed;
                {
                    py::gil_scoped_release release;
                    packed = stevejob::messaging::SJC1Compressor::instance().pack(model);
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
                return stevejob::messaging::SJC1Compressor::document(bytes);
            },
            py::arg("blob"));

    module.def("coerce_matrix", &stevejob::bridge::ValueCodec::coerce_matrix, py::arg("value"), py::arg("label"), py::arg("allow_none") = false);
    module.def("coerce_vector", &stevejob::bridge::ValueCodec::coerce_vector, py::arg("value"), py::arg("label"), py::arg("allow_none") = false);
    module.def("parameter_count", [](py::handle layers, std::size_t max_parameters) {
        return stevejob::bridge::CheckpointBridge::collect_parameters(layers, true, max_parameters).size();
    }, py::arg("layers"), py::arg("max_parameters"));
    module.def("checkpoint_summary", &stevejob::bridge::CheckpointBridge::checkpoint_summary, py::arg("layers"), py::arg("max_parameters"));
    module.def("layer_diagnostics", &stevejob::bridge::CheckpointBridge::layer_diagnostics, py::arg("layers"));
    module.def("update_checkpoint", &stevejob::bridge::CheckpointBridge::update_checkpoint,
               py::arg("layers"), py::arg("gradients"), py::arg("learning_rate"), py::arg("weight_decay"), py::arg("dampening"));
    module.def("receipt_stats", &stevejob::bridge::CheckpointBridge::receipt_stats, py::arg("layers"), py::arg("outputs"));
    module.def("output_count", &stevejob::bridge::CheckpointBridge::output_count, py::arg("outputs"));
    module.def("eval_summary", &stevejob::bridge::EvaluationBridge::summarize, py::arg("payload"), py::arg("max_residuals"), py::arg("max_anomalies"));
    module.def("validate_tiny_inputs", &stevejob::bridge::CheckpointBridge::validate_tiny_inputs, py::arg("inputs"), py::arg("max_rows"), py::arg("max_cols"));
    module.def("single_layer", &stevejob::bridge::CheckpointBridge::single_layer, py::arg("checkpoint"));
    module.def("pack_checkpoint", &stevejob::bridge::CheckpointBridge::pack_checkpoint, py::arg("layers"), py::arg("outputs"));
    module.def("omit", &stevejob::bridge::ValueCodec::omit, py::arg("mapping"), py::arg("keys"));
    module.def("parse_connection_parts", &stevejob::bridge::SharedKeyText::parse_connection_parts, py::arg("raw"));
    module.def("canonical_ms_headers", &stevejob::bridge::SharedKeyText::canonical_ms_headers, py::arg("headers"));
    module.def("canonical_resource", &stevejob::bridge::SharedKeyText::canonical_resource, py::arg("account"), py::arg("resource_path"), py::arg("query"));
    module.def("query_string", &stevejob::bridge::SharedKeyText::query_string, py::arg("query"));
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
                response = stevejob::bridge::StorageExchange::exchange(connection, method, path, items, payload, content_type);
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
            return stevejob::bridge::StorageExchange::load(std::string(data == nullptr ? "" : data, data == nullptr ? 0 : size));
        },
        py::arg("raw"));
    module.def(
        "storage_json_dump",
        [](py::handle value) { return py::bytes(stevejob::bridge::StorageExchange::dump(value)); },
        py::arg("value"));
    module.def(
        "content_checksum",
        [](py::bytes raw) {
            const char* data = PyBytes_AsString(raw.ptr());
            const auto size = static_cast<std::size_t>(PyBytes_Size(raw.ptr()));
            return stevejob::bridge::StorageExchange::checksum(std::string(data == nullptr ? "" : data, data == nullptr ? 0 : size));
        },
        py::arg("payload"));
    py::class_<stevejob::models::ModelEvaluation>(module, "ModelEvaluation")
        .def_readonly("primary_loss", &stevejob::models::ModelEvaluation::primary_loss)
        .def("get_metric_value", &stevejob::models::ModelEvaluation::get_metric_value, py::arg("key"), py::arg("fallback") = 0.0);
    py::class_<stevejob::messaging::TotrEngine>(module, "TotrEngine")
        .def_static("encode_matrix", &stevejob::messaging::TotrEngine::encode_matrix, py::arg("matrix"), py::arg("rank") = 1)
        .def_static("decode_matrix", &stevejob::messaging::TotrEngine::decode_matrix, py::arg("encoded"), py::arg("rows"), py::arg("cols"))
        .def_static("encode_vector", &stevejob::messaging::TotrEngine::encode_vector, py::arg("values"))
        .def_static("decode_vector", &stevejob::messaging::TotrEngine::decode_vector, py::arg("text"), py::arg("count"))
        .def_static("integer_distance", &stevejob::messaging::TotrEngine::integer_distance, py::arg("candidate"), py::arg("active"))
        .def_static("clamp_int", &stevejob::messaging::TotrEngine::clamp_int, py::arg("code"), py::arg("bound") = 99);
    py::class_<stevejob::storage::SlidingJournal>(module, "SlidingJournal")
        .def(py::init<std::size_t>(), py::arg("max_history") = 5)
        .def("add_entry", &stevejob::storage::SlidingJournal::add_entry)
        .def("set_capacity", &stevejob::storage::SlidingJournal::set_capacity)
        .def("serialize", &stevejob::storage::SlidingJournal::serialize)
        .def("deserialize", &stevejob::storage::SlidingJournal::deserialize)
        .def("size", &stevejob::storage::SlidingJournal::size);
    module.def("read_state_native",
               &stevejob::bridge::Tournament::read_checkpoint,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               "Download a checkpoint and journal in C++ and return the model's tokens plus journal metrics.");
    module.def("read_totr_state_native",
               &stevejob::bridge::Tournament::read_checkpoint,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               "Download a checkpoint and sliding journal in C++ and return the model's tokens plus journal metrics.");
    module.def(
        "score_totr_candidate",
        [](const std::string& l1_w, const std::string& l2_w, const std::string& l1_b, const std::string& l2_b) {
            return stevejob::bridge::Tournament::score_layers({l1_w, l2_w}, {l1_b, l2_b}, {{2, 4}, {4, 1}});
        },
        py::arg("l1_w"),
        py::arg("l2_w"),
        py::arg("l1_b"),
        py::arg("l2_b"),
        "Score one 2-4-1 candidate with the regressor's own evaluation. Does not upload.");
    module.def(
        "commit_winner_native",
        [](const std::string& winning_comp_l1_w,
           const std::string& winning_comp_l2_w,
           const std::string& winning_l1_b,
           const std::string& winning_l2_b,
           const std::string& target_blob,
           const std::string& journal_blob) {
            return stevejob::bridge::EngineGateway::commit_layers(
                {winning_comp_l1_w, winning_comp_l2_w},
                {winning_l1_b, winning_l2_b},
                {{2, 4}, {4, 1}},
                target_blob,
                journal_blob,
                5);
        },
        py::arg("winning_comp_l1_w"),
        py::arg("winning_comp_l2_w"),
        py::arg("winning_l1_b"),
        py::arg("winning_l2_b"),
        py::arg("target_blob"),
        py::arg("journal_blob") = "regression_journal.json",
        "Score one 2-4-1 candidate through the tournament and store it only when loss drops.");
    module.def("seed_stack_native",
               [](const std::vector<std::vector<int>>& shapes,
                  const std::string& target_blob,
                  const std::string& journal_blob) {
                   return stevejob::models::ArchitectureCatalog::visit(
                       stevejob::models::NeuralNetwork::from_shapes(shapes),
                       [&](auto& model) {
                           return stevejob::bridge::EngineGateway::seed_model(model, target_blob, journal_blob, 5);
                       });
               },
               py::arg("shapes"),
               py::arg("target_blob"),
               py::arg("journal_blob"),
               "Build a NeuralNetwork from layer shapes, score it, and store the SJC1 checkpoint.");
    module.def("read_model_meta",
               &stevejob::bridge::EngineGateway::read_meta,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               "Return layer-slice names and the network loss. No weight matrices.");
    module.def("read_layer_slice",
               &stevejob::bridge::EngineGateway::read_slice,
               py::arg("checkpoint_blob"),
               py::arg("journal_blob"),
               py::arg("slice_id"),
               "Return one slice: shape, the model's parameter tokens, probe, and candidate scheme.");
    module.def("commit_slice_native",
               [](const std::string& slice_id,
                  const std::string& totr_w,
                  const std::string& bias_str,
                  const std::string& target_blob,
                  const std::string& journal_blob) {
                   return stevejob::bridge::EngineGateway::commit_tournament(
                       slice_id,
                       {stevejob::bridge::TournamentCandidate{slice_id, totr_w, bias_str}},
                       target_blob,
                       journal_blob,
                       5);
               },
               py::arg("slice_id"),
               py::arg("totr_w"),
               py::arg("bias_str"),
               py::arg("target_blob"),
               py::arg("journal_blob"),
               "Patch one slice, score the full stack, and store it only when loss drops.");

    py::class_<stevejob::models::IActivation>(module, "IActivation")
        .def("name", &stevejob::models::IActivation::name)
        .def("apply", &stevejob::models::IActivation::apply)
        .def("derivative", &stevejob::models::IActivation::derivative);
    py::class_<stevejob::models::LinearActivation, stevejob::models::IActivation>(module, "LinearActivation")
        .def(py::init<>());
    py::class_<stevejob::models::ReLUActivation, stevejob::models::IActivation>(module, "ReLUActivation")
        .def(py::init<>());
    py::class_<stevejob::models::TanhActivation, stevejob::models::IActivation>(module, "TanhActivation")
        .def(py::init<>());
    py::class_<stevejob::models::SigmoidActivation, stevejob::models::IActivation>(module, "SigmoidActivation")
        .def(py::init<>());
    py::class_<stevejob::models::NeuralNetwork>(module, "NeuralNetwork")
        .def(py::init<std::string>(), py::arg("model_name") = "Model")
        .def(
            "add_layer",
            [](stevejob::models::NeuralNetwork& model,
               const std::string& name,
               std::size_t in_dim,
               std::size_t out_dim,
               py::object activation) {
                if (activation.is_none()) {
                    return model.add_layer(name, in_dim, out_dim);
                }
                return model.add_layer(name, in_dim, out_dim, activation.cast<const stevejob::models::IActivation&>());
            },
            py::arg("name"),
            py::arg("in_dim"),
            py::arg("out_dim"),
            py::arg("activation") = py::none(),
            py::return_value_policy::reference)
        .def("initialize_default_weights", &stevejob::models::NeuralNetwork::initialize_default_weights)
        .def("seed_to_storage",
             [](stevejob::models::NeuralNetwork& model, const std::string& target_blob, const std::string& journal_blob) {
                 return stevejob::models::ArchitectureCatalog::visit(
                     stevejob::models::NeuralNetwork(model),
                     [&](auto& arch) {
                         return stevejob::bridge::EngineGateway::seed_model(arch, target_blob, journal_blob, 5);
                     });
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
    py::class_<stevejob::models::DemoScaleModel>(module, "DemoScaleModel")
        .def(py::init<>())
        .def("seed_to_storage",
             [](stevejob::models::DemoScaleModel& model,
                const std::string& target_blob,
                const std::string& journal_blob,
                std::size_t journal_capacity) {
                 return stevejob::bridge::EngineGateway::seed_model(model, target_blob, journal_blob, journal_capacity);
             },
             py::arg("target_blob"),
             py::arg("journal_blob"),
             py::arg("journal_capacity") = 5);
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
            parsed.reserve(static_cast<std::size_t>(candidates.size()));
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
    module.def(
        "compile_prompt",
        [](const std::string& target_blob,
           const std::string& journal_blob,
           const std::string& slice_id,
           std::size_t visible_history,
           int int_bound,
           const std::string& retry_note) {
            return stevejob::bridge::EngineGateway::compile_prompt(
                target_blob, journal_blob, slice_id, visible_history, int_bound, retry_note);
        },
        py::arg("target_blob"),
        py::arg("journal_blob"),
        py::arg("slice_id"),
        py::arg("visible_history") = 3,
        py::arg("int_bound") = 25,
        py::arg("retry_note") = "");
    module.def(
        "extract_candidates",
        [](const std::string& architecture,
           const std::string& text,
           int rows,
           int cols,
           const py::dict& active,
           const py::dict& scheme) {
            try {
                return stevejob::compiler::CandidateReader::extract(architecture, text, rows, cols, active, scheme);
            } catch (const std::invalid_argument& exc) {
                throw py::value_error(exc.what());
            }
        },
        py::arg("architecture"),
        py::arg("text"),
        py::arg("rows"),
        py::arg("cols"),
        py::arg("active"),
        py::arg("scheme") = py::dict(),
        "Read a completion with the architecture's candidate scheme. Weight and bias tokens are accepted by that model's encodings.");
    module.def(
        "read_journal_text",
        [](const std::string& journal_blob) {
            stevejob::storage::AzuriteClient client;
            return client.download_blob("evals", stevejob::storage::BlobPath::bare(journal_blob, "evals/"));
        },
        py::arg("journal_blob"),
        "Download the journal document the model stored.");
}
