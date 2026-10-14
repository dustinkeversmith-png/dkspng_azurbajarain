#pragma once

#include "forward_analysis/analysis/network_observations.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace forward_analysis {

// Serializes the measured facts so they can be inspected or handed to a downstream LLM.
struct ObservationJsonWriter {
    static nlohmann::json to_json(const NetworkObservations& obs) {
        nlohmann::json out;
        out["shape"] = {{"num_samples", obs.shape.num_samples},
                        {"input_dim", obs.shape.input_dim},
                        {"hidden_widths", obs.shape.hidden_widths},
                        {"output_dim", obs.shape.output_dim}};
        out["layers"] = nlohmann::json::array();
        for (const LayerObservation& layer : obs.layers) {
            out["layers"].push_back({{"width", layer.width},
                                     {"dead_units", layer.dead_units},
                                     {"always_active_units", layer.always_active_units},
                                     {"effective_rank", layer.effective_rank},
                                     {"lies_on_ray", layer.lies_on_ray}});
        }
        nlohmann::json gating = {{"closed_samples", obs.gating.closed_samples},
                                 {"open_samples", obs.gating.open_samples},
                                 {"deeper_layers_constant", obs.gating.deeper_layers_constant}};
        gating["gate_coordinate"] =
            obs.gating.has_gate_coordinate ? nlohmann::json(obs.gating.gate_coordinate) : nlohmann::json(nullptr);
        out["gating"] = gating;
        out["drivers"] = nlohmann::json::array();
        for (const InputDriver& d : obs.drivers) {
            out["drivers"].push_back({{"coordinate", d.coordinate},
                                      {"axis", d.axis == 0 ? "leading" : "secondary"},
                                      {"direction", d.direction > 0 ? "rises" : "falls"},
                                      {"strength", d.strength}});
        }
        out["target_related_coordinates"] = obs.target_related_coordinates;
        out["outputs"] = nlohmann::json::array();
        for (const OutputObservation& o : obs.outputs) {
            out["outputs"].push_back({{"index", o.index},
                                      {"sign_preserved", o.sign_preserved},
                                      {"order_preserved", o.order_preserved},
                                      {"amplitude_ratio", o.amplitude_ratio},
                                      {"correlation", o.correlation},
                                      {"flat", o.flat},
                                      {"constant_on_closed_samples", o.constant_on_closed_samples},
                                      {"open_side_correlation", o.open_side_correlation},
                                      {"closed_error_share", o.closed_error_share}});
        }
        out["outputs_move_in_lockstep"] = obs.outputs_move_in_lockstep;
        out["eigen_signs"] = std::vector<double>(obs.eigen_signs.data(), obs.eigen_signs.data() + obs.eigen_signs.size());
        return out;
    }

    static std::string write(const NetworkObservations& obs, int indent = 2) {
        return to_json(obs).dump(indent);
    }
};

}  // namespace forward_analysis
