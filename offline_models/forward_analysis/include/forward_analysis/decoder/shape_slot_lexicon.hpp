#pragma once

#include "forward_analysis/analysis/network_observations.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace forward_analysis {

// Named, measured quantities of one network that slot tokens refer to. Slot names are shared across
// networks (W_1 is always the first hidden width); the values come from that network's telemetry.
struct ShapeSlotLexicon {
    // Scalar sizes and counts, e.g. {"W_1": 24, "DEAD_L1": 5, "N": 8, "D_IN": 16, "CLOSED": 3}.
    std::map<std::string, std::size_t> counts;
    // Index lists, e.g. {"DEAD_L1": [3, 8, 14, 19, 22], "CLOSED": [1, 4, 5]}.
    std::map<std::string, std::vector<std::size_t>> lists;
    // Named input coordinates: GATE, RISE_k / FALL_k (leading-axis drivers by direction), SEC_k (secondary-axis
    // drivers), TGT_k (target-related coordinates), each k counted from 1 in strength order.
    std::map<std::string, std::size_t> coordinates;
    // Number of hidden layers (ORD:LAYER_k slots exist for k <= layer_count).
    std::size_t layer_count = 0;

    static ShapeSlotLexicon from(const NetworkObservations& obs) {
        ShapeSlotLexicon lex;
        lex.layer_count = obs.layers.size();
        lex.counts["N"] = obs.shape.num_samples;
        lex.counts["D_IN"] = obs.shape.input_dim;
        lex.counts["D_OUT"] = obs.shape.output_dim;
        lex.counts["D_IN_MINUS_1"] = obs.shape.input_dim - 1;
        lex.counts["DRIVERS"] = obs.drivers.size();
        lex.counts["OTHER_INPUTS"] = obs.shape.input_dim - std::min(obs.drivers.size(), obs.shape.input_dim);
        for (std::size_t l = 0; l < obs.layers.size(); ++l) {
            const LayerObservation& layer = obs.layers[l];
            const std::string suffix = "_L" + std::to_string(l + 1);
            std::vector<std::size_t> partial;
            for (std::size_t u = 0; u < layer.width; ++u) {
                const bool dead = std::find(layer.dead_units.begin(), layer.dead_units.end(), u) != layer.dead_units.end();
                const bool always = std::find(layer.always_active_units.begin(), layer.always_active_units.end(), u) !=
                                    layer.always_active_units.end();
                if (!dead && !always) {
                    partial.push_back(u);
                }
            }
            lex.counts["W_" + std::to_string(l + 1)] = layer.width;
            lex.counts["DEAD" + suffix] = layer.dead_units.size();
            lex.counts["LIVE" + suffix] = layer.live_count();
            lex.counts["ACTIVE" + suffix] = layer.always_active_units.size();
            lex.counts["PARTIAL" + suffix] = partial.size();
            lex.lists["DEAD" + suffix] = layer.dead_units;
            lex.lists["ACTIVE" + suffix] = layer.always_active_units;
            lex.lists["PARTIAL" + suffix] = partial;
        }
        if (obs.gating.present()) {
            lex.counts["CLOSED"] = obs.gating.closed_samples.size();
            lex.counts["OPEN"] = obs.gating.open_samples.size();
            lex.lists["CLOSED"] = obs.gating.closed_samples;
            lex.lists["OPEN"] = obs.gating.open_samples;
        }
        if (obs.gating.has_gate_coordinate) {
            lex.coordinates["GATE"] = obs.gating.gate_coordinate;
        }
        std::size_t rises = 0;
        std::size_t falls = 0;
        std::size_t secondary = 0;
        for (const InputDriver& d : obs.drivers) {
            if (d.axis == 1) {
                lex.coordinates["SEC_" + std::to_string(++secondary)] = d.coordinate;
            } else if (d.direction > 0) {
                lex.coordinates["RISE_" + std::to_string(++rises)] = d.coordinate;
            } else {
                lex.coordinates["FALL_" + std::to_string(++falls)] = d.coordinate;
            }
        }
        for (std::size_t k = 0; k < obs.target_related_coordinates.size(); ++k) {
            lex.coordinates["TGT_" + std::to_string(k + 1)] = obs.target_related_coordinates[k];
        }
        return lex;
    }

    // Name of the coordinate slot holding input index c, or "" when c has no measured role.
    std::string coordinate_name(std::size_t c) const {
        if (coordinates.count("GATE") > 0 && coordinates.at("GATE") == c) {
            return "GATE";
        }
        for (const auto& [name, value] : coordinates) {
            if (value == c) {
                return name;
            }
        }
        return "";
    }
};

}  // namespace forward_analysis
