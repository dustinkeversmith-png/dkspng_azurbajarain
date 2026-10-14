#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/description/number_words.hpp"
#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/text/english_list_formatter.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Writes the four observation fields from measured facts only; used wherever the decoder's text
// cannot be verified against the telemetry.
struct ObservationNarrator {
    static ObservationText describe(const NetworkObservations& obs) {
        ObservationText text;
        text.pathway_activity = pathway_activity(obs);
        text.feature_sensitivity = feature_sensitivity(obs);
        text.representational_structure = representational_structure(obs);
        text.functional_mapping = functional_mapping(obs);
        return text;
    }

    static std::string pathway_activity(const NetworkObservations& obs) {
        const GatingObservation& g = obs.gating;
        const std::string width = obs.layers.empty() ? "" : NumberWords::word(obs.layers.front().width);
        if (g.present()) {
            const std::string closed = NumberWords::word(g.closed_samples.size());
            const std::string open = NumberWords::word(g.open_samples.size());
            std::string out;
            if (g.has_gate_coordinate) {
                const std::string x = "x" + std::to_string(g.gate_coordinate);
                out = "The sign of " + x + " gates the first layer. On the " + closed + " samples where " + x +
                      " is negative every one of the " + width + " units returns exactly zero, while the " + open +
                      " samples with positive " + x + " open the layer.";
            } else {
                out = NumberWords::capitalized(g.closed_samples.size()) + " of the " +
                      NumberWords::word(obs.shape.num_samples) +
                      " samples leave every first-layer unit at exactly zero, while the remaining " + open +
                      " samples open the layer.";
            }
            if (g.deeper_layers_constant) {
                out += " In the deeper layers the closed samples share one identical activation pattern.";
            }
            return out;
        }

        std::string out;
        for (std::size_t l = 0; l < obs.layers.size(); ++l) {
            const LayerObservation& layer = obs.layers[l];
            const std::string name = layer_adjective(l);
            std::string sentence;
            if (!layer.dead_units.empty()) {
                sentence = NumberWords::capitalized(layer.dead_units.size()) + " of the " +
                           NumberWords::word(layer.width) + " " + name + " units";
                if (layer.dead_units.size() <= 12) {
                    sentence += " (units " + EnglishListFormatter::indices(layer.dead_units) + ")";
                }
                sentence += " output exactly zero on every sample, leaving " + NumberWords::word(layer.live_count()) +
                            " live units.";
            } else if (layer.always_active_units.size() == layer.width) {
                sentence = "Every one of the " + NumberWords::word(layer.width) + " " + name +
                           " units is active on every sample.";
            } else {
                sentence = "All " + NumberWords::word(layer.width) + " " + name +
                           " units fire on at least part of the batch.";
            }
            if (layer.lies_on_ray) {
                sentence += " The active units are fixed multiples of one per-sample intensity, so the layer carries "
                            "a single signal.";
            }
            out += (out.empty() ? "" : " ") + sentence;
        }
        return out;
    }

    static std::string feature_sensitivity(const NetworkObservations& obs) {
        const GatingObservation& g = obs.gating;
        std::vector<std::size_t> rises;
        std::vector<std::size_t> falls;
        std::vector<std::size_t> secondary;
        for (const InputDriver& d : obs.drivers) {
            if (g.has_gate_coordinate && d.coordinate == g.gate_coordinate) {
                continue;
            }
            if (d.axis == 1) {
                secondary.push_back(d.coordinate);
            } else {
                (d.direction > 0 ? rises : falls).push_back(d.coordinate);
            }
        }

        if (g.has_gate_coordinate) {
            std::string out = "x" + std::to_string(g.gate_coordinate) +
                              " acts as a switch on the whole pathway: its sign decides whether the first layer "
                              "responds at all.";
            std::vector<std::size_t> shaping = rises;
            shaping.insert(shaping.end(), falls.begin(), falls.end());
            if (!shaping.empty()) {
                out += " Within the positive half-space the response is shaped mostly by " +
                       EnglishListFormatter::coordinates(shaping) + ".";
            }
            return out;
        }
        if (rises.empty() && falls.empty()) {
            return "No single input coordinate shows a consistent relation to the hidden activity; sensitivity is "
                   "spread across many inputs.";
        }

        std::string out = "The leading hidden axis";
        if (!rises.empty()) {
            out += " rises with " + EnglishListFormatter::coordinates(rises);
        }
        if (!falls.empty()) {
            out += std::string(rises.empty() ? "" : " and") + " falls with " + EnglishListFormatter::coordinates(falls);
        }
        out += ".";
        if (!secondary.empty()) {
            out += " A secondary axis follows " + EnglishListFormatter::coordinates(secondary) + ".";
        }
        const bool ray = !obs.layers.empty() && obs.layers.front().lies_on_ray;
        out += ray ? " Any input change that leaves that projection unchanged leaves every hidden unit unchanged."
                   : " The remaining coordinates show no consistent relation to the hidden activity.";
        return out;
    }

    static std::string representational_structure(const NetworkObservations& obs) {
        if (obs.layers.empty()) {
            return "";
        }
        const LayerObservation& first = obs.layers.front();
        const GatingObservation& g = obs.gating;
        if (g.present()) {
            return std::string("The batch splits into two groups: the closed samples collapse onto the origin of the "
                               "first layer") +
                   (g.deeper_layers_constant ? " and onto one shared pattern deeper in the network" : "") +
                   ", while the open samples spread over " +
                   (first.effective_rank >= 2 ? "a multi-dimensional region." : "a single direction.");
        }
        if (first.lies_on_ray) {
            std::string out = "The first-layer activations lie on a single ray from the origin, and samples differ "
                              "only in their distance along it.";
            for (std::size_t l = 1; l < obs.layers.size(); ++l) {
                if (obs.layers[l].lies_on_ray) {
                    out += " The " + layer_ordinal(l) + " layer maps that ray onto another ray, adding no new "
                           "dimensions.";
                }
            }
            return out;
        }
        std::string out = "The first-layer activations occupy " + dimension_phrase(first.effective_rank);
        out += first.dead_units.empty() ? ", and the samples remain distinct from one another."
                                        : ", and the dead coordinates add no separation between samples.";
        return out;
    }

    static std::string functional_mapping(const NetworkObservations& obs) {
        if (obs.outputs.empty()) {
            return "";
        }
        if (obs.gating.present()) {
            bool pinned = true;
            double open_corr = 0.0;
            double closed_share = 0.0;
            for (const OutputObservation& o : obs.outputs) {
                pinned = pinned && o.constant_on_closed_samples;
                open_corr += o.open_side_correlation / static_cast<double>(obs.outputs.size());
                closed_share += o.closed_error_share / static_cast<double>(obs.outputs.size());
            }
            if (pinned) {
                std::string out = "On the closed side every output is pinned to one constant regardless of the "
                                  "target; on the open side the predictions ";
                out += open_corr >= 0.9 ? "follow the targets closely" : "vary from sample to sample";
                out += closed_share >= 0.7 ? ", so most of the error sits on the closed side." : ".";
                return out;
            }
        }

        std::array<std::vector<std::size_t>, 6> groups;
        std::array<double, 6> amplitude_sum{};
        for (const OutputObservation& o : obs.outputs) {
            const std::size_t c = category(o);
            groups[c].push_back(o.index);
            amplitude_sum[c] += o.amplitude_ratio;
        }

        const bool single = obs.outputs.size() == 1;
        std::string out;
        if (obs.outputs_move_in_lockstep) {
            out = "All outputs are driven by one shared signal, so their predictions move in rigid proportion.";
        }
        for (std::size_t c = 0; c < groups.size(); ++c) {
            if (groups[c].empty()) {
                continue;
            }
            const bool plural = groups[c].size() > 1;
            const std::string subject =
                single ? "The prediction"
                       : (plural ? "Outputs " : "Output ") + EnglishListFormatter::indices(groups[c]);
            const std::string their = plural ? "their targets" : "its target";
            std::string clause;
            switch (c) {
                case 0:
                    clause = std::string(plural ? " keep" : " keeps") + " the sign and ordering of " + their + " but " +
                             (plural ? "are" : "is") + " damped to " +
                             fraction_phrase(amplitude_sum[c] / static_cast<double>(groups[c].size())) +
                             " of the target amplitude";
                    break;
                case 1: clause = std::string(plural ? " follow " : " follows ") + their + " closely"; break;
                case 2:
                    clause = std::string(plural ? " stay nearly flat and carry" : " stays nearly flat and carries") +
                             " no information about " + their;
                    break;
                case 3: clause = std::string(plural ? " follow " : " follows ") + their + " only partially"; break;
                case 4: clause = std::string(plural ? " move against " : " moves against ") + their; break;
                default: clause = std::string(plural ? " bear" : " bears") + " little relation to " + their; break;
            }
            out += (out.empty() ? "" : " ") + subject + clause + ".";
        }
        return out;
    }

private:
    // 0 damped but sign/order kept, 1 tracks closely, 2 flat, 3 partial, 4 inverted, 5 unrelated.
    static std::size_t category(const OutputObservation& o) {
        if (o.sign_preserved && o.order_preserved && o.amplitude_ratio < 0.7) {
            return 0;
        }
        if (o.correlation >= 0.95) {
            return 1;
        }
        if (o.flat) {
            return 2;
        }
        if (o.correlation >= 0.6) {
            return 3;
        }
        if (o.correlation <= -0.3) {
            return 4;
        }
        return 5;
    }

    static std::string fraction_phrase(double ratio) {
        if (ratio < 0.2) {
            return "a small fraction";
        }
        if (ratio < 0.3) {
            return "roughly a quarter";
        }
        if (ratio < 0.42) {
            return "roughly a third";
        }
        if (ratio < 0.6) {
            return "roughly half";
        }
        return "most";
    }

    static std::string dimension_phrase(std::size_t rank) {
        if (rank <= 1) {
            return "an essentially one-dimensional band";
        }
        if (rank == 2) {
            return "a two-dimensional sheet";
        }
        return "a " + NumberWords::word(rank) + "-dimensional region";
    }

    static std::string layer_ordinal(std::size_t layer) {
        static const std::array<std::string, 4> names = {"first", "second", "third", "fourth"};
        return layer < names.size() ? names[layer] : "layer-" + std::to_string(layer + 1);
    }

    static std::string layer_adjective(std::size_t layer) {
        return layer_ordinal(layer) + "-layer";
    }
};

}  // namespace forward_analysis
