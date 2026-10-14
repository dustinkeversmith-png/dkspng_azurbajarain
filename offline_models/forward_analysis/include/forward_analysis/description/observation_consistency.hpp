#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/description/number_words.hpp"
#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace forward_analysis {

// Checks the verifiable claims in observation text against measured facts: named coordinates are
// measured drivers, the gate or target-related coordinates, unit indices exist, counts attached to units / samples / outputs and
// "<count> of the <total>" pairs are real sizes, statements about dead units, rays, sheets, deeper
// layers, gating, preserved signs and close / flat outputs hold for the telemetry, and
// the text does not loop.
struct ObservationConsistency {
    static std::array<bool, ObservationText::kFieldCount> verify(const ObservationText& text,
                                                                 const NetworkObservations& obs) {
        std::array<bool, ObservationText::kFieldCount> out{};
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            out[f] = verify_field(text.field(f), obs);
        }
        return out;
    }

    static bool verify_field(const std::string& text, const NetworkObservations& obs) {
        return violation(text, obs).empty();
    }

    // The first claim in the text that the measured facts contradict; empty when every checked claim holds.
    static std::string violation(const std::string& text, const NetworkObservations& obs) {
        const std::vector<std::string> lex = LexemeSplitter::split(LexemeSplitter::lowercase(text));
        if (lex.empty()) {
            return "empty text";
        }
        for (const std::string& reason : {indices_exist(lex, obs), coordinates_measured(lex, obs),
                                          count_pairs_match(lex, obs), noun_counts_match(lex, obs),
                                          claims_hold(lex, obs), repetition(lex)}) {
            if (!reason.empty()) {
                return reason;
            }
        }
        return "";
    }

private:
    static bool all_digits(const std::string& s, std::size_t from = 0) {
        if (s.size() <= from) {
            return false;
        }
        return std::all_of(s.begin() + static_cast<std::ptrdiff_t>(from), s.end(),
                           [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; });
    }

    static bool is_coordinate(const std::string& token) {
        return token.size() > 1 && token[0] == 'x' && all_digits(token, 1);
    }

    static const std::set<std::string>& count_nouns() {
        static const std::set<std::string> nouns = {"units", "samples", "outputs", "coordinates", "inputs",
                                                    "first-layer", "second-layer", "hidden", "observations"};
        return nouns;
    }

    // lex[i] is the integer or fractional part of a decimal number split at its point, e.g. "1" "." "825".
    static bool is_decimal_part(const std::vector<std::string>& lex, std::size_t i) {
        return (i + 1 < lex.size() && LexemeSplitter::is_decimal_point(lex, i + 1)) ||
               (i > 0 && LexemeSplitter::is_decimal_point(lex, i - 1));
    }

    // lex[i] directly precedes a count noun, or precedes one with a single modifier between ("25 input coordinates").
    static bool before_count_noun(const std::vector<std::string>& lex, std::size_t i) {
        return (i + 1 < lex.size() && count_nouns().count(lex[i + 1]) > 0) ||
               (i + 2 < lex.size() && lex[i + 1] != "of" && count_nouns().count(lex[i + 2]) > 0);
    }

    static std::string indices_exist(const std::vector<std::string>& lex, const NetworkObservations& obs) {
        std::size_t largest = std::max(obs.shape.num_samples, obs.shape.output_dim);
        for (std::size_t w : obs.shape.hidden_widths) {
            largest = std::max(largest, w);
        }
        for (std::size_t i = 0; i < lex.size(); ++i) {
            const std::string& token = lex[i];
            if (is_coordinate(token) && std::stoul(token.substr(1)) >= obs.shape.input_dim) {
                return token + " exceeds input width " + std::to_string(obs.shape.input_dim);
            }
            if (all_digits(token) && !is_decimal_part(lex, i) && !before_count_noun(lex, i) &&
                std::stoul(token) >= largest) {
                return "index " + token + " exceeds every layer, output and sample count";
            }
        }
        return "";
    }

    static std::string coordinates_measured(const std::vector<std::string>& lex, const NetworkObservations& obs) {
        std::set<std::size_t> known;
        if (obs.gating.has_gate_coordinate) {
            known.insert(obs.gating.gate_coordinate);
        }
        for (const InputDriver& d : obs.drivers) {
            known.insert(d.coordinate);
        }
        known.insert(obs.target_related_coordinates.begin(), obs.target_related_coordinates.end());
        for (const std::string& token : lex) {
            if (is_coordinate(token) && known.count(static_cast<std::size_t>(std::stoul(token.substr(1)))) == 0) {
                return token + " is not the gate coordinate, a measured driver or a target-related coordinate";
            }
        }
        return "";
    }

    static std::string count_pairs_match(const std::vector<std::string>& lex, const NetworkObservations& obs) {
        std::set<std::pair<std::size_t, std::size_t>> allowed;
        for (const LayerObservation& layer : obs.layers) {
            allowed.insert({layer.dead_units.size(), layer.width});
            allowed.insert({layer.live_count(), layer.width});
            allowed.insert({layer.always_active_units.size(), layer.width});
        }
        allowed.insert({obs.gating.closed_samples.size(), obs.shape.num_samples});
        allowed.insert({obs.gating.open_samples.size(), obs.shape.num_samples});
        for (std::size_t i = 0; i + 3 < lex.size(); ++i) {
            std::size_t part = 0;
            std::size_t whole = 0;
            if (lex[i + 1] != "of" || (lex[i + 2] != "the" && lex[i + 2] != "its") ||
                !NumberWords::parse(lex[i], part) || !NumberWords::parse(lex[i + 3], whole)) {
                continue;
            }
            if (i > 0 && (lex[i - 1] == "every" || lex[i - 1] == "each")) {
                continue;
            }
            if (allowed.count({part, whole}) == 0) {
                return "\"" + lex[i] + " of " + lex[i + 2] + " " + lex[i + 3] + "\" matches no layer or sample split";
            }
        }
        return "";
    }

    static std::string noun_counts_match(const std::vector<std::string>& lex, const NetworkObservations& obs) {
        std::set<std::size_t> allowed = {1};
        for (const LayerObservation& layer : obs.layers) {
            allowed.insert({layer.width, layer.dead_units.size(), layer.live_count(), layer.always_active_units.size(),
                            layer.width - layer.always_active_units.size()});
        }
        const std::size_t drivers = obs.drivers.size();
        std::set<std::string> named;
        for (const std::string& token : lex) {
            if (is_coordinate(token)) {
                named.insert(token);
            }
        }
        allowed.insert({obs.shape.num_samples, obs.gating.closed_samples.size(), obs.gating.open_samples.size(),
                        obs.shape.output_dim, obs.shape.input_dim, drivers, obs.shape.input_dim - 1,
                        obs.shape.input_dim - std::min(drivers, obs.shape.input_dim),
                        obs.shape.input_dim - std::min(named.size(), obs.shape.input_dim)});
        for (std::size_t i = 0; i + 1 < lex.size(); ++i) {
            std::size_t value = 0;
            if (is_decimal_part(lex, i) || !NumberWords::parse(lex[i], value) || negated(lex, i)) {
                continue;
            }
            if (before_count_noun(lex, i) && allowed.count(value) == 0) {
                return "count \"" + lex[i] + "\" is not a size present in the telemetry";
            }
        }
        return "";
    }

    static bool negated(const std::vector<std::string>& lex, std::size_t i) {
        for (std::size_t back = 1; back <= 3 && back <= i; ++back) {
            const std::string& w = lex[i - back];
            if (w == "no" || w == "not" || w == "none" || w == "without") {
                return true;
            }
        }
        return false;
    }

    static std::string claims_hold(const std::vector<std::string>& lex, const NetworkObservations& obs) {
        bool any_dead = false;
        bool any_ray = false;
        for (const LayerObservation& layer : obs.layers) {
            any_dead = any_dead || !layer.dead_units.empty();
            any_ray = any_ray || layer.lies_on_ray;
        }
        bool any_sign_kept = false;
        bool any_close = false;
        bool any_flat = false;
        for (const OutputObservation& o : obs.outputs) {
            any_sign_kept = any_sign_kept || o.sign_preserved;
            any_close = any_close || std::max(o.correlation, o.open_side_correlation) >= 0.9;
            any_flat = any_flat || o.flat;
        }
        const bool multi_layer = obs.layers.size() >= 2;
        const bool planar = !obs.layers.empty() && !obs.layers.front().lies_on_ray &&
                            obs.layers.front().effective_rank >= 2;

        for (std::size_t i = 0; i < lex.size(); ++i) {
            if (negated(lex, i)) {
                continue;
            }
            const std::string& w = lex[i];
            const std::string next = i + 1 < lex.size() ? lex[i + 1] : "";
            const bool dead_claim =
                w == "dead" || w == "silent" || w == "dormant" ||
                (w == "never" && (next == "fire" || next == "fires" || next == "activate" || next == "leave"));
            if (dead_claim && !any_dead) {
                return "\"" + w + "\" but no unit is dead";
            }
            if (w == "ray" && !any_ray) {
                return "\"ray\" but no layer lies on a ray";
            }
            if ((w == "gate" || w == "gates" || w == "gated" || w == "half-space") && !obs.gating.present()) {
                return "\"" + w + "\" but no samples close the first layer";
            }
            if ((w == "keep" || w == "keeps") && next == "the" && i + 2 < lex.size() && lex[i + 2] == "sign" &&
                !any_sign_kept) {
                return "\"keeps the sign\" but no output preserves its target's sign";
            }
            const bool deeper_layer_claim = w == "second-layer" || w == "deeper" || w == "depth" ||
                                            (w == "second" && (next == "layer" || next == "layer's")) ||
                                            (w == "both" && next == "layers");
            if (deeper_layer_claim && !multi_layer) {
                return "\"" + w + (next.empty() ? "" : " " + next) + "\" but the network has one hidden layer";
            }
            if ((w == "two-dimensional" || w == "sheet" || w == "plane") && !planar) {
                return "\"" + w + "\" but the first layer has fewer than two effective dimensions";
            }
            if (w == "closely" && !any_close) {
                return "\"closely\" but no output correlates with its target at 0.9 or above";
            }
            if (w == "flat" && !any_flat) {
                return "\"flat\" but every prediction varies with at least 0.2 of its target's spread";
            }
            if (w == "the" && next == "prediction" && obs.shape.output_dim > 1) {
                return "\"the prediction\" but the network has " + std::to_string(obs.shape.output_dim) + " outputs";
            }
        }
        return "";
    }

    static std::string repetition(const std::vector<std::string>& lex) {
        std::map<std::string, std::size_t> counts;
        for (std::size_t i = 0; i + 4 <= lex.size(); ++i) {
            const std::string key = lex[i] + ' ' + lex[i + 1] + ' ' + lex[i + 2] + ' ' + lex[i + 3];
            if (++counts[key] >= 3) {
                return "\"" + key + "\" repeats three or more times";
            }
        }
        return "";
    }
};

}  // namespace forward_analysis
