#pragma once

#include "forward_analysis/analysis/network_observations.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Verifies the batch-statistic statements of the observation text against measured facts, wherever they occur:
//   "across N observations"                                   N = batch size
//   "hidden widths are a and b"                               hidden widths in order
//   "layer L has Z coordinates that remain zero"              Z = always-zero units of layer L
//   "K observations with no positive coordinates"             K = samples with no positive unit in the layer named last
//   "input coordinates a and b have the strongest"            the two strongest-associated inputs, strongest first
//   "coordinates a and b have the weakest"                    the two weakest-associated inputs, weakest first
//   "numerical ranks r1, r2"                                  numerical rank of every layer in order
//   "narrow observed representation" / "multiple directions"  every layer rank <= narrow_rank_limit / some layer above
//   "over K output coordinates"                               K = output width
//   "a have a smaller prediction range", "b have a larger range", "c have broadly comparable ranges"
//   "agree in A of E entries"                                 A = sign agreements, E = prediction entries
// Every number lexeme it verifies is marked in `verified` so other checks do not reinterpret it.
struct BatchStatementChecker {
    // lex: lowercase lexemes of one field. Returns the first contradicted statement, or "" when all hold.
    static std::string verify(const std::vector<std::string>& lex, const NetworkObservations& obs, std::vector<bool>& verified) {
        verified.assign(lex.size(), false);
        std::size_t layer_context = 0;
        for (std::size_t i = 0; i < lex.size(); ++i) {
            std::vector<std::size_t> v;
            std::vector<std::size_t> at;
            if (match(lex, i, {"across", "#", "observations"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.shape.num_samples) {
                    return "\"across " + lex[at[0]] + " observations\" but the batch has " + std::to_string(obs.shape.num_samples);
                }
            } else if (match(lex, i, {"hidden", "widths", "are", "#*"}, v, at)) {
                mark(verified, at);
                if (v != obs.shape.hidden_widths) {
                    return "stated hidden widths differ from the telemetry";
                }
            } else if (match(lex, i, {"layer", "#", "has", "#", "coordinates", "that", "remain", "zero"}, v, at)) {
                mark(verified, at);
                if (v[0] == 0 || v[0] > obs.layers.size()) {
                    return "\"layer " + lex[at[0]] + "\" but the network has " + std::to_string(obs.layers.size()) + " hidden layers";
                }
                layer_context = v[0];
                if (v[1] != obs.layers[v[0] - 1].dead_units.size()) {
                    return "layer " + lex[at[0]] + " has " + std::to_string(obs.layers[v[0] - 1].dead_units.size()) +
                           " always-zero coordinates, not " + lex[at[1]];
                }
            } else if (match(lex, i, {"#", "observations", "with", "no", "positive", "coordinates"}, v, at)) {
                mark(verified, at);
                if (layer_context == 0) {
                    return "\"" + lex[at[0]] + " observations with no positive coordinates\" names no layer";
                }
                if (v[0] != obs.layers[layer_context - 1].nonpositive_samples) {
                    return "layer " + std::to_string(layer_context) + " has " +
                           std::to_string(obs.layers[layer_context - 1].nonpositive_samples) +
                           " observations with no positive coordinate, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"input", "coordinates", "#", "and", "#", "have", "the", "strongest"}, v, at)) {
                mark(verified, at);
                if (obs.strongest_association <= 0.0) {
                    return "\"strongest\" associations but no input correlates with the final hidden layer";
                }
                if (v != obs.strongest_inputs) {
                    return "strongest-associated inputs are " + pair_text(obs.strongest_inputs) + ", not " + pair_text(v);
                }
            } else if (match(lex, i, {"coordinates", "#", "and", "#", "have", "the", "weakest"}, v, at)) {
                mark(verified, at);
                if (obs.strongest_association <= 0.0) {
                    return "\"weakest\" associations but no input correlates with the final hidden layer";
                }
                if (v != obs.weakest_inputs) {
                    return "weakest-associated inputs are " + pair_text(obs.weakest_inputs) + ", not " + pair_text(v);
                }
            } else if (match(lex, i, {"numerical", "ranks", "#*"}, v, at)) {
                mark(verified, at);
                std::vector<std::size_t> ranks;
                for (const LayerObservation& layer : obs.layers) {
                    ranks.push_back(layer.numerical_rank);
                }
                if (v != ranks) {
                    return "stated numerical ranks differ from the measured ranks";
                }
            } else if (match(lex, i, {"narrow", "observed", "representation"}, v, at) && !obs.narrow_representation) {
                return "\"narrow observed representation\" but a layer's numerical rank exceeds the narrow limit";
            } else if (match(lex, i, {"multiple", "directions"}, v, at) && obs.narrow_representation) {
                return "\"multiple directions\" but every layer's numerical rank is within the narrow limit";
            } else if (match(lex, i, {"over", "#", "output", "coordinates"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.shape.output_dim) {
                    return "\"over " + lex[at[0]] + " output coordinates\" but the network has " + std::to_string(obs.shape.output_dim);
                }
            } else if (match(lex, i, {"#", "have", "a", "smaller", "prediction", "range"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.outputs_with_smaller_range) {
                    return std::to_string(obs.outputs_with_smaller_range) + " outputs have a smaller range, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"#", "have", "a", "larger", "range"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.outputs_with_larger_range) {
                    return std::to_string(obs.outputs_with_larger_range) + " outputs have a larger range, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"#", "have", "broadly", "comparable", "ranges"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.outputs_with_comparable_range) {
                    return std::to_string(obs.outputs_with_comparable_range) + " outputs have a comparable range, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"agree", "in", "#", "of", "#", "entries"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.sign_agreements || v[1] != obs.sign_entries) {
                    return "signs agree in " + std::to_string(obs.sign_agreements) + " of " + std::to_string(obs.sign_entries) +
                           " entries, not " + lex[at[0]] + " of " + lex[at[1]];
                }
            }
        }
        return "";
    }

    // True for a lexeme made only of decimal digits (at most nine).
    static bool number(const std::string& token) {
        if (token.empty()) {
            return false;
        }
        for (char c : token) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return token.size() < 10;
    }

    // Matches pattern at lex[start]: "#" is one number, "#*" a list of numbers joined by "," or "and"; values and
    // positions of every matched number are returned in order.
    static bool match(const std::vector<std::string>& lex,
                      std::size_t start,
                      const std::vector<std::string>& pattern,
                      std::vector<std::size_t>& values,
                      std::vector<std::size_t>& positions) {
        values.clear();
        positions.clear();
        std::size_t i = start;
        for (const std::string& p : pattern) {
            if (p == "#" || p == "#*") {
                if (i >= lex.size() || !number(lex[i])) {
                    return false;
                }
                values.push_back(static_cast<std::size_t>(std::stoul(lex[i])));
                positions.push_back(i++);
                while (p == "#*" && i + 1 < lex.size() && (lex[i] == "," || lex[i] == "and") && number(lex[i + 1])) {
                    values.push_back(static_cast<std::size_t>(std::stoul(lex[i + 1])));
                    positions.push_back(i + 1);
                    i += 2;
                }
            } else if (i < lex.size() && lex[i] == p) {
                ++i;
            } else {
                return false;
            }
        }
        return true;
    }

private:
    static void mark(std::vector<bool>& verified, const std::vector<std::size_t>& positions) {
        for (std::size_t p : positions) {
            verified[p] = true;
        }
    }

    static std::string pair_text(const std::vector<std::size_t>& v) {
        std::string out;
        for (std::size_t k = 0; k < v.size(); ++k) {
            out += (k == 0 ? "" : " and ") + std::to_string(v[k]);
        }
        return out;
    }
};

}  // namespace forward_analysis
