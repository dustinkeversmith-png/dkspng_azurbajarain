#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/description/observation_text.hpp"

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

// Writes the four observation fields as batch statistics measured from the telemetry, in the phrasing of the
// corrected dataset (manifest version corrected-2026-10-14): always-zero coordinates and observations with no
// positive coordinate per layer, strongest / weakest input-to-final-hidden sample correlations, numerical ranks,
// and prediction-range and sign agreement against the targets.
struct BatchStatisticsNarrator {
    static ObservationText describe(const NetworkObservations& obs, const DistillerConfig& config = DistillerConfig{}) {
        ObservationText text;
        text.pathway_activity = pathway_activity(obs);
        text.feature_sensitivity = feature_sensitivity(obs);
        text.representational_structure = representational_structure(obs);
        text.functional_mapping = functional_mapping(obs, config);
        return text;
    }

private:
    // "27 and 14", "36", "8, 12 and 20".
    static std::string list(const std::vector<std::size_t>& values, const std::string& separator) {
        std::string out;
        for (std::size_t k = 0; k < values.size(); ++k) {
            if (k > 0) {
                out += (separator == "and" && k + 1 == values.size()) ? " and " : ", ";
            }
            out += std::to_string(values[k]);
        }
        return out;
    }

    static std::string pathway_activity(const NetworkObservations& obs) {
        std::string out = "Across " + std::to_string(obs.shape.num_samples) + " observations, the hidden widths are " +
                          list(obs.shape.hidden_widths, "and") + ".";
        for (std::size_t l = 0; l < obs.layers.size(); ++l) {
            out += " Layer " + std::to_string(l + 1) + " has " + std::to_string(obs.layers[l].dead_units.size()) +
                   " coordinates that remain zero throughout the batch and " +
                   std::to_string(obs.layers[l].nonpositive_samples) + " observations with no positive coordinates.";
        }
        return out;
    }

    static std::string feature_sensitivity(const NetworkObservations& obs) {
        if (obs.strongest_association <= 0.0 || obs.strongest_inputs.size() < 2 || obs.weakest_inputs.size() < 2) {
            return "Observed association only: no input coordinate has a nonzero sample correlation with any final hidden "
                   "coordinate. These associations do not establish causal feature influence.";
        }
        return "Observed association only: input coordinates " + list(obs.strongest_inputs, "and") +
               " have the strongest absolute sample correlations with at least one final hidden coordinate; coordinates " +
               list(obs.weakest_inputs, "and") +
               " have the weakest. These associations do not establish causal feature influence.";
    }

    static std::string representational_structure(const NetworkObservations& obs) {
        std::vector<std::size_t> ranks;
        for (const LayerObservation& layer : obs.layers) {
            ranks.push_back(layer.numerical_rank);
        }
        return "The centered activation matrices have observed numerical ranks " + list(ranks, ",") +
               " across successive hidden layers. " +
               (obs.narrow_representation
                    ? "This is a narrow observed representation relative to its hidden widths."
                    : "The observed activation variation occupies multiple directions in at least one layer.");
    }

    static std::string functional_mapping(const NetworkObservations& obs, const DistillerConfig& config) {
        std::ostringstream window;
        window << config.range_ratio_low << " to " << config.range_ratio_high;
        return "Comparing predictions against targets over " + std::to_string(obs.shape.output_dim) + " output coordinates, " +
               std::to_string(obs.outputs_with_smaller_range) + " have a smaller prediction range, " +
               std::to_string(obs.outputs_with_larger_range) + " have a larger range, and " +
               std::to_string(obs.outputs_with_comparable_range) +
               " have broadly comparable ranges using a range-ratio window from " + window.str() +
               ". Prediction and target signs agree in " + std::to_string(obs.sign_agreements) + " of " +
               std::to_string(obs.sign_entries) + " entries. These statements describe only the observed batch.";
    }
};

}  // namespace forward_analysis
