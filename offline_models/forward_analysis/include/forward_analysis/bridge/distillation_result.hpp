#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/heads/prediction_result.hpp"

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <map>
#include <string>

namespace forward_analysis {

struct DistillationResult {
    // Number of channel tokens extracted for this telemetry.
    std::size_t channel_count = 0;
    // Total number of multi-family tokens (Channel, Sample, Relationship, Layer, Global) read by the Perceiver encoder.
    std::size_t encoder_token_count = 0;
    // Encoder latent bottleneck Z (num_latents x d_model).
    Eigen::MatrixXd latent_embeddings;
    // Aux1 head's predicted observation descriptor (14-dim).
    Eigen::VectorXd predicted_observation_descriptor;
    // Domain-head conditioning vector (14-dim) passed to the decoder.
    Eigen::VectorXd decoder_conditioning_vector;
    // Aux2 head's predicted eigenvalue signs (4-dim).
    Eigen::VectorXd predicted_eigen_signs;
    // Typed predictions from every registered NumericalPredictionHead across the six mathematical domains.
    std::map<std::string, PredictionResult> domain_predictions;
    // Facts measured directly from the telemetry.
    NetworkObservations measured;
    // The four fields exactly as the decoder generated them, slot tokens unfilled, e.g. "<COUNT:DEAD_L1|W> of the ...".
    ObservationText decoder_slot_observations;
    // Per field: true when every slot token was fillable from this network's measurements.
    std::array<bool, ObservationText::kFieldCount> decoder_slots_filled{};
    // The four decoder fields after filling slots from the measured telemetry (the slot text where a slot was unfillable).
    ObservationText decoder_observations;
    // The measured-facts narration (config.narration_style) written without the decoder.
    ObservationText narrated_observations;
    // Final output: each decoder field that passed consistency, otherwise the measured-fact narration.
    ObservationText observations;
    // Per field: true when the decoder's text passed the consistency check against the measured facts.
    std::array<bool, ObservationText::kFieldCount> decoder_field_consistent{};
    // Per field: number of tokens the decoder emitted (BOS and EOS included).
    std::array<std::size_t, ObservationText::kFieldCount> decoder_token_counts{};
};

}  // namespace forward_analysis
