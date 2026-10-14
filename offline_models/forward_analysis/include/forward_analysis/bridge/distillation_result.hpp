#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/description/observation_text.hpp"

#include <Eigen/Core>

#include <array>
#include <cstddef>

namespace forward_analysis {

struct DistillationResult {
    // Number of channel tokens the encoder read for this telemetry.
    std::size_t channel_count = 0;
    // Encoder latent bottleneck Z (num_latents x d_model).
    Eigen::MatrixXd latent_embeddings;
    // Aux1 head's predicted observation descriptor.
    Eigen::VectorXd predicted_observation_descriptor;
    // Aux2 head's predicted dominant-eigenvector signs.
    Eigen::VectorXd predicted_eigen_signs;
    // Facts measured directly from the telemetry.
    NetworkObservations measured;
    // The four fields exactly as the decoder generated them, slot tokens unfilled, e.g. "<COUNT:DEAD_L1|W> of the ...".
    ObservationText decoder_slot_observations;
    // Per field: true when every slot token was fillable from this network's measurements.
    std::array<bool, ObservationText::kFieldCount> decoder_slots_filled{};
    // The four decoder fields after filling slots from the measured telemetry (the slot text where a slot was unfillable).
    ObservationText decoder_observations;
    // Final output: each decoder field that passed consistency, otherwise the measured-fact narration.
    ObservationText observations;
    // Per field: true when the decoder's text passed the consistency check against the measured facts.
    std::array<bool, ObservationText::kFieldCount> decoder_field_consistent{};
    // Per field: number of tokens the decoder emitted (BOS and EOS included).
    std::array<std::size_t, ObservationText::kFieldCount> decoder_token_counts{};
};

}  // namespace forward_analysis
