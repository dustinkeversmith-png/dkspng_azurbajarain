#pragma once

#include <Eigen/Core>

#include <cstddef>

namespace forward_analysis {

// Heterogeneous numerical token matrices extracted from a telemetry batch before projection into d_model.
struct MultiFamilyTokenBatch {
    // Channel tokens (C x F_channel, where F_channel is 25 for kBaseline25 or 38 for expanded modes).
    Eigen::MatrixXd channel_tokens;
    // Bounded observation-level sample tokens (S_tokens x 16), empty when not active.
    Eigen::MatrixXd sample_tokens;
    // Pairwise correlation, covariance, cross-layer, and output-alignment relationship tokens (R_tokens x 16).
    Eigen::MatrixXd relationship_tokens;
    // Per-hidden-layer structural and spectral tokens (L_tokens x 16).
    Eigen::MatrixXd layer_tokens;
    // Global mathematical domain summary tokens from FeatureExtractorRegistry (G_tokens x 16).
    Eigen::MatrixXd global_tokens;
    // Validity mask per row of layer_tokens (1.0 when the layer exists, 0.0 for padded layer slots).
    Eigen::VectorXd layer_valid_mask;

    // Total number of tokens across all active families in this batch.
    std::size_t total_tokens() const noexcept {
        return static_cast<std::size_t>(channel_tokens.rows() + sample_tokens.rows() +
                                        relationship_tokens.rows() + layer_tokens.rows() +
                                        global_tokens.rows());
    }
};

}  // namespace forward_analysis
