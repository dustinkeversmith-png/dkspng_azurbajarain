#pragma once

#include "forward_analysis/math/adam_moment.hpp"

#include <Eigen/Core>

namespace forward_analysis {

// Addresses of every decoder parameter tensor and its Adam state, used to copy them to and from a device.
struct DecoderTensorRefs {
    // Previous-token embedding (vocab x d_model) and its Adam state.
    Eigen::MatrixXd* token_embeddings = nullptr;
    AdamMoment* token_moment = nullptr;
    // Two-back token embedding (vocab x d_model) and its Adam state.
    Eigen::MatrixXd* second_token_embeddings = nullptr;
    AdamMoment* token2_moment = nullptr;
    // Field embedding (4 x d_model) and its Adam state.
    Eigen::MatrixXd* field_embeddings = nullptr;
    AdamMoment* field_moment = nullptr;
    // Fixed sinusoidal step vectors (max_field_tokens x d_model); never trained.
    Eigen::MatrixXd* positions = nullptr;
    // Attention projections W_Q, W_K, W_V (d_model x d_model) and their Adam state.
    Eigen::MatrixXd* query_weights = nullptr;
    AdamMoment* query_moment = nullptr;
    Eigen::MatrixXd* key_weights = nullptr;
    AdamMoment* key_moment = nullptr;
    Eigen::MatrixXd* value_weights = nullptr;
    AdamMoment* value_moment = nullptr;
    // Descriptor projection W_aux (d_model x 14) and its Adam state.
    Eigen::MatrixXd* descriptor_weights = nullptr;
    AdamMoment* descriptor_moment = nullptr;
    // Hidden layer W1 (hidden x d_model), b1 (hidden) and their Adam state.
    Eigen::MatrixXd* hidden_weights = nullptr;
    AdamMoment* hidden_weights_moment = nullptr;
    Eigen::VectorXd* hidden_biases = nullptr;
    AdamMoment* hidden_biases_moment = nullptr;
    // Vocabulary projection W_out (vocab x hidden), b_out (vocab) and their Adam state.
    Eigen::MatrixXd* output_weights = nullptr;
    AdamMoment* output_weights_moment = nullptr;
    Eigen::VectorXd* output_biases = nullptr;
    AdamMoment* output_biases_moment = nullptr;
};

}  // namespace forward_analysis
