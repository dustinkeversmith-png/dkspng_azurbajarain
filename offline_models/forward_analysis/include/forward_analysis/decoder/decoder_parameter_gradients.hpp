#pragma once

#include <Eigen/Core>

namespace forward_analysis {

// Gradients of one field's mean token cross-entropy with respect to every decoder parameter and the latents.
struct DecoderParameterGradients {
    // Previous-token embeddings (V x d).
    Eigen::MatrixXd token_embeddings;
    // Second-previous-token embeddings (V x d).
    Eigen::MatrixXd second_token_embeddings;
    // Field embeddings (4 x d).
    Eigen::MatrixXd field_embeddings;
    // Attention query projection (d x d).
    Eigen::MatrixXd query_weights;
    // Attention key projection (d x d).
    Eigen::MatrixXd key_weights;
    // Attention value projection (d x d).
    Eigen::MatrixXd value_weights;
    // Predicted-descriptor projection (d x descriptor_dim).
    Eigen::MatrixXd descriptor_weights;
    // Hidden layer weights (H x d).
    Eigen::MatrixXd hidden_weights;
    // Hidden layer biases (H).
    Eigen::VectorXd hidden_biases;
    // Vocabulary projection weights (V x H).
    Eigen::MatrixXd output_weights;
    // Vocabulary projection biases (V).
    Eigen::VectorXd output_biases;
    // Latent bottleneck Z (M x d), one distinct row per latent through the keys and values.
    Eigen::MatrixXd latents;
};

}  // namespace forward_analysis
