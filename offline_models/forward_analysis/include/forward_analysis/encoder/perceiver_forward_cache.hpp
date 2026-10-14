#pragma once

#include <Eigen/Core>

namespace forward_analysis {

// Intermediate tensors from one Perceiver forward pass, kept so the backward pass can reuse them.
// N = number of probes, M = number of latents, D = d_model.
struct PerceiverForwardCache {
    // Adapted probe sequence fed to the encoder (N x D).
    Eigen::MatrixXd input_sequence;
    // Latent queries projected by W_q (M x D).
    Eigen::MatrixXd queries;
    // Probe sequence projected by W_k (N x D).
    Eigen::MatrixXd keys;
    // Probe sequence projected by W_v (N x D).
    Eigen::MatrixXd values;
    // Softmax cross-attention weights; row m is how latent m distributes attention over probes (M x N).
    Eigen::MatrixXd attn_weights;
    // Attention-weighted sum of values per latent (M x D).
    Eigen::MatrixXd context;
    // Latent queries + output projection of context, before layer norm (M x D).
    Eigen::MatrixXd pre_norm_latents;
    // Final layer-normalized latent bottleneck Z (M x D).
    Eigen::MatrixXd latents_z;
};

}  // namespace forward_analysis
