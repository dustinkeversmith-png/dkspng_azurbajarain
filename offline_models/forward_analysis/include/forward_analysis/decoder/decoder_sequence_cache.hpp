#pragma once

#include <Eigen/Core>

#include <vector>

namespace forward_analysis {

// Intermediate values of one teacher-forced decoder pass over a field sequence, kept for backprop.
// T = steps, M = latents, d = d_model, H = hidden width, V = vocabulary size.
struct DecoderSequenceCache {
    // Token fed in at each step (T); BOS at step 0.
    std::vector<int> previous_tokens;
    // Token two positions back at each step (T); PAD where none exists.
    std::vector<int> second_previous_tokens;
    // Inverted dropout mask applied to the previous-token embedding rows (T x d); all ones in eval mode.
    Eigen::MatrixXd embedding_mask;
    // Inverted dropout mask applied to the second-previous-token embedding rows (T x d).
    Eigen::MatrixXd second_embedding_mask;
    // Inverted dropout mask applied to the tanh hidden layer (T x H).
    Eigen::MatrixXd hidden_mask;
    // Layer-normalized token/field/position sum used as attention queries' input (T x d).
    Eigen::MatrixXd query_inputs;
    // Per-row 1/std of that layer norm (T).
    Eigen::VectorXd query_inverse_std;
    // Attention queries Q = LN(u) W_Q (T x d).
    Eigen::MatrixXd queries;
    // Attention keys K = Z W_K (M x d).
    Eigen::MatrixXd keys;
    // Attention values V = Z W_V (M x d).
    Eigen::MatrixXd values;
    // Attention weights softmax(Q K^T / sqrt(d)) over the latents (T x M); every row sums to 1.
    Eigen::MatrixXd attention;
    // Layer-normalized step inputs LN(u + c + W_aux d_hat) (T x d).
    Eigen::MatrixXd normalized_inputs;
    // Per-row 1/std used by that layer norm (T).
    Eigen::VectorXd inverse_std;
    // tanh hidden activations before dropout (T x H).
    Eigen::MatrixXd hidden;
    // Softmax next-token distribution at each step (T x V).
    Eigen::MatrixXd probabilities;
};

}  // namespace forward_analysis
