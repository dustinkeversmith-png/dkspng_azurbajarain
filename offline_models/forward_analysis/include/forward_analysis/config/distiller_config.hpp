#pragma once

#include <cstddef>
#include <stdexcept>

namespace forward_analysis {

struct DistillerConfig {
    // Width of every channel token, latent and decoder input vector.
    std::size_t d_model = 256;
    // Number of learned Perceiver latent queries (M); the encoder output Z is num_latents x d_model.
    std::size_t num_latents = 8;
    // How many components of the dominant covariance eigenvector the eigen-sign head predicts.
    std::size_t eigen_sign_dim = 4;
    // Width of the decoder's tanh hidden layer between its input vector and the vocabulary logits.
    std::size_t decoder_hidden_dim = 256;
    // Longest token sequence (including BOS/EOS) the decoder trains on or generates for one observation field.
    std::size_t max_field_tokens = 160;
    // Adam step size for the decoder parameters.
    double decoder_learning_rate = 0.001;
    // Probability of zeroing each decoder token-embedding entry and tanh hidden activation during training (inverted dropout).
    double decoder_dropout = 0.15;
    // Decoupled weight decay on the decoder's vocabulary projection: W_out -= lr * decay * W_out after every Adam step.
    double decoder_output_weight_decay = 0.1;
    // Fine-tuning stops once the held-out token cross-entropy has not decreased for this many consecutive epochs.
    std::size_t early_stopping_patience = 5;
    // Weight of the observation-descriptor regression loss (Aux1) in the total loss.
    double aux1_observation_weight = 0.3;
    // Weight of the eigen-sign auxiliary loss (Aux2) in the total loss.
    double aux2_eigen_sign_weight = 0.2;
    // |activation| at or below this value counts as "not firing" on a sample.
    double dead_activation_threshold = 1e-6;
    // Singular values below this fraction of the largest one do not count toward effective rank.
    double spectral_rank_tolerance = 0.05;
    // A layer lies on a ray from the origin when its second uncentered singular value is below this fraction of the first.
    double ray_tolerance = 1e-3;
    // Multiple correlation with the two leading hidden axis scores at or above this value makes an input coordinate a driver.
    double driver_correlation_threshold = 0.75;
    // A prediction whose spread is at most this fraction of its target's spread counts as flat.
    double flat_output_ratio = 0.2;

    void validate() const {
        if (d_model == 0) {
            throw std::invalid_argument("d_model must be positive");
        }
        if (num_latents == 0) {
            throw std::invalid_argument("num_latents must be positive");
        }
        if (decoder_hidden_dim == 0) {
            throw std::invalid_argument("decoder_hidden_dim must be positive");
        }
        if (max_field_tokens < 2 || max_field_tokens > 512) {
            throw std::invalid_argument("max_field_tokens must be in [2, 512]");
        }
        if (!(decoder_learning_rate > 0.0)) {
            throw std::invalid_argument("decoder_learning_rate must be positive");
        }
        if (!(decoder_dropout >= 0.0 && decoder_dropout < 1.0)) {
            throw std::invalid_argument("decoder_dropout must be in [0, 1)");
        }
        if (!(decoder_output_weight_decay >= 0.0)) {
            throw std::invalid_argument("decoder_output_weight_decay must be non-negative");
        }
        if (early_stopping_patience == 0) {
            throw std::invalid_argument("early_stopping_patience must be positive");
        }
    }
};

}  // namespace forward_analysis
