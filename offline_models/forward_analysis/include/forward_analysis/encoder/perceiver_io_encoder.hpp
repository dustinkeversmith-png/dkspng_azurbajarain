#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/encoder/perceiver_forward_cache.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

class PerceiverIOEncoder {
public:
    explicit PerceiverIOEncoder(const DistillerConfig& config = DistillerConfig{},
                                std::uint64_t seed = 2026ULL)
        : config_(config) {
        config_.validate();
        initialize_parameters(seed);
    }

    void initialize_parameters(std::uint64_t seed) {
        DeterministicRng rng(seed);
        const std::size_t m = config_.num_latents;
        const std::size_t d = config_.d_model;

        latent_queries_ = rng.xavier_matrix(m, d);
        w_q_ = rng.xavier_matrix(d, d);
        w_k_ = rng.xavier_matrix(d, d);
        w_v_ = rng.xavier_matrix(d, d);
        w_o_ = rng.xavier_matrix(d, d);
        b_o_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d));
    }

    PerceiverForwardCache forward_with_cache(const Eigen::MatrixXd& input_sequence) const {
        if (input_sequence.rows() == 0) {
            throw std::invalid_argument("PerceiverIOEncoder input_sequence cannot be empty");
        }
        if (input_sequence.cols() != static_cast<Eigen::Index>(config_.d_model)) {
            throw std::invalid_argument("PerceiverIOEncoder input dimension mismatch with d_model");
        }

        const double scale = 1.0 / std::sqrt(static_cast<double>(config_.d_model));

        PerceiverForwardCache cache;
        cache.input_sequence = input_sequence;
        cache.queries.noalias() = latent_queries_ * w_q_.transpose();
        cache.keys.noalias() = input_sequence * w_k_.transpose();
        cache.values.noalias() = input_sequence * w_v_.transpose();
        cache.attn_weights = MatrixOps::softmax_rows((cache.queries * cache.keys.transpose()) * scale);
        cache.context.noalias() = cache.attn_weights * cache.values;
        cache.pre_norm_latents = latent_queries_ + MatrixOps::affine_rows(cache.context, w_o_, b_o_);
        cache.latents_z = MatrixOps::layer_norm_rows(cache.pre_norm_latents);
        return cache;
    }

    Eigen::MatrixXd encode(const Eigen::MatrixXd& input_sequence) const {
        return forward_with_cache(input_sequence).latents_z;
    }

    Eigen::MatrixXd backward_and_update(const PerceiverForwardCache& cache,
                                        const Eigen::MatrixXd& d_latents_z,
                                        double learning_rate) {
        const Eigen::Index n = cache.input_sequence.rows();
        const Eigen::Index m = static_cast<Eigen::Index>(config_.num_latents);
        const Eigen::Index d = static_cast<Eigen::Index>(config_.d_model);

        if (learning_rate <= 0.0 || d_latents_z.rows() != m || d_latents_z.cols() != d) {
            return Eigen::MatrixXd::Zero(n, d);
        }

        const double lr = MatrixOps::clamp(learning_rate, 0.0, 0.25);
        const Eigen::MatrixXd d_pre = d_latents_z.cwiseMax(-4.0).cwiseMin(4.0);
        const Eigen::MatrixXd d_context = d_pre * w_o_;

        b_o_ -= lr * d_pre.colwise().sum().transpose();
        w_o_.noalias() -= lr * (d_pre.transpose() * cache.context);

        const Eigen::MatrixXd d_values =
            (cache.attn_weights.transpose() * d_context).cwiseMax(-4.0).cwiseMin(4.0);
        // Attention weight gradient (M x N) from context = attn_weights * values.
        const Eigen::MatrixXd d_attn = d_context * cache.values.transpose();
        // Row-wise dot product sum_c(d_attn * attn_weights) for softmax backward.
        const Eigen::VectorXd row_dot = d_attn.cwiseProduct(cache.attn_weights).rowwise().sum();
        // Scaled pre-softmax score gradient (M x N).
        const double scale = 1.0 / std::sqrt(static_cast<double>(config_.d_model));
        // Softmax Jacobian product scaled by 1 / sqrt(d_model).
        const Eigen::MatrixXd d_scores =
            cache.attn_weights.cwiseProduct(d_attn - row_dot.replicate(1, d_attn.cols())) * scale;
        // Query gradient (M x D).
        const Eigen::MatrixXd d_queries = (d_scores * cache.keys).cwiseMax(-4.0).cwiseMin(4.0);
        // Key gradient (N x D).
        const Eigen::MatrixXd d_keys = (d_scores.transpose() * cache.queries).cwiseMax(-4.0).cwiseMin(4.0);

        latent_queries_ -= lr * (d_pre + d_queries * w_q_);
        w_q_.noalias() -= lr * (d_queries.transpose() * latent_queries_);

        Eigen::MatrixXd d_input = d_values * w_v_ + d_keys * w_k_;

        const double n_scale = lr / static_cast<double>(std::max<Eigen::Index>(1, n));
        w_v_.noalias() -= n_scale * (d_values.transpose() * cache.input_sequence);
        w_k_.noalias() -= n_scale * (d_keys.transpose() * cache.input_sequence);
        return d_input;
    }

    const Eigen::MatrixXd& latent_queries() const noexcept {
        return latent_queries_;
    }

    std::size_t num_latents() const noexcept { return config_.num_latents; }
    std::size_t d_model() const noexcept { return config_.d_model; }

private:
    // Model dimensions (num_latents, d_model).
    DistillerConfig config_;
    // Learned latent query vectors, one row per latent (M x D); also the residual base of Z.
    Eigen::MatrixXd latent_queries_;
    // Query projection applied to latent queries (D x D).
    Eigen::MatrixXd w_q_;
    // Key projection applied to the probe sequence (D x D).
    Eigen::MatrixXd w_k_;
    // Value projection applied to the probe sequence (D x D).
    Eigen::MatrixXd w_v_;
    // Output projection applied to the attention context (D x D).
    Eigen::MatrixXd w_o_;
    // Output projection bias (D).
    Eigen::VectorXd b_o_;
};

}  // namespace forward_analysis
