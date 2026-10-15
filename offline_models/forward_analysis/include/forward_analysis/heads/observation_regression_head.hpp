#pragma once

#include "forward_analysis/math/adam_moment.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

// Aux1: two-layer residual MLP + Adam regression head from pooled latents to the ObservationDescriptor.
class ObservationRegressionHead {
public:
    // Width of the internal tanh hidden layer.
    static constexpr std::size_t kHiddenDim = 64;

    explicit ObservationRegressionHead(std::size_t d_model = 256,
                                       std::size_t descriptor_dim = 14,
                                       std::uint64_t seed = 3001ULL)
        : d_model_(d_model),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(descriptor_dim))),
          hidden_biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(kHiddenDim))) {
        // Parameter initialization RNG.
        DeterministicRng rng(seed);
        weights_ = rng.xavier_matrix(descriptor_dim, d_model_) * 0.5;
        hidden_weights_ = rng.xavier_matrix(kHiddenDim, d_model_);
        out_weights_ = rng.xavier_matrix(descriptor_dim, kHiddenDim) * 0.5;
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& pooled_z) const {
        if (pooled_z.size() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("ObservationRegressionHead input width must equal d_model");
        }
        // Hidden MLP activations (kHiddenDim).
        const Eigen::VectorXd h = (hidden_weights_ * pooled_z + hidden_biases_).array().tanh().matrix();
        return weights_ * pooled_z + out_weights_ * h + biases_;
    }

    double compute_loss(const Eigen::VectorXd& pooled_z, const Eigen::VectorXd& target) const {
        if (target.size() != biases_.size()) {
            return 0.0;
        }
        return 0.5 * (predict(pooled_z) - target).squaredNorm() / static_cast<double>(target.size());
    }

    double compute_loss_and_step(const Eigen::VectorXd& pooled_z,
                                 const Eigen::VectorXd& target,
                                 double learning_rate,
                                 double loss_weight,
                                 Eigen::VectorXd& d_pooled_accum) {
        if (target.size() != biases_.size()) {
            return 0.0;
        }
        if (d_pooled_accum.size() != static_cast<Eigen::Index>(d_model_)) {
            d_pooled_accum = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d_model_));
        }
        // Hidden MLP activations (kHiddenDim).
        const Eigen::VectorXd h = (hidden_weights_ * pooled_z + hidden_biases_).array().tanh().matrix();
        // Prediction error vector (descriptor_dim).
        const Eigen::VectorXd error = (weights_ * pooled_z + out_weights_ * h + biases_) - target;
        // Reciprocal of descriptor dimension.
        const double inv_k = 1.0 / static_cast<double>(target.size());
        // Scaled output gradient.
        const Eigen::VectorXd d_param = error * inv_k;
        // Hidden gradient.
        const Eigen::VectorXd d_h = out_weights_.transpose() * d_param;
        // Pre-tanh hidden gradient.
        const Eigen::VectorXd d_pre_h = d_h.array() * (1.0 - h.array().square());

        d_pooled_accum.noalias() +=
            (weights_.transpose() * d_param + hidden_weights_.transpose() * d_pre_h) * loss_weight;
        if (learning_rate > 0.0) {
            ++adam_step_;
            // Adam step size.
            const double adam_lr = std::min(0.003, learning_rate * 0.06);
            // Weight gradients.
            const Eigen::MatrixXd g_skip = d_param * pooled_z.transpose();
            // Output layer weight gradient.
            const Eigen::MatrixXd g_out = d_param * h.transpose();
            // Hidden layer weight gradient.
            const Eigen::MatrixXd g_hid = d_pre_h * pooled_z.transpose();

            m_weights_.apply(weights_, g_skip, adam_lr, adam_step_);
            m_out_.apply(out_weights_, g_out, adam_lr, adam_step_);
            m_biases_.apply(biases_, d_param, adam_lr, adam_step_);
            m_hid_w_.apply(hidden_weights_, g_hid, adam_lr, adam_step_);
            m_hid_b_.apply(hidden_biases_, d_pre_h, adam_lr, adam_step_);
        }
        return 0.5 * error.squaredNorm() * inv_k;
    }

    std::size_t descriptor_dim() const noexcept { return static_cast<std::size_t>(biases_.size()); }

private:
    // Width of the pooled latent vector this head reads.
    std::size_t d_model_ = 256;
    // Adam step counter.
    std::uint64_t adam_step_ = 0;
    // Linear skip map from pooled latent to descriptor (descriptor_dim x d_model).
    Eigen::MatrixXd weights_;
    // Bias per descriptor component (descriptor_dim).
    Eigen::VectorXd biases_;
    // Hidden MLP weights (kHiddenDim x d_model).
    Eigen::MatrixXd hidden_weights_;
    // Hidden MLP biases (kHiddenDim).
    Eigen::VectorXd hidden_biases_;
    // Output projection from hidden MLP to descriptor (descriptor_dim x kHiddenDim).
    Eigen::MatrixXd out_weights_;
    // Adam moments for all parameter tensors.
    AdamMoment m_weights_;
    AdamMoment m_biases_;
    AdamMoment m_hid_w_;
    AdamMoment m_hid_b_;
    AdamMoment m_out_;
};

}  // namespace forward_analysis
