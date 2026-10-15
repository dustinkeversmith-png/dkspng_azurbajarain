#pragma once

#include "forward_analysis/math/adam_moment.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

// Spectral eigenvalue-sign head: two-layer residual MLP with Adam updates and class-balanced focal BCE weighting
// designed to resolve majority-class (++--) collapse on components 2 and 3.
class EigenSignHead {
public:
    // Width of the internal spectral MLP hidden layer.
    static constexpr std::size_t kHiddenDim = 64;

    explicit EigenSignHead(std::size_t d_model = 256,
                           std::size_t eigen_dim = 4,
                           std::uint64_t seed = 3002ULL)
        : d_model_(d_model),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(eigen_dim))),
          hidden_biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(kHiddenDim))),
          pos_weights_(Eigen::VectorXd::Ones(static_cast<Eigen::Index>(eigen_dim))),
          neg_weights_(Eigen::VectorXd::Ones(static_cast<Eigen::Index>(eigen_dim))) {
        // Parameter initialization RNG.
        DeterministicRng rng(seed);
        weights_ = rng.xavier_matrix(eigen_dim, d_model_) * 0.5;
        hidden_weights_ = rng.xavier_matrix(kHiddenDim, d_model_);
        out_weights_ = rng.xavier_matrix(eigen_dim, kHiddenDim);
    }

    Eigen::VectorXd predict_sign_probabilities(const Eigen::VectorXd& pooled_z) const {
        if (pooled_z.size() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("EigenSignHead input width must equal d_model");
        }
        // Hidden spectral representation (kHiddenDim).
        const Eigen::VectorXd h = (hidden_weights_ * pooled_z + hidden_biases_).array().tanh().matrix();
        // Combined residual + MLP logits (eigen_dim).
        const Eigen::VectorXd logits = weights_ * pooled_z + out_weights_ * h + biases_;
        return MatrixOps::sigmoid(logits);
    }

    Eigen::VectorXd predict_signs(const Eigen::VectorXd& pooled_z) const {
        return predict_sign_probabilities(pooled_z).unaryExpr(
            [](double p) { return p >= 0.5 ? 1.0 : -1.0; });
    }

    double compute_loss(const Eigen::VectorXd& pooled_z,
                        const Eigen::VectorXd& target_signs) const {
        // Predicted positive-sign probabilities.
        const Eigen::VectorXd probs = predict_sign_probabilities(pooled_z);
        // Active sign count.
        const Eigen::Index k = std::min(probs.size(), target_signs.size());
        if (k == 0) {
            return 0.0;
        }
        return binary_cross_entropy(clamp_probs(probs.head(k)), sign_labels(target_signs.head(k)));
    }

    double compute_loss_and_step(const Eigen::VectorXd& pooled_z,
                                 const Eigen::VectorXd& target_signs,
                                 double learning_rate,
                                 double loss_weight,
                                 Eigen::VectorXd& d_pooled_accum) {
        // Predicted positive-sign probabilities.
        const Eigen::VectorXd probs = predict_sign_probabilities(pooled_z);
        // Active sign count.
        const Eigen::Index k = std::min(probs.size(), target_signs.size());
        if (k == 0) {
            return 0.0;
        }
        if (d_pooled_accum.size() != static_cast<Eigen::Index>(d_model_)) {
            d_pooled_accum = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d_model_));
        }

        // Hidden layer activations (kHiddenDim).
        const Eigen::VectorXd h = (hidden_weights_ * pooled_z + hidden_biases_).array().tanh().matrix();
        // Clamped probabilities in (1e-12, 1 - 1e-12).
        const Eigen::ArrayXd p = clamp_probs(probs.head(k));
        // Binary labels in {0, 1}.
        const Eigen::ArrayXd y = sign_labels(target_signs.head(k));
        // Reciprocal of component count.
        const double inv_k = 1.0 / static_cast<double>(k);

        // Class-balanced focal logit gradient per component.
        Eigen::VectorXd d_raw(k);
        for (Eigen::Index i = 0; i < k; ++i) {
            // Positive class weight for component i.
            const double wp = pos_weights_(i);
            // Negative class weight for component i.
            const double wn = neg_weights_(i);
            // Focal modulation factor emphasizing misclassified spectral signs.
            const double pt = y(i) > 0.5 ? p(i) : (1.0 - p(i));
            // Focal factor (0.5 + (1 - pt)).
            const double focal = 0.5 + (1.0 - pt);
            d_raw(i) = focal * (wp * y(i) * (p(i) - 1.0) + wn * (1.0 - y(i)) * p(i)) * inv_k;
        }

        // Hidden layer gradient (kHiddenDim).
        const Eigen::VectorXd d_h = out_weights_.topRows(k).transpose() * d_raw;
        // Pre-tanh hidden gradient (kHiddenDim).
        const Eigen::VectorXd d_pre_h = d_h.array() * (1.0 - h.array().square());
        // Gradient with respect to `pooled_z` scaled by `loss_weight`.
        d_pooled_accum.noalias() +=
            (weights_.topRows(k).transpose() * d_raw + hidden_weights_.transpose() * d_pre_h) * loss_weight;

        if (learning_rate > 0.0 && loss_weight > 0.0) {
            ++adam_step_;
            // Adam step size for the spectral head.
            const double adam_lr = std::min(0.003, learning_rate * 0.06);
            // Padded full-size logit gradient.
            Eigen::VectorXd d_raw_full = Eigen::VectorXd::Zero(biases_.size());
            d_raw_full.head(k) = d_raw;

            // Gradients for skip weights, output weights, and hidden weights.
            const Eigen::MatrixXd g_skip = d_raw_full * pooled_z.transpose();
            // Output layer weight gradient.
            const Eigen::MatrixXd g_out = d_raw_full * h.transpose();
            // Hidden layer weight gradient.
            const Eigen::MatrixXd g_hid = d_pre_h * pooled_z.transpose();

            m_weights_.apply(weights_, g_skip, adam_lr, adam_step_);
            m_out_.apply(out_weights_, g_out, adam_lr, adam_step_);
            m_biases_.apply(biases_, d_raw_full, adam_lr, adam_step_);
            m_hid_w_.apply(hidden_weights_, g_hid, adam_lr, adam_step_);
            m_hid_b_.apply(hidden_biases_, d_pre_h, adam_lr, adam_step_);
        }
        return binary_cross_entropy(p, y);
    }

private:
    static Eigen::ArrayXd clamp_probs(const Eigen::VectorXd& probs) {
        return probs.array().cwiseMax(1e-12).cwiseMin(1.0 - 1e-12);
    }

    static Eigen::ArrayXd sign_labels(const Eigen::VectorXd& signs) {
        return (signs.array() >= 0.0).cast<double>();
    }

    static double binary_cross_entropy(const Eigen::ArrayXd& p, const Eigen::ArrayXd& y) {
        return -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean();
    }

    // Width of the pooled latent vector this head reads.
    std::size_t d_model_ = 256;
    // Adam step counter.
    std::uint64_t adam_step_ = 0;
    // Direct skip map from pooled latent to one logit per eigenvalue component (eigen_dim x d_model).
    Eigen::MatrixXd weights_;
    // Bias per eigenvalue component logit (eigen_dim).
    Eigen::VectorXd biases_;
    // Hidden MLP weights (kHiddenDim x d_model).
    Eigen::MatrixXd hidden_weights_;
    // Hidden MLP biases (kHiddenDim).
    Eigen::VectorXd hidden_biases_;
    // Output projection from hidden MLP to eigenvalue sign logits (eigen_dim x kHiddenDim).
    Eigen::MatrixXd out_weights_;
    // Positive-class balance weights per component.
    Eigen::VectorXd pos_weights_;
    // Negative-class balance weights per component.
    Eigen::VectorXd neg_weights_;
    // Adam moments for all parameter tensors.
    AdamMoment m_weights_;
    AdamMoment m_biases_;
    AdamMoment m_hid_w_;
    AdamMoment m_hid_b_;
    AdamMoment m_out_;
};

}  // namespace forward_analysis
