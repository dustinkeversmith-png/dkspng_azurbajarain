#pragma once

#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

class EigenSignHead {
public:
    explicit EigenSignHead(std::size_t d_model = 256,
                           std::size_t eigen_dim = 4,
                           std::uint64_t seed = 3002ULL)
        : d_model_(d_model),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(eigen_dim))) {
        DeterministicRng rng(seed);
        weights_ = rng.xavier_matrix(eigen_dim, d_model_);
    }

    Eigen::VectorXd predict_sign_probabilities(const Eigen::VectorXd& pooled_z) const {
        if (pooled_z.size() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("EigenSignHead input width must equal d_model");
        }
        return MatrixOps::sigmoid(weights_ * pooled_z + biases_);
    }

    Eigen::VectorXd predict_signs(const Eigen::VectorXd& pooled_z) const {
        return predict_sign_probabilities(pooled_z).unaryExpr(
            [](double p) { return p >= 0.5 ? 1.0 : -1.0; });
    }

    double compute_loss(const Eigen::VectorXd& pooled_z,
                        const Eigen::VectorXd& target_signs) const {
        const Eigen::VectorXd probs = predict_sign_probabilities(pooled_z);
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
        const Eigen::VectorXd probs = predict_sign_probabilities(pooled_z);
        const Eigen::Index k = std::min(probs.size(), target_signs.size());
        if (k == 0) {
            return 0.0;
        }
        if (d_pooled_accum.size() != static_cast<Eigen::Index>(d_model_)) {
            d_pooled_accum = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d_model_));
        }

        const Eigen::ArrayXd p = clamp_probs(probs.head(k));
        const Eigen::ArrayXd y = sign_labels(target_signs.head(k));
        const double inv_k = 1.0 / static_cast<double>(k);
        const Eigen::VectorXd d_logit = ((p - y) * (inv_k * loss_weight)).matrix();

        d_pooled_accum.noalias() += weights_.topRows(k).transpose() * d_logit;
        if (learning_rate > 0.0) {
            biases_.head(k) -= learning_rate * d_logit;
            weights_.topRows(k).noalias() -= (learning_rate * d_logit) * pooled_z.transpose();
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
    // Linear map from pooled latent to one logit per eigenvector component (eigen_dim x d_model).
    Eigen::MatrixXd weights_;
    // Bias per eigenvector component logit (eigen_dim).
    Eigen::VectorXd biases_;
};

}  // namespace forward_analysis
