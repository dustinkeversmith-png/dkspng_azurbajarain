#pragma once

#include "forward_analysis/math/deterministic_rng.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

// Aux1: linear regression from the pooled latent vector to the ObservationDescriptor.
class ObservationRegressionHead {
public:
    explicit ObservationRegressionHead(std::size_t d_model = 256,
                                       std::size_t descriptor_dim = 14,
                                       std::uint64_t seed = 3001ULL)
        : d_model_(d_model),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(descriptor_dim))) {
        DeterministicRng rng(seed);
        weights_ = rng.xavier_matrix(descriptor_dim, d_model_);
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& pooled_z) const {
        if (pooled_z.size() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("ObservationRegressionHead input width must equal d_model");
        }
        return weights_ * pooled_z + biases_;
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
        const Eigen::VectorXd error = predict(pooled_z) - target;
        const double inv_k = 1.0 / static_cast<double>(target.size());
        const Eigen::VectorXd d_out = error * (inv_k * loss_weight);

        d_pooled_accum.noalias() += weights_.transpose() * d_out;
        if (learning_rate > 0.0) {
            const Eigen::VectorXd d_param = error * inv_k;
            biases_ -= learning_rate * d_param;
            weights_.noalias() -= (learning_rate * d_param) * pooled_z.transpose();
        }
        return 0.5 * error.squaredNorm() * inv_k;
    }

    std::size_t descriptor_dim() const noexcept { return static_cast<std::size_t>(biases_.size()); }

private:
    // Width of the pooled latent vector this head reads.
    std::size_t d_model_ = 256;
    // Linear map from pooled latent to descriptor (descriptor_dim x d_model).
    Eigen::MatrixXd weights_;
    // Bias per descriptor component (descriptor_dim).
    Eigen::VectorXd biases_;
};

}  // namespace forward_analysis
