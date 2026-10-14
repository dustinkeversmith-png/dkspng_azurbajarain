#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/heads/auxiliary_evaluation.hpp"
#include "forward_analysis/heads/eigen_sign_head.hpp"
#include "forward_analysis/heads/observation_regression_head.hpp"

#include <Eigen/Core>

#include <cstdint>

namespace forward_analysis {

// Aux1 (observation descriptor) and Aux2 (eigen signs) heads, both reading mean-pooled latents.
class AuxiliaryAlignmentHeads {
public:
    explicit AuxiliaryAlignmentHeads(const DistillerConfig& config = DistillerConfig{},
                                     std::uint64_t seed = 3000ULL)
        : config_(config),
          observation_head_(config.d_model, ObservationDescriptor::dim(), seed + 1ULL),
          eigen_sign_head_(config.d_model, config.eigen_sign_dim, seed + 2ULL) {
        config_.validate();
    }

    AuxiliaryEvaluation evaluate(const Eigen::MatrixXd& latents_z,
                                 const Eigen::VectorXd& target_descriptor = Eigen::VectorXd(),
                                 const Eigen::VectorXd& target_eigen_signs = Eigen::VectorXd()) const {
        const Eigen::VectorXd pooled = latents_z.colwise().mean().transpose();
        AuxiliaryEvaluation out;
        out.predicted_observation_descriptor = observation_head_.predict(pooled);
        out.predicted_eigen_signs = eigen_sign_head_.predict_signs(pooled);
        out.aux1_observation_loss = observation_head_.compute_loss(pooled, target_descriptor);
        if (target_eigen_signs.size() > 0) {
            out.aux2_eigen_sign_loss = eigen_sign_head_.compute_loss(pooled, target_eigen_signs);
        }
        return out;
    }

    AuxiliaryEvaluation compute_losses_and_step(const Eigen::MatrixXd& latents_z,
                                                const Eigen::VectorXd& target_descriptor,
                                                const Eigen::VectorXd& target_eigen_signs,
                                                double learning_rate,
                                                Eigen::MatrixXd& d_latents_z_accum) {
        const Eigen::VectorXd pooled = latents_z.colwise().mean().transpose();
        Eigen::VectorXd d_pooled = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(config_.d_model));

        AuxiliaryEvaluation out;
        out.predicted_observation_descriptor = observation_head_.predict(pooled);
        out.predicted_eigen_signs = eigen_sign_head_.predict_signs(pooled);
        out.aux1_observation_loss = observation_head_.compute_loss_and_step(
            pooled, target_descriptor, learning_rate, config_.aux1_observation_weight, d_pooled);
        out.aux2_eigen_sign_loss = eigen_sign_head_.compute_loss_and_step(
            pooled, target_eigen_signs, learning_rate, config_.aux2_eigen_sign_weight, d_pooled);

        if (d_latents_z_accum.rows() != latents_z.rows() || d_latents_z_accum.cols() != latents_z.cols()) {
            d_latents_z_accum = Eigen::MatrixXd::Zero(latents_z.rows(), latents_z.cols());
        }
        d_latents_z_accum.rowwise() += d_pooled.transpose() / static_cast<double>(latents_z.rows());
        return out;
    }

    const ObservationRegressionHead& observation_head() const noexcept { return observation_head_; }
    const EigenSignHead& eigen_sign_head() const noexcept { return eigen_sign_head_; }

private:
    // d_model, eigen_sign_dim and the Aux1/Aux2 loss weights.
    DistillerConfig config_;
    // Aux1: regresses the observation descriptor from the mean-pooled latents.
    ObservationRegressionHead observation_head_;
    // Aux2: predicts dominant-eigenvector signs from the mean-pooled latents.
    EigenSignHead eigen_sign_head_;
};

}  // namespace forward_analysis
