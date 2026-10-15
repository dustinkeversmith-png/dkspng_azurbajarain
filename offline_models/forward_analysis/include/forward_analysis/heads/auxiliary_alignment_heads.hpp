#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/heads/auxiliary_evaluation.hpp"
#include "forward_analysis/heads/eigen_sign_head.hpp"
#include "forward_analysis/heads/encoded_representation.hpp"
#include "forward_analysis/heads/numerical_prediction_head_registry.hpp"
#include "forward_analysis/heads/numerical_targets.hpp"
#include "forward_analysis/heads/observation_regression_head.hpp"

#include <Eigen/Core>

#include <cstdint>

namespace forward_analysis {

// Coordinates the NumericalPredictionHeadRegistry across all six mathematical domains alongside the
// backward-compatible ObservationRegressionHead (Aux1) and EigenSignHead (Aux2).
class AuxiliaryAlignmentHeads {
public:
    explicit AuxiliaryAlignmentHeads(const DistillerConfig& config = DistillerConfig{},
                                     std::uint64_t seed = 3000ULL)
        : config_(config),
          observation_head_(config.d_model, ObservationDescriptor::dim(), seed + 1ULL),
          registry_(config, seed + 100ULL) {
        config_.validate();
    }

    AuxiliaryEvaluation evaluate(const Eigen::MatrixXd& latents_z,
                                 const Eigen::VectorXd& target_descriptor = Eigen::VectorXd(),
                                 const Eigen::VectorXd& target_eigen_signs = Eigen::VectorXd(),
                                 const NetworkObservations* measured_obs = nullptr) const {
        // Encoded representation wrapping Z and domain-specific latent vectors.
        const EncodedRepresentation rep = EncodedRepresentation::from_latents(latents_z);
        // Output evaluation bundle.
        AuxiliaryEvaluation out;
        out.domain_predictions = registry_.forward_all(rep);

        // Direct regression head prediction blended with the domain-head synthesized descriptor.
        const Eigen::VectorXd direct_desc = observation_head_.predict(rep.pooled_z);
        // Descriptor synthesized from the calibrated domain heads.
        const Eigen::VectorXd synth_desc =
            NumericalPredictionHeadRegistry::synthesize_descriptor(out.domain_predictions);
        out.predicted_observation_descriptor = 0.5 * direct_desc + 0.5 * synth_desc;
        out.decoder_conditioning_vector = NumericalPredictionHeadRegistry::synthesize_decoder_conditioning(
            out.domain_predictions, out.predicted_observation_descriptor);

        if (out.domain_predictions.count("eigen_signs") > 0) {
            out.predicted_eigen_signs = out.domain_predictions.at("eigen_signs").discrete_predictions;
        } else {
            out.predicted_eigen_signs = registry_.eigen_sign_head().predict_signs(rep.pooled_z);
        }
        populate_structural_probabilities(out);

        if (target_descriptor.size() == out.predicted_observation_descriptor.size()) {
            out.aux1_observation_loss =
                0.5 * (out.predicted_observation_descriptor - target_descriptor).squaredNorm() /
                static_cast<double>(target_descriptor.size());
        }
        if (target_eigen_signs.size() > 0) {
            out.aux2_eigen_sign_loss =
                registry_.eigen_sign_head().compute_loss(rep.domain_vector(AnalysisDomain::Spectral), target_eigen_signs);
        }
        if (measured_obs != nullptr) {
            // Ground-truth multi-head targets built from measured observations.
            const NumericalTargets num_targets =
                NumericalTargets::from_observations(*measured_obs, target_eigen_signs, config_);
            out.domain_losses = registry_.evaluate_losses(out.domain_predictions, num_targets);
        }
        return out;
    }

    AuxiliaryEvaluation compute_losses_and_step(const Eigen::MatrixXd& latents_z,
                                                const Eigen::VectorXd& target_descriptor,
                                                const Eigen::VectorXd& target_eigen_signs,
                                                double learning_rate,
                                                Eigen::MatrixXd& d_latents_z_accum,
                                                const NetworkObservations* measured_obs = nullptr) {
        if (d_latents_z_accum.rows() != latents_z.rows() || d_latents_z_accum.cols() != latents_z.cols()) {
            d_latents_z_accum = Eigen::MatrixXd::Zero(latents_z.rows(), latents_z.cols());
        }
        // Encoded representation wrapping Z.
        const EncodedRepresentation rep = EncodedRepresentation::from_latents(latents_z);
        // Pooled gradient accumulator for the direct ObservationRegressionHead.
        Eigen::VectorXd d_pooled = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(config_.d_model));

        // Output evaluation bundle.
        AuxiliaryEvaluation out;
        out.domain_predictions = registry_.forward_all(rep);

        // Direct regression prediction before update.
        const Eigen::VectorXd direct_desc = observation_head_.predict(rep.pooled_z);
        // Domain-head synthesized descriptor before update.
        const Eigen::VectorXd synth_desc =
            NumericalPredictionHeadRegistry::synthesize_descriptor(out.domain_predictions);
        out.predicted_observation_descriptor = 0.5 * direct_desc + 0.5 * synth_desc;
        out.decoder_conditioning_vector = NumericalPredictionHeadRegistry::synthesize_decoder_conditioning(
            out.domain_predictions, out.predicted_observation_descriptor);
        out.predicted_eigen_signs = out.domain_predictions.at("eigen_signs").discrete_predictions;
        populate_structural_probabilities(out);

        if (target_descriptor.size() == out.predicted_observation_descriptor.size()) {
            out.aux1_observation_loss =
                0.5 * (out.predicted_observation_descriptor - target_descriptor).squaredNorm() /
                static_cast<double>(target_descriptor.size());
        }
        if (target_eigen_signs.size() > 0) {
            out.aux2_eigen_sign_loss =
                registry_.eigen_sign_head().compute_loss(rep.domain_vector(AnalysisDomain::Spectral), target_eigen_signs);
        }

        // Update the direct regression head.
        observation_head_.compute_loss_and_step(
            rep.pooled_z, target_descriptor, learning_rate, config_.aux1_observation_weight, d_pooled);

        if (measured_obs != nullptr) {
            // Multi-head numerical targets with missing-label masks.
            const NumericalTargets num_targets =
                NumericalTargets::from_observations(*measured_obs, target_eigen_signs, config_);
            out.domain_losses =
                registry_.compute_losses_and_step(rep, num_targets, learning_rate, d_latents_z_accum);
        } else if (target_eigen_signs.size() > 0) {
            // Fallback when only target_eigen_signs is provided.
            Eigen::VectorXd d_spec = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(config_.d_model));
            registry_.eigen_sign_head().compute_loss_and_step(
                rep.domain_vector(AnalysisDomain::Spectral), target_eigen_signs, learning_rate,
                config_.aux2_eigen_sign_weight, d_spec);
            EncodedRepresentation::accumulate_domain_gradient(AnalysisDomain::Spectral, d_spec, d_latents_z_accum);
        }

        d_latents_z_accum.rowwise() += d_pooled.transpose() / static_cast<double>(latents_z.rows());
        return out;
    }

    const ObservationRegressionHead& observation_head() const noexcept { return observation_head_; }
    const EigenSignHead& eigen_sign_head() const noexcept { return registry_.eigen_sign_head(); }
    const NumericalPredictionHeadRegistry& registry() const noexcept { return registry_; }
    NumericalPredictionHeadRegistry& registry() noexcept { return registry_; }

private:
    static void populate_structural_probabilities(AuxiliaryEvaluation& out) {
        if (out.domain_predictions.count("layer_structure") > 0 &&
            out.domain_predictions.at("layer_structure").calibrated_outputs.size() >= 1) {
            out.predicted_two_layer_probability =
                out.domain_predictions.at("layer_structure").calibrated_outputs(0);
        }
        if (out.domain_predictions.count("narrow_representation") > 0 &&
            out.domain_predictions.at("narrow_representation").calibrated_outputs.size() >= 1) {
            out.predicted_narrow_probability =
                out.domain_predictions.at("narrow_representation").calibrated_outputs(0);
        }
        if (out.domain_predictions.count("feature_association") > 0 &&
            out.domain_predictions.at("feature_association").calibrated_outputs.size() >= 5) {
            out.predicted_nonzero_association_probability =
                out.domain_predictions.at("feature_association").calibrated_outputs(4);
        }
    }

    // d_model, eigen_sign_dim and the Aux1/Aux2 loss weights.
    DistillerConfig config_;
    // Aux1: regresses the observation descriptor from the mean-pooled latents.
    ObservationRegressionHead observation_head_;
    // Registry of domain-specific NumericalPredictionHeads (including EigenSignPredictionHead).
    NumericalPredictionHeadRegistry registry_;
};

}  // namespace forward_analysis
