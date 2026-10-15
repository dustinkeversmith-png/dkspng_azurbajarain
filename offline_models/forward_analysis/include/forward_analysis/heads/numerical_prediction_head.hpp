#pragma once

#include "forward_analysis/domains/analysis_domain.hpp"
#include "forward_analysis/heads/encoded_representation.hpp"
#include "forward_analysis/heads/loss_result.hpp"
#include "forward_analysis/heads/numerical_targets.hpp"
#include "forward_analysis/heads/prediction_result.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <string>

namespace forward_analysis {

// Abstract interface for modular, domain-specific numerical prediction heads.
class NumericalPredictionHead {
public:
    virtual ~NumericalPredictionHead() = default;

    // Unique identifier of this prediction head.
    virtual std::string id() const = 0;

    // Mathematical analysis domain of this head.
    virtual AnalysisDomain domain() const = 0;

    // Dimensionality of this head's output vector.
    virtual std::size_t output_dim() const noexcept = 0;

    // Task loss weight used when balancing multi-head objectives.
    virtual double loss_weight() const noexcept = 0;

    // Evaluates this head on the shared Perceiver EncodedRepresentation.
    virtual PredictionResult forward(const EncodedRepresentation& representation) const = 0;

    // Computes the masked loss and metrics without updating parameters.
    virtual LossResult compute_loss(const PredictionResult& prediction,
                                    const NumericalTargets& targets) const = 0;

    // Computes loss, updates head parameters, and accumulates dLoss/dZ into `d_latents_z_accum`.
    virtual LossResult compute_loss_and_step(const EncodedRepresentation& representation,
                                             const NumericalTargets& targets,
                                             double learning_rate,
                                             Eigen::MatrixXd& d_latents_z_accum) = 0;
};

}  // namespace forward_analysis
