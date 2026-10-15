#pragma once

#include "forward_analysis/heads/eigen_sign_head.hpp"
#include "forward_analysis/heads/numerical_prediction_head.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace forward_analysis {

// Domain B head wrapping EigenSignHead inside the NumericalPredictionHead registry interface.
class EigenSignPredictionHead : public NumericalPredictionHead {
public:
    explicit EigenSignPredictionHead(std::size_t d_model = 256,
                                     std::size_t eigen_dim = 4,
                                     double weight = 0.20,
                                     std::uint64_t seed = 3002ULL)
        : eigen_dim_(eigen_dim),
          weight_(weight),
          head_(d_model, eigen_dim, seed) {}

    std::string id() const override { return "eigen_signs"; }
    AnalysisDomain domain() const override { return AnalysisDomain::Spectral; }
    std::size_t output_dim() const noexcept override { return eigen_dim_; }
    double loss_weight() const noexcept override { return weight_; }

    PredictionResult forward(const EncodedRepresentation& representation) const override {
        // Spectral domain vector routed from Perceiver latents.
        const Eigen::VectorXd x = representation.domain_vector(AnalysisDomain::Spectral);
        // Predicted positive-sign probabilities in [0, 1].
        const Eigen::VectorXd probs = head_.predict_sign_probabilities(x);
        // Thresholded signs in {-1, +1}.
        const Eigen::VectorXd signs = head_.predict_signs(x);

        // Assembled prediction result.
        PredictionResult res;
        res.head_id = id();
        res.domain = domain();
        res.raw_outputs = probs;
        res.calibrated_outputs = probs;
        res.discrete_predictions = signs;
        res.confidence = (2.0 * (probs.array() - 0.5).abs()).mean();
        return res;
    }

    LossResult compute_loss(const PredictionResult& prediction,
                            const NumericalTargets& targets) const override {
        // Assembled loss result.
        LossResult out;
        out.head_id = id();
        out.domain = domain();
        if (!targets.has(id())) {
            return out;
        }
        // Binary target vector in {0, 1}.
        const Eigen::VectorXd& y = targets.targets.at(id());
        // Target signs in {-1, +1}.
        const Eigen::VectorXd target_signs = y.unaryExpr([](double v) { return v >= 0.5 ? 1.0 : -1.0; });
        // Number of evaluated components.
        const Eigen::Index k = std::min(prediction.calibrated_outputs.size(), y.size());
        if (k == 0) {
            return out;
        }

        // Clamped probabilities.
        const Eigen::ArrayXd p = prediction.calibrated_outputs.head(k).array().cwiseMax(1e-12).cwiseMin(1.0 - 1e-12);
        // Binary labels.
        const Eigen::ArrayXd lbl = y.head(k).array();
        out.unweighted_loss = -(lbl * p.log() + (1.0 - lbl) * (1.0 - p).log()).mean();
        out.weighted_loss = out.unweighted_loss * weight_;
        out.mean_absolute_error = (p - lbl).abs().mean();
        // Matching sign count.
        std::size_t matches = 0;
        for (Eigen::Index i = 0; i < k; ++i) {
            if (prediction.discrete_predictions(i) == target_signs(i)) {
                ++matches;
            }
        }
        out.accuracy = static_cast<double>(matches) / static_cast<double>(k);
        out.valid_entries = static_cast<std::size_t>(k);
        return out;
    }

    LossResult compute_loss_and_step(const EncodedRepresentation& representation,
                                     const NumericalTargets& targets,
                                     double learning_rate,
                                     Eigen::MatrixXd& d_latents_z_accum) override {
        // Forward evaluation before step.
        const PredictionResult pred = forward(representation);
        // Loss metrics before step.
        const LossResult loss = compute_loss(pred, targets);
        if (!targets.has(id()) || loss.valid_entries == 0) {
            return loss;
        }
        // Binary targets in {0, 1} converted back to {-1, +1} for EigenSignHead.
        const Eigen::VectorXd& y = targets.targets.at(id());
        // Target signs in {-1, +1}.
        const Eigen::VectorXd target_signs = y.unaryExpr([](double v) { return v >= 0.5 ? 1.0 : -1.0; });
        // Spectral domain vector.
        const Eigen::VectorXd x = representation.domain_vector(AnalysisDomain::Spectral);
        // Gradient with respect to the spectral domain vector.
        Eigen::VectorXd d_x = Eigen::VectorXd::Zero(x.size());
        head_.compute_loss_and_step(x, target_signs, learning_rate, weight_, d_x);
        EncodedRepresentation::accumulate_domain_gradient(AnalysisDomain::Spectral, d_x, d_latents_z_accum);
        return loss;
    }

    const EigenSignHead& inner_head() const noexcept { return head_; }
    EigenSignHead& inner_head() noexcept { return head_; }

private:
    // Number of eigenvalue sign components.
    std::size_t eigen_dim_ = 4;
    // Task loss weight.
    double weight_ = 0.20;
    // Underlying spectral MLP eigen-sign head.
    EigenSignHead head_;
};

}  // namespace forward_analysis
