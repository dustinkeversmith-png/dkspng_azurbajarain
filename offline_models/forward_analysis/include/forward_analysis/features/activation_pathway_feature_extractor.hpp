#pragma once

#include "forward_analysis/features/domain_feature_extractor.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Domain D: Activation and pathway analysis (dead/active neurons, nonpositive observations, saturation, gating, layer propagation).
class ActivationPathwayFeatureExtractor : public DomainFeatureExtractor {
public:
    std::string extractor_id() const override { return "activation_pathway_analysis"; }

    AnalysisDomain domain() const override { return AnalysisDomain::Activation; }

    std::vector<std::string> required_tensors() const override {
        return {"layer_activations"};
    }

    FeatureNormalizationStrategy normalization_strategy() const noexcept override {
        return FeatureNormalizationStrategy::UnitInterval;
    }

    double computational_cost() const noexcept override { return 1.5; }

    std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                           const DistillerConfig& config) const override {
        // Output feature list for the activation and pathway domain.
        std::vector<FeatureDescriptor> out;
        if (!is_valid(telemetry)) {
            // Invalid placeholder descriptor.
            FeatureDescriptor invalid;
            invalid.id = "activation.invalid";
            invalid.domain = AnalysisDomain::Activation;
            invalid.valid = false;
            out.push_back(invalid);
            return out;
        }

        // Number of observations in the batch.
        const double n = static_cast<double>(telemetry.num_samples());
        // Per-layer activation and sparsity profile (up to 2 layers, 5 features per layer).
        FeatureDescriptor pathway;
        pathway.id = "activation.layer_sparsity_and_gating";
        pathway.domain = AnalysisDomain::Activation;
        pathway.type = FeatureValueType::Vector;
        pathway.normalization = FeatureNormalizationStrategy::UnitInterval;
        pathway.required_tensors = {"layer_activations"};
        pathway.shape = {11};
        pathway.computational_cost = 1.5;

        for (std::size_t l = 0; l < 2; ++l) {
            if (l < telemetry.layer_activations.size()) {
                // Activation matrix of hidden layer l.
                const Eigen::MatrixXd& act = telemetry.layer_activations[l];
                // Absolute activation magnitudes.
                const Eigen::ArrayXXd mag = act.array().abs();
                // Width of hidden layer l.
                const double w = static_cast<double>(std::max<Eigen::Index>(1, act.cols()));
                // Count of units that remain zero throughout the batch.
                std::size_t dead_count = 0;
                // Count of units active on every observation in the batch.
                std::size_t always_active = 0;
                for (Eigen::Index u = 0; u < act.cols(); ++u) {
                    if (mag.col(u).maxCoeff() <= config.dead_activation_threshold) {
                        ++dead_count;
                    } else if (mag.col(u).minCoeff() > config.dead_activation_threshold) {
                        ++always_active;
                    }
                }
                // Count of observations with no positive coordinates in layer l.
                std::size_t nonpos_samples = 0;
                // Count of completely closed observations in layer l.
                std::size_t closed_samples = 0;
                for (Eigen::Index i = 0; i < act.rows(); ++i) {
                    if (act.row(i).maxCoeff() <= 0.0) {
                        ++nonpos_samples;
                    }
                    if (mag.row(i).maxCoeff() <= config.dead_activation_threshold) {
                        ++closed_samples;
                    }
                }
                pathway.values.push_back(static_cast<double>(dead_count) / w);
                pathway.values.push_back(static_cast<double>(always_active) / w);
                pathway.values.push_back(static_cast<double>(nonpos_samples) / n);
                pathway.values.push_back(static_cast<double>(closed_samples) / n);
                pathway.values.push_back(std::tanh(mag.mean()));
            } else {
                for (int k = 0; k < 5; ++k) {
                    pathway.values.push_back(0.0);
                }
            }
        }

        // Cross-layer energy propagation ratio (1.0 for 1-layer networks).
        double propagation_ratio = 1.0;
        if (telemetry.layer_activations.size() >= 2) {
            // Mean absolute activation of the first layer.
            const double e0 = telemetry.layer_activations.front().cwiseAbs().mean();
            // Mean absolute activation of the final layer.
            const double e1 = telemetry.layer_activations.back().cwiseAbs().mean();
            propagation_ratio = std::tanh(e1 / std::max(1e-9, e0));
        }
        pathway.values.push_back(propagation_ratio);
        out.push_back(pathway);

        return out;
    }
};

}  // namespace forward_analysis
