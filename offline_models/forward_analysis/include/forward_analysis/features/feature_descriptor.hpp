#pragma once

#include "forward_analysis/domains/analysis_domain.hpp"
#include "forward_analysis/features/feature_normalization_strategy.hpp"
#include "forward_analysis/features/feature_value_type.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Self-describing numerical feature produced by a domain feature extractor.
struct FeatureDescriptor {
    // Unique feature identifier within its extractor, e.g. "spectral.input_cov_eigenvalues".
    std::string id;
    // Mathematical analysis domain this feature belongs to.
    AnalysisDomain domain = AnalysisDomain::Statistical;
    // Value category (Scalar, Boolean, Integer, Vector, Matrix, Distribution, Categorical).
    FeatureValueType type = FeatureValueType::Scalar;
    // Normalization strategy applied to produce bounded values.
    FeatureNormalizationStrategy normalization = FeatureNormalizationStrategy::SignedUnitInterval;
    // Names of telemetry tensors required to compute this feature.
    std::vector<std::string> required_tensors;
    // Flattened numerical values after normalization or missing-data substitution.
    std::vector<double> values;
    // Logical tensor dimensions of `values` (e.g. {1} for scalar, {4, 4} for matrix).
    std::vector<std::size_t> shape;
    // True when the input telemetry met all numerical validity conditions for this feature.
    bool valid = true;
    // Fallback value substituted when a feature entry is missing or degenerate.
    double missing_fill_value = 0.0;
    // Relative computational cost estimate (1.0 = single pass over a column).
    double computational_cost = 1.0;

    // Applies the configured differentiable transformation to a raw value.
    static double normalize_value(double raw,
                                  FeatureNormalizationStrategy strategy,
                                  double fallback = 0.0) noexcept {
        if (!std::isfinite(raw)) {
            return fallback;
        }
        switch (strategy) {
            case FeatureNormalizationStrategy::Identity:
                return raw;
            case FeatureNormalizationStrategy::UnitInterval:
                return raw < 0.0 ? 0.0 : (raw > 1.0 ? 1.0 : raw);
            case FeatureNormalizationStrategy::SignedUnitInterval:
                return raw < -1.0 ? -1.0 : (raw > 1.0 ? 1.0 : raw);
            case FeatureNormalizationStrategy::TanhSquash:
                return std::tanh(raw);
            case FeatureNormalizationStrategy::LogScale:
                return std::tanh(std::log1p(std::abs(raw)) * (raw >= 0.0 ? 1.0 : -1.0));
            case FeatureNormalizationStrategy::StandardScore:
                return std::tanh(0.5 * raw);
        }
        return raw;
    }

    // Derivative of normalize_value with respect to the raw value.
    static double normalize_derivative(double raw, FeatureNormalizationStrategy strategy) noexcept {
        if (!std::isfinite(raw)) {
            return 0.0;
        }
        switch (strategy) {
            case FeatureNormalizationStrategy::Identity:
                return 1.0;
            case FeatureNormalizationStrategy::UnitInterval:
                return (raw >= 0.0 && raw <= 1.0) ? 1.0 : 0.0;
            case FeatureNormalizationStrategy::SignedUnitInterval:
                return (raw >= -1.0 && raw <= 1.0) ? 1.0 : 0.0;
            case FeatureNormalizationStrategy::TanhSquash: {
                // Hyperbolic tangent value at `raw`.
                const double t = std::tanh(raw);
                return 1.0 - t * t;
            }
            case FeatureNormalizationStrategy::LogScale: {
                // Log-scaled magnitude with sign preserved.
                const double u = std::log1p(std::abs(raw)) * (raw >= 0.0 ? 1.0 : -1.0);
                // Hyperbolic tangent of the log-scaled value.
                const double t = std::tanh(u);
                return (1.0 - t * t) / (1.0 + std::abs(raw));
            }
            case FeatureNormalizationStrategy::StandardScore: {
                // Scaled hyperbolic tangent at `0.5 * raw`.
                const double t = std::tanh(0.5 * raw);
                return 0.5 * (1.0 - t * t);
            }
        }
        return 1.0;
    }
};

}  // namespace forward_analysis
