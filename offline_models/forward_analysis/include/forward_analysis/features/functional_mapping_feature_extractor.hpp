#pragma once

#include "forward_analysis/features/domain_feature_extractor.hpp"
#include "forward_analysis/math/statistics_ops.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Domain E: Functional mapping analysis (prediction-to-target alignment, sign preservation, range ratios, residual structure, monotonicity).
class FunctionalMappingFeatureExtractor : public DomainFeatureExtractor {
public:
    std::string extractor_id() const override { return "functional_mapping_analysis"; }

    AnalysisDomain domain() const override { return AnalysisDomain::Functional; }

    std::vector<std::string> required_tensors() const override {
        return {"predictions", "targets", "residuals"};
    }

    FeatureNormalizationStrategy normalization_strategy() const noexcept override {
        return FeatureNormalizationStrategy::SignedUnitInterval;
    }

    double computational_cost() const noexcept override { return 1.5; }

    std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                           const DistillerConfig& config) const override {
        // Output feature list for the functional mapping domain.
        std::vector<FeatureDescriptor> out;
        if (!is_valid(telemetry)) {
            // Invalid placeholder descriptor.
            FeatureDescriptor invalid;
            invalid.id = "functional.invalid";
            invalid.domain = AnalysisDomain::Functional;
            invalid.valid = false;
            out.push_back(invalid);
            return out;
        }

        // Number of output coordinates.
        const Eigen::Index d_out = telemetry.targets.cols();
        // Reciprocal of output coordinate count.
        const double inv_out = 1.0 / static_cast<double>(std::max<Eigen::Index>(1, d_out));
        // Count of outputs with smaller prediction range than target range.
        std::size_t smaller_range = 0;
        // Count of outputs with larger prediction range than target range.
        std::size_t larger_range = 0;
        // Count of outputs with broadly comparable prediction and target ranges.
        std::size_t comparable_range = 0;
        // Total entry-wise sign agreements across all samples and outputs.
        std::size_t sign_agree = 0;
        // Count of outputs preserving sign on every sample.
        std::size_t all_sign_preserved = 0;
        // Count of outputs preserving pairwise sample order.
        std::size_t order_preserved = 0;
        // Mean Pearson correlation between prediction and target columns.
        double mean_corr = 0.0;
        // Mean Spearman rank correlation between prediction and target columns.
        double mean_spearman = 0.0;
        // Mean range ratio clamped to [0, 2].
        double mean_range_ratio = 0.0;
        // Mean residual-to-target standard deviation ratio.
        double mean_residual_ratio = 0.0;
        // Mean correlation between residuals and targets (captures systematic bias/underfitting).
        double mean_residual_target_corr = 0.0;

        for (Eigen::Index k = 0; k < d_out; ++k) {
            // Target column k.
            const Eigen::VectorXd t = telemetry.targets.col(k);
            // Prediction column k.
            const Eigen::VectorXd p = telemetry.predictions.col(k);
            // Residual column k.
            const Eigen::VectorXd r = telemetry.residuals.col(k);
            // Target peak-to-peak range.
            const double t_range = t.size() > 0 ? (t.maxCoeff() - t.minCoeff()) : 0.0;
            // Prediction-to-target range ratio.
            const double ratio = t_range > 1e-12 ? (p.maxCoeff() - p.minCoeff()) / t_range : 1.0;
            if (ratio < config.range_ratio_low) {
                ++smaller_range;
            } else if (ratio > config.range_ratio_high) {
                ++larger_range;
            } else {
                ++comparable_range;
            }
            mean_range_ratio += std::min(2.0, std::max(0.0, ratio)) * inv_out;

            // Whether every sample in column k preserves sign.
            bool col_sign_ok = true;
            for (Eigen::Index i = 0; i < t.size(); ++i) {
                if (std::abs(t(i)) > 1e-12 && !(t(i) * p(i) > 0.0)) {
                    col_sign_ok = false;
                }
                if (((t(i) > 0.0) - (t(i) < 0.0)) == ((p(i) > 0.0) - (p(i) < 0.0))) {
                    ++sign_agree;
                }
            }
            if (col_sign_ok) {
                ++all_sign_preserved;
            }
            if (StatisticsOps::same_order(t, p)) {
                ++order_preserved;
            }
            mean_corr += StatisticsOps::pearson(t, p) * inv_out;
            mean_spearman += StatisticsOps::spearman(t, p) * inv_out;
            // Target standard deviation for residual scaling.
            const double sd_t = std::max(1e-9, StatisticsOps::standard_deviation(t));
            mean_residual_ratio += std::tanh(StatisticsOps::standard_deviation(r) / sd_t) * inv_out;
            mean_residual_target_corr += StatisticsOps::pearson(r, t) * inv_out;
        }

        // Total number of output entries (samples x outputs).
        const double total_entries = static_cast<double>( std::max<Eigen::Index>(1, telemetry.targets.size()));

        // Functional mapping alignment and error structure descriptor.
        FeatureDescriptor func;
        func.id = "functional.alignment_and_residual_structure";
        func.domain = AnalysisDomain::Functional;
        func.type = FeatureValueType::Vector;
        func.normalization = FeatureNormalizationStrategy::SignedUnitInterval;
        func.required_tensors = {"predictions", "targets", "residuals"};
        func.shape = {11};
        func.computational_cost = 1.5;
        func.values = {
            static_cast<double>(smaller_range) * inv_out,
            static_cast<double>(larger_range) * inv_out,
            static_cast<double>(comparable_range) * inv_out,
            static_cast<double>(sign_agree) / total_entries,
            static_cast<double>(all_sign_preserved) * inv_out,
            static_cast<double>(order_preserved) * inv_out,
            FeatureDescriptor::normalize_value(mean_corr, FeatureNormalizationStrategy::SignedUnitInterval),
            FeatureDescriptor::normalize_value(mean_spearman, FeatureNormalizationStrategy::SignedUnitInterval),
            0.5 * mean_range_ratio,
            mean_residual_ratio,
            FeatureDescriptor::normalize_value(mean_residual_target_corr, FeatureNormalizationStrategy::SignedUnitInterval)};
        out.push_back(func);

        return out;
    }
};

}  // namespace forward_analysis
