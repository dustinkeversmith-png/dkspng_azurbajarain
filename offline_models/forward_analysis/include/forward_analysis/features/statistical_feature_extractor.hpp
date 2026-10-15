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

// Domain A: Statistical analysis (moments, distribution shape, correlation/dependence, sample order, conditional stats).
class StatisticalFeatureExtractor : public DomainFeatureExtractor {
public:
    std::string extractor_id() const override { return "statistical_analysis"; }

    AnalysisDomain domain() const override { return AnalysisDomain::Statistical; }

    std::vector<std::string> required_tensors() const override {
        return {"inputs", "layer_activations", "predictions", "targets"};
    }

    FeatureNormalizationStrategy normalization_strategy() const noexcept override {
        return FeatureNormalizationStrategy::TanhSquash;
    }

    double computational_cost() const noexcept override { return 2.0; }

    std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                           const DistillerConfig& config) const override {
        // Output feature list for the statistical domain.
        std::vector<FeatureDescriptor> out;
        // Validity flag for this telemetry batch.
        const bool ok = is_valid(telemetry);
        if (!ok) {
            out.push_back(make_invalid("statistical.input_moments"));
            return out;
        }

        // Number of observations in the batch.
        const Eigen::Index n = telemetry.inputs.rows();
        // Number of input coordinates.
        const Eigen::Index d_in = telemetry.inputs.cols();
        // Final hidden layer activation matrix.
        const Eigen::MatrixXd& last_hidden = telemetry.layer_activations.back();

        // Mean, standard deviation, skewness, and kurtosis summaries across input columns.
        double mean_acc = 0.0;
        // Accumulated standard deviation across input columns.
        double std_acc = 0.0;
        // Accumulated third standardized moment (skewness) across input columns.
        double skew_acc = 0.0;
        // Accumulated fourth standardized moment (excess kurtosis) across input columns.
        double kurt_acc = 0.0;
        // Share of entries exceeding 2 standard deviations from their column mean.
        double outlier_acc = 0.0;
        // Mean lag-1 sample autocorrelation across input columns (distinguishes sample orderings).
        double lag1_autocorr_acc = 0.0;
        // Mean correlation with sample index 0..N-1 (captures monotone sample ordering).
        double index_corr_acc = 0.0;

        // Linear ramp 0, 1, ..., N-1 for sample-order sensitivity.
        Eigen::VectorXd ramp(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            ramp(i) = static_cast<double>(i);
        }

        for (Eigen::Index c = 0; c < d_in; ++c) {
            // Single input coordinate column vector.
            const Eigen::VectorXd col = telemetry.inputs.col(c);
            // Column sample mean.
            const double mu = col.mean();
            // Column standard deviation.
            const double sd = StatisticsOps::standard_deviation(col);
            mean_acc += mu;
            std_acc += sd;
            if (sd > 1e-9) {
                // Standardized column values.
                const Eigen::ArrayXd z = (col.array() - mu) / sd;
                skew_acc += z.cube().mean();
                kurt_acc += z.square().square().mean() - 3.0;
                outlier_acc += (z.abs() > 2.0).cast<double>().mean();
            }
            if (n >= 3) {
                lag1_autocorr_acc += StatisticsOps::pearson(col.head(n - 1), col.tail(n - 1));
            }
            index_corr_acc += StatisticsOps::pearson(col, ramp);
        }
        // Reciprocal of the input dimension.
        const double inv_in = 1.0 / static_cast<double>(std::max<Eigen::Index>(1, d_in));

        // Moment and distribution-shape descriptor.
        FeatureDescriptor moments;
        moments.id = "statistical.input_moments_and_shape";
        moments.domain = AnalysisDomain::Statistical;
        moments.type = FeatureValueType::Vector;
        moments.normalization = FeatureNormalizationStrategy::TanhSquash;
        moments.required_tensors = {"inputs"};
        moments.shape = {5};
        moments.computational_cost = 1.0;
        moments.values = {
            FeatureDescriptor::normalize_value(mean_acc * inv_in, FeatureNormalizationStrategy::TanhSquash),
            FeatureDescriptor::normalize_value(std_acc * inv_in, FeatureNormalizationStrategy::TanhSquash),
            FeatureDescriptor::normalize_value(skew_acc * inv_in, FeatureNormalizationStrategy::StandardScore),
            FeatureDescriptor::normalize_value(kurt_acc * inv_in, FeatureNormalizationStrategy::StandardScore),
            FeatureDescriptor::normalize_value(outlier_acc * inv_in, FeatureNormalizationStrategy::UnitInterval)};
        out.push_back(moments);

        // Sample-ordering descriptor (sensitive to permutations of rows that preserve column marginals).
        FeatureDescriptor ordering;
        ordering.id = "statistical.sample_order_dependence";
        ordering.domain = AnalysisDomain::Statistical;
        ordering.type = FeatureValueType::Vector;
        ordering.normalization = FeatureNormalizationStrategy::SignedUnitInterval;
        ordering.required_tensors = {"inputs", "layer_activations"};
        ordering.shape = {3};
        ordering.computational_cost = 0.5;
        // First hidden layer lag-1 autocorrelation across columns.
        double hidden_lag1 = 0.0;
        if (n >= 3) {
            for (Eigen::Index u = 0; u < last_hidden.cols(); ++u) {
                hidden_lag1 += StatisticsOps::pearson(last_hidden.col(u).head(n - 1), last_hidden.col(u).tail(n - 1));
            }
            hidden_lag1 /= static_cast<double>( std::max<Eigen::Index>(1, last_hidden.cols()));
        }
        ordering.values = {
            FeatureDescriptor::normalize_value(lag1_autocorr_acc * inv_in, FeatureNormalizationStrategy::SignedUnitInterval),
            FeatureDescriptor::normalize_value(index_corr_acc * inv_in, FeatureNormalizationStrategy::SignedUnitInterval),
            FeatureDescriptor::normalize_value(hidden_lag1, FeatureNormalizationStrategy::SignedUnitInterval)};
        out.push_back(ordering);

        // Pairwise off-diagonal input correlation structure (sensitive to covariance changes with identical marginals).
        double abs_corr_sum = 0.0;
        // Signed off-diagonal correlation mean.
        double signed_corr_sum = 0.0;
        // Maximum off-diagonal absolute correlation among inputs.
        double max_abs_corr = 0.0;
        // Number of off-diagonal input pairs inspected.
        std::size_t pair_count = 0;
        for (Eigen::Index i = 0; i < d_in; ++i) {
            for (Eigen::Index j = i + 1; j < d_in; ++j) {
                // Pairwise Pearson correlation between input i and input j.
                const double r = StatisticsOps::pearson(telemetry.inputs.col(i), telemetry.inputs.col(j));
                abs_corr_sum += std::abs(r);
                signed_corr_sum += r;
                max_abs_corr = std::max(max_abs_corr, std::abs(r));
                ++pair_count;
            }
        }
        // Reciprocal of pair count.
        const double inv_pairs = pair_count > 0 ? 1.0 / static_cast<double>(pair_count) : 0.0;

        // Input-to-final-hidden association profile (strongest, second strongest, weakest, nonzero flag).
        std::vector<double> input_assoc(static_cast<std::size_t>(d_in), 0.0);
        for (Eigen::Index c = 0; c < d_in; ++c) {
            // Peak absolute correlation between input c and any unit in the final hidden layer.
            double best = 0.0;
            for (Eigen::Index k = 0; k < last_hidden.cols(); ++k) {
                best = std::max(best, std::abs(StatisticsOps::pearson(telemetry.inputs.col(c), last_hidden.col(k))));
            }
            input_assoc[static_cast<std::size_t>(c)] = best;
        }
        // Sorted copy of input associations in descending order.
        std::vector<double> sorted_assoc = input_assoc;
        std::sort(sorted_assoc.begin(), sorted_assoc.end(), std::greater<double>());
        // Strongest input association value.
        const double strongest = sorted_assoc.empty() ? 0.0 : sorted_assoc.front();
        // Second-strongest input association value.
        const double second_strongest = sorted_assoc.size() >= 2 ? sorted_assoc[1] : strongest;
        // Weakest input association value.
        const double weakest = sorted_assoc.empty() ? 0.0 : sorted_assoc.back();

        // Correlation and association descriptor.
        FeatureDescriptor dependence;
        dependence.id = "statistical.correlation_and_association";
        dependence.domain = AnalysisDomain::Statistical;
        dependence.type = FeatureValueType::Vector;
        dependence.normalization = FeatureNormalizationStrategy::SignedUnitInterval;
        dependence.required_tensors = {"inputs", "layer_activations"};
        dependence.shape = {7};
        dependence.computational_cost = 1.0;
        dependence.values = {
            FeatureDescriptor::normalize_value(abs_corr_sum * inv_pairs, FeatureNormalizationStrategy::UnitInterval),
            FeatureDescriptor::normalize_value(signed_corr_sum * inv_pairs, FeatureNormalizationStrategy::SignedUnitInterval),
            FeatureDescriptor::normalize_value(max_abs_corr, FeatureNormalizationStrategy::UnitInterval),
            FeatureDescriptor::normalize_value(strongest, FeatureNormalizationStrategy::UnitInterval),
            FeatureDescriptor::normalize_value(second_strongest, FeatureNormalizationStrategy::UnitInterval),
            FeatureDescriptor::normalize_value(weakest, FeatureNormalizationStrategy::UnitInterval),
            strongest > config.dead_activation_threshold ? 1.0 : 0.0};
        out.push_back(dependence);

        return out;
    }

private:
    static FeatureDescriptor make_invalid(const std::string& id) {
        // Invalid placeholder descriptor returned when telemetry preconditions fail.
        FeatureDescriptor d;
        d.id = id;
        d.domain = AnalysisDomain::Statistical;
        d.valid = false;
        return d;
    }
};

}  // namespace forward_analysis
