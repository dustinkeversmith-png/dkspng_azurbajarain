#pragma once

#include "forward_analysis/features/domain_feature_extractor.hpp"
#include "forward_analysis/math/spectral_ops.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Domain C: Geometric analysis (latent dimensionality, narrow vs multidirectional, subspace collapse, collinearity, angular similarity, distance preservation).
class GeometricFeatureExtractor : public DomainFeatureExtractor {
public:
    std::string extractor_id() const override { return "geometric_analysis"; }

    AnalysisDomain domain() const override { return AnalysisDomain::Geometric; }

    std::vector<std::string> required_tensors() const override {
        return {"inputs", "layer_activations"};
    }

    FeatureNormalizationStrategy normalization_strategy() const noexcept override {
        return FeatureNormalizationStrategy::UnitInterval;
    }

    double computational_cost() const noexcept override { return 2.0; }

    std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                           const DistillerConfig& config) const override {
        // Output feature list for the geometric domain.
        std::vector<FeatureDescriptor> out;
        if (!is_valid(telemetry)) {
            // Invalid placeholder descriptor.
            FeatureDescriptor invalid;
            invalid.id = "geometric.invalid";
            invalid.domain = AnalysisDomain::Geometric;
            invalid.valid = false;
            out.push_back(invalid);
            return out;
        }

        // True when every hidden layer's numerical rank is within narrow_rank_limit.
        bool all_narrow = true;
        // Maximum numerical rank across hidden layers.
        std::size_t max_num_rank = 0;
        // Minimum numerical rank across hidden layers.
        std::size_t min_num_rank = 9999;
        // Count of layers lying on a ray from the origin.
        std::size_t ray_layers = 0;

        for (const Eigen::MatrixXd& act : telemetry.layer_activations) {
            // Centered activation matrix.
            const Eigen::MatrixXd centered = act.rowwise() - act.colwise().mean();
            // Centered singular values.
            const Eigen::VectorXd sv = Eigen::BDCSVD<Eigen::MatrixXd>(centered).singularValues();
            // Numerical rank of this layer.
            const std::size_t rank =
                static_cast<std::size_t>((sv.array() > config.numerical_rank_tolerance).count());
            all_narrow = all_narrow && (rank <= config.narrow_rank_limit);
            max_num_rank = std::max(max_num_rank, rank);
            min_num_rank = std::min(min_num_rank, rank);

            // Uncentered singular values for ray collinearity check.
            const Eigen::VectorXd raw_sv = SpectralOps::uncentered_singular_values(act);
            // Non-negativity check for ray collapse.
            const bool non_neg = act.minCoeff() >= -config.dead_activation_threshold;
            if (non_neg && raw_sv.size() > 0 && raw_sv(0) > 1e-12 &&
                (raw_sv.size() < 2 || raw_sv(1) <= config.ray_tolerance * raw_sv(0))) {
                ++ray_layers;
            }
        }

        // Final hidden layer matrix for angular similarity and distance preservation.
        const Eigen::MatrixXd& last = telemetry.layer_activations.back();
        // Number of observations in the batch.
        const Eigen::Index n = last.rows();
        // Mean pairwise cosine similarity between sample activation vectors.
        double mean_cosine = 0.0;
        // Pairwise Euclidean distance correlation between input space and final hidden space.
        double dist_corr = 0.0;
        if (n >= 2) {
            // Flattened pairwise input distances.
            std::vector<double> in_dists;
            // Flattened pairwise hidden distances.
            std::vector<double> hid_dists;
            // Count of sample pairs.
            std::size_t pairs = 0;
            for (Eigen::Index i = 0; i < n; ++i) {
                // Norm of sample i in the final hidden layer.
                const double ni = last.row(i).norm();
                for (Eigen::Index j = i + 1; j < n; ++j) {
                    // Norm of sample j in the final hidden layer.
                    const double nj = last.row(j).norm();
                    if (ni > 1e-12 && nj > 1e-12) {
                        mean_cosine += last.row(i).dot(last.row(j)) / (ni * nj);
                    }
                    in_dists.push_back((telemetry.inputs.row(i) - telemetry.inputs.row(j)).norm());
                    hid_dists.push_back((last.row(i) - last.row(j)).norm());
                    ++pairs;
                }
            }
            if (pairs > 0) {
                mean_cosine /= static_cast<double>(pairs);
                // Eigen vector of input pairwise distances.
                const Eigen::VectorXd v_in =
                    Eigen::Map<const Eigen::VectorXd>(in_dists.data(), static_cast<Eigen::Index>(pairs));
                // Eigen vector of hidden pairwise distances.
                const Eigen::VectorXd v_hid =
                    Eigen::Map<const Eigen::VectorXd>(hid_dists.data(), static_cast<Eigen::Index>(pairs));
                dist_corr = StatisticsOps::pearson(v_in, v_hid);
            }
        }

        // Geometric manifold and subspace descriptor.
        FeatureDescriptor geom;
        geom.id = "geometric.manifold_and_subspace";
        geom.domain = AnalysisDomain::Geometric;
        geom.type = FeatureValueType::Vector;
        geom.normalization = FeatureNormalizationStrategy::SignedUnitInterval;
        geom.required_tensors = {"inputs", "layer_activations"};
        geom.shape = {6};
        geom.computational_cost = 2.0;
        geom.values = {
            all_narrow ? 1.0 : 0.0,
            std::min(1.0, static_cast<double>(max_num_rank) / 8.0),
            std::min(1.0, static_cast<double>(min_num_rank == 9999 ? 0 : min_num_rank) / 8.0),
            static_cast<double>(ray_layers) / static_cast<double>(telemetry.layer_activations.size()),
            FeatureDescriptor::normalize_value(mean_cosine, FeatureNormalizationStrategy::SignedUnitInterval),
            FeatureDescriptor::normalize_value(dist_corr, FeatureNormalizationStrategy::SignedUnitInterval)};
        out.push_back(geom);

        return out;
    }
};

}  // namespace forward_analysis
