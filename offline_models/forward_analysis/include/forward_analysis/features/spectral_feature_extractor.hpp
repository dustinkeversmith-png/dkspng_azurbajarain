#pragma once

#include "forward_analysis/features/domain_feature_extractor.hpp"
#include "forward_analysis/math/spectral_ops.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Domain B: Spectral analysis (zero-diagonal input covariance eigenvalues/signs/gaps, singular values, numerical and effective rank).
class SpectralFeatureExtractor : public DomainFeatureExtractor {
public:
    std::string extractor_id() const override { return "spectral_analysis"; }

    AnalysisDomain domain() const override { return AnalysisDomain::Spectral; }

    std::vector<std::string> required_tensors() const override {
        return {"inputs", "layer_activations"};
    }

    FeatureNormalizationStrategy normalization_strategy() const noexcept override {
        return FeatureNormalizationStrategy::SignedUnitInterval;
    }

    double computational_cost() const noexcept override { return 3.0; }

    std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                           const DistillerConfig& config) const override {
        // Output feature list for the spectral domain.
        std::vector<FeatureDescriptor> out;
        if (!is_valid(telemetry)) {
            // Invalid placeholder descriptor.
            FeatureDescriptor invalid;
            invalid.id = "spectral.invalid";
            invalid.domain = AnalysisDomain::Spectral;
            invalid.valid = false;
            out.push_back(invalid);
            return out;
        }

        // Number of input coordinates used for the zero-diagonal covariance spectrum.
        const Eigen::Index dim = static_cast<Eigen::Index>(config.eigen_sign_dim);
        // Active input columns available in this telemetry batch.
        const Eigen::Index used = std::min(dim, telemetry.inputs.cols());
        // Descending eigenvalues of the zero-diagonal covariance (padded with 0).
        Eigen::VectorXd descending_eigs = Eigen::VectorXd::Zero(dim);
        // Signs (+1 / -1) of the descending eigenvalues (zero counted as +1).
        Eigen::VectorXd eig_signs = Eigen::VectorXd::Ones(dim);
        // Upper-triangle entries of the 4x4 zero-diagonal covariance matrix (6 pairs).
        std::vector<double> off_diag_pairs(6, 0.0);

        if (used > 0 && telemetry.inputs.rows() > 0) {
            // Mean-centered first `used` input columns.
            const Eigen::MatrixXd centered =
                telemetry.inputs.leftCols(used).rowwise() - telemetry.inputs.leftCols(used).colwise().mean();
            // Sample covariance matrix with Bessel divisor max(1, N - 1).
            Eigen::MatrixXd cov =
                centered.transpose() * centered / std::max(1.0, static_cast<double>(telemetry.inputs.rows() - 1));
            cov.diagonal().setZero();
            // Frobenius norm of the zero-diagonal covariance for scale-invariant normalization.
            const double cov_scale = std::max(1e-9, cov.norm());
            // Pair index into off_diag_pairs.
            std::size_t p_idx = 0;
            for (Eigen::Index r = 0; r < 4; ++r) {
                for (Eigen::Index c = r + 1; c < 4; ++c) {
                    if (r < used && c < used) {
                        off_diag_pairs[p_idx] = cov(r, c) / cov_scale;
                    }
                    ++p_idx;
                }
            }
            // Self-adjoint solver on the zero-diagonal covariance matrix.
            const Eigen::VectorXd solved = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(cov).eigenvalues().reverse();
            descending_eigs.head(used) = solved / cov_scale;
            eig_signs.head(used) = solved.unaryExpr([](double v) { return v >= 0.0 ? 1.0 : -1.0; });
        }

        // Feature 1: Eigenvalue signs of the zero-diagonal input covariance.
        FeatureDescriptor sign_desc;
        sign_desc.id = "spectral.input_zero_diag_eigen_signs";
        sign_desc.domain = AnalysisDomain::Spectral;
        sign_desc.type = FeatureValueType::Vector;
        sign_desc.normalization = FeatureNormalizationStrategy::SignedUnitInterval;
        sign_desc.required_tensors = {"inputs"};
        sign_desc.shape = {static_cast<std::size_t>(dim)};
        sign_desc.computational_cost = 1.0;
        for (Eigen::Index i = 0; i < dim; ++i) {
            sign_desc.values.push_back(eig_signs(i));
        }
        out.push_back(sign_desc);

        // Feature 2: Normalized descending eigenvalues, spectral gaps, and off-diagonal covariance entries.
        FeatureDescriptor cov_spectrum;
        cov_spectrum.id = "spectral.input_zero_diag_spectrum_and_gaps";
        cov_spectrum.domain = AnalysisDomain::Spectral;
        cov_spectrum.type = FeatureValueType::Vector;
        cov_spectrum.normalization = FeatureNormalizationStrategy::SignedUnitInterval;
        cov_spectrum.required_tensors = {"inputs"};
        cov_spectrum.shape = {static_cast<std::size_t>(dim + 3 + off_diag_pairs.size())};
        cov_spectrum.computational_cost = 1.0;
        for (Eigen::Index i = 0; i < dim; ++i) {
            cov_spectrum.values.push_back(
                FeatureDescriptor::normalize_value(descending_eigs(i), FeatureNormalizationStrategy::SignedUnitInterval));
        }
        for (Eigen::Index i = 0; i < 3; ++i) {
            // Spectral gap between consecutive descending eigenvalues.
            const double gap = (i + 1 < dim) ? (descending_eigs(i) - descending_eigs(i + 1)) : 0.0;
            cov_spectrum.values.push_back(
                FeatureDescriptor::normalize_value(gap, FeatureNormalizationStrategy::SignedUnitInterval));
        }
        for (double pair_val : off_diag_pairs) {
            cov_spectrum.values.push_back(
                FeatureDescriptor::normalize_value(pair_val, FeatureNormalizationStrategy::SignedUnitInterval));
        }
        out.push_back(cov_spectrum);

        // Feature 3: Layer-wise singular value spectrum, numerical rank, effective rank, and spectral concentration.
        FeatureDescriptor layer_spectra;
        layer_spectra.id = "spectral.layer_singular_spectra";
        layer_spectra.domain = AnalysisDomain::Spectral;
        layer_spectra.type = FeatureValueType::Vector;
        layer_spectra.normalization = FeatureNormalizationStrategy::UnitInterval;
        layer_spectra.required_tensors = {"layer_activations"};
        layer_spectra.shape = {12};
        layer_spectra.computational_cost = 1.5;

        for (std::size_t l = 0; l < 2; ++l) {
            if (l < telemetry.layer_activations.size()) {
                // Activation matrix of hidden layer l.
                const Eigen::MatrixXd& act = telemetry.layer_activations[l];
                // Centered activation matrix of hidden layer l.
                const Eigen::MatrixXd centered = act.rowwise() - act.colwise().mean();
                // Singular values of the centered activation matrix.
                const Eigen::VectorXd sv = Eigen::BDCSVD<Eigen::MatrixXd>(centered).singularValues();
                // Maximum possible rank for this layer.
                const double max_rank = static_cast<double>(std::max<Eigen::Index>(1, std::min(act.rows(), act.cols())));
                // Count of singular values above numerical_rank_tolerance.
                const std::size_t num_rank =
                    static_cast<std::size_t>((sv.array() > config.numerical_rank_tolerance).count());
                // Count of singular values above spectral_rank_tolerance * sv(0).
                const std::size_t eff_rank =
                    (sv.size() > 0 && sv(0) > 1e-12)
                        ? static_cast<std::size_t>((sv.array() >= config.spectral_rank_tolerance * sv(0)).count())
                        : 0;
                // Sum of singular values for spectral concentration.
                const double sv_sum = std::max(1e-12, sv.sum());
                // Leading singular value share.
                const double s0_share = sv.size() > 0 ? sv(0) / sv_sum : 0.0;
                // Second singular value share.
                const double s1_share = sv.size() > 1 ? sv(1) / sv_sum : 0.0;
                // Third singular value ratio relative to numerical_rank_tolerance (critical for narrow rank <= 2).
                const double s2_excess = sv.size() > 2 ? std::tanh((sv(2) - config.numerical_rank_tolerance) * 50.0) : -1.0;

                layer_spectra.values.push_back(static_cast<double>(num_rank) / max_rank);
                layer_spectra.values.push_back(static_cast<double>(eff_rank) / max_rank);
                layer_spectra.values.push_back(num_rank <= config.narrow_rank_limit ? 1.0 : 0.0);
                layer_spectra.values.push_back(s0_share);
                layer_spectra.values.push_back(s1_share);
                layer_spectra.values.push_back(0.5 * (s2_excess + 1.0));
            } else {
                // Missing second layer padded with zeros.
                for (int k = 0; k < 6; ++k) {
                    layer_spectra.values.push_back(0.0);
                }
            }
        }
        out.push_back(layer_spectra);

        return out;
    }
};

}  // namespace forward_analysis
