#pragma once

#include "forward_analysis/math/spectral_profile.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace forward_analysis {

struct SpectralOps {
    static Eigen::MatrixXd covariance(const Eigen::MatrixXd& samples) {
        if (samples.size() == 0) {
            return Eigen::MatrixXd();
        }
        const Eigen::RowVectorXd mean = samples.colwise().mean();
        const Eigen::MatrixXd centered = samples.rowwise() - mean;
        const double denom =
            samples.rows() > 1 ? static_cast<double>(samples.rows() - 1) : 1.0;
        return (centered.transpose() * centered) / denom;
    }

    static SpectralProfile analyze_activation_manifold(const Eigen::MatrixXd& activations,
                                                       std::size_t max_components = 4) {
        SpectralProfile profile;
        if (activations.size() == 0) {
            return profile;
        }

        if (activations.cols() > 64 && activations.cols() > 2 * activations.rows()) {
            const Eigen::RowVectorXd mean = activations.colwise().mean();
            const Eigen::MatrixXd centered = activations.rowwise() - mean;
            const double denom =
                activations.rows() > 1 ? static_cast<double>(activations.rows() - 1) : 1.0;
            const Eigen::MatrixXd gram = (centered * centered.transpose()) / denom;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(gram);
            const Eigen::VectorXd evals = eig.eigenvalues().reverse().cwiseMax(0.0);
            const Eigen::MatrixXd evecs = eig.eigenvectors().rowwise().reverse();
            const Eigen::Index k = std::min<Eigen::Index>(
                static_cast<Eigen::Index>(max_components), evals.size());
            profile.singular_values = evals.head(k);
            profile.principal_axes = Eigen::MatrixXd::Zero(activations.cols(), k);
            for (Eigen::Index c = 0; c < k; ++c) {
                const double sigma = std::sqrt(evals(c) * denom);
                if (sigma > 1e-12) {
                    profile.principal_axes.col(c) = (centered.transpose() * evecs.col(c)) / sigma;
                }
                Eigen::Index max_row = 0;
                profile.principal_axes.col(c).cwiseAbs().maxCoeff(&max_row);
                if (profile.principal_axes(max_row, c) < 0.0) {
                    profile.principal_axes.col(c) *= -1.0;
                }
            }
            if (k > 0) {
                profile.dominant_eigen_signs = profile.principal_axes.col(0).unaryExpr(
                    [](double v) { return v >= -1e-9 ? 1.0 : -1.0; });
            }
            const Eigen::VectorXd& s = profile.singular_values;
            if (s.size() >= 2 && s(0) > 1e-9) {
                const double ratio = s(1) / s(0);
                profile.eccentricity = std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
            } else if (s.size() == 1 && s(0) > 1e-9) {
                profile.eccentricity = 1.0;
            }
            return profile;
        }

        const Eigen::MatrixXd cov = covariance(activations);
        Eigen::BDCSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

        const Eigen::Index k = std::min<Eigen::Index>(
            static_cast<Eigen::Index>(max_components), svd.singularValues().size());
        profile.singular_values = svd.singularValues().head(k);
        profile.principal_axes = svd.matrixV().leftCols(k);

        // SVD axes are sign-ambiguous; pin each so its largest-magnitude entry is positive.
        for (Eigen::Index c = 0; c < k; ++c) {
            Eigen::Index max_row = 0;
            profile.principal_axes.col(c).cwiseAbs().maxCoeff(&max_row);
            if (profile.principal_axes(max_row, c) < 0.0) {
                profile.principal_axes.col(c) *= -1.0;
            }
        }

        // Entries of zero-variance (dead) units are zero up to round-off and count as +1.
        if (k > 0) {
            profile.dominant_eigen_signs = profile.principal_axes.col(0).unaryExpr(
                [](double v) { return v >= -1e-9 ? 1.0 : -1.0; });
        }

        const Eigen::VectorXd& s = profile.singular_values;
        if (s.size() >= 2 && s(0) > 1e-9) {
            const double ratio = s(1) / s(0);
            profile.eccentricity = std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
        } else if (s.size() == 1 && s(0) > 1e-9) {
            profile.eccentricity = 1.0;
        }
        return profile;
    }

    // Per-sample scores on the leading principal axes (N x components); missing axes score 0.
    // Each axis is oriented like analyze_activation_manifold: largest-magnitude loading positive.
    static Eigen::MatrixXd principal_scores(const Eigen::MatrixXd& activations, std::size_t components) {
        const Eigen::Index k = static_cast<Eigen::Index>(components);
        Eigen::MatrixXd scores = Eigen::MatrixXd::Zero(activations.rows(), k);
        if (activations.rows() < 2 || activations.cols() == 0) {
            return scores;
        }
        const Eigen::MatrixXd centered = activations.rowwise() - activations.colwise().mean();
        const SpectralProfile profile = analyze_activation_manifold(activations, components);
        const Eigen::VectorXd& s = profile.singular_values;
        for (Eigen::Index c = 0; c < std::min<Eigen::Index>(k, s.size()); ++c) {
            if (s(c) > 1e-12 * std::max(1.0, s(0))) {
                scores.col(c) = centered * profile.principal_axes.col(c);
            }
        }
        return scores;
    }

    // Singular values of the raw (uncentered) activation matrix, largest first.
    static Eigen::VectorXd uncentered_singular_values(const Eigen::MatrixXd& activations) {
        if (activations.size() == 0) {
            return Eigen::VectorXd();
        }
        if (activations.cols() > 64 && activations.cols() > 2 * activations.rows()) {
            const Eigen::MatrixXd gram = activations * activations.transpose();
            const Eigen::VectorXd evals =
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(gram, Eigen::EigenvaluesOnly)
                    .eigenvalues()
                    .reverse()
                    .cwiseMax(0.0);
            return evals.cwiseSqrt();
        }
        return Eigen::BDCSVD<Eigen::MatrixXd>(activations).singularValues();
    }
};

}  // namespace forward_analysis
