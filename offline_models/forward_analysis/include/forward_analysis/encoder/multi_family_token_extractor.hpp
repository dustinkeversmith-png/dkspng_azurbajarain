#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/config/feature_ablation_level.hpp"
#include "forward_analysis/encoder/channel_feature_extractor.hpp"
#include "forward_analysis/encoder/multi_family_token_batch.hpp"
#include "forward_analysis/features/feature_extractor_registry.hpp"
#include "forward_analysis/math/spectral_ops.hpp"
#include "forward_analysis/math/statistics_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace forward_analysis {

// Extracts the five token families (Channel, Sample, Relationship, Layer, Global) according to FeatureAblationLevel.
struct MultiFamilyTokenExtractor {
    // Width of an expanded channel token (25 baseline features + 13 structural/spectral/functional features).
    static constexpr std::size_t kExpandedChannelDim = 38;
    // Width of each sample token.
    static constexpr std::size_t kSampleTokenDim = 16;
    // Width of each relationship token.
    static constexpr std::size_t kRelationshipTokenDim = 16;
    // Width of each layer token.
    static constexpr std::size_t kLayerTokenDim = 16;
    // Width of each global domain token.
    static constexpr std::size_t kGlobalTokenDim = 16;
    // Maximum number of sample tokens emitted (bounded pooling for large batches).
    static constexpr std::size_t kMaxSampleTokens = 16;

    static MultiFamilyTokenBatch extract(const ForwardTelemetryTensor& telemetry,
                                         const DistillerConfig& config,
                                         const FeatureExtractorRegistry& registry) {
        // Output batch of multi-family token matrices.
        MultiFamilyTokenBatch batch;
        // Baseline 25-feature channel matrix (C x 25).
        const Eigen::MatrixXd base_channels = ChannelFeatureExtractor::extract(telemetry, config);
        if (config.feature_ablation_level == FeatureAblationLevel::kBaseline25) {
            batch.channel_tokens = base_channels;
            return batch;
        }

        batch.channel_tokens = build_expanded_channels(telemetry, config, base_channels);
        if (config.feature_ablation_level == FeatureAblationLevel::kChannelPlusSample ||
            config.feature_ablation_level == FeatureAblationLevel::kCompleteArchitecture) {
            batch.sample_tokens = build_sample_tokens(telemetry, config);
        }
        if (config.feature_ablation_level == FeatureAblationLevel::kChannelPlusRelationship ||
            config.feature_ablation_level == FeatureAblationLevel::kCompleteArchitecture) {
            batch.relationship_tokens = build_relationship_tokens(telemetry, config);
        }
        if (config.feature_ablation_level == FeatureAblationLevel::kCompleteArchitecture) {
            batch.layer_tokens = build_layer_tokens(telemetry, config, batch.layer_valid_mask);
            batch.global_tokens = build_global_tokens(telemetry, config, registry);
        }
        return batch;
    }

private:
    static Eigen::MatrixXd build_expanded_channels(const ForwardTelemetryTensor& telemetry,
                                                   const DistillerConfig& config,
                                                   const Eigen::MatrixXd& base_channels) {
        // Total channel count.
        const Eigen::Index c_total = base_channels.rows();
        // Expanded channel matrix (C x 38).
        Eigen::MatrixXd expanded = Eigen::MatrixXd::Zero(c_total, static_cast<Eigen::Index>(kExpandedChannelDim));
        expanded.leftCols(static_cast<Eigen::Index>(ChannelFeatureExtractor::kFeatureDim)) = base_channels;

        // Number of hidden layers.
        const std::size_t num_layers = telemetry.layer_activations.size();
        // Indicator for 1-layer topology.
        const double is_one_layer = num_layers == 1 ? 1.0 : 0.0;
        // Indicator for 2-or-more-layer topology.
        const double is_two_layer = num_layers >= 2 ? 1.0 : 0.0;
        // Final hidden layer activations.
        const Eigen::MatrixXd& last_hidden = telemetry.layer_activations.back();
        // Number of input coordinates.
        const Eigen::Index d_in = telemetry.inputs.cols();

        // Zero-diagonal 4x4 input covariance matrix and its eigenvectors/eigenvalues.
        const Eigen::Index used_eig = std::min<Eigen::Index>(4, d_in);
        // Normalized 4x4 zero-diagonal covariance.
        Eigen::MatrixXd cov4 = Eigen::MatrixXd::Zero(4, 4);
        // Descending eigenvalues of cov4.
        Eigen::VectorXd eig_vals = Eigen::VectorXd::Zero(4);
        if (used_eig > 0 && telemetry.inputs.rows() > 0) {
            // Centered leading inputs.
            const Eigen::MatrixXd centered_in =
                telemetry.inputs.leftCols(used_eig).rowwise() - telemetry.inputs.leftCols(used_eig).colwise().mean();
            // Sample covariance of leading inputs.
            Eigen::MatrixXd raw_cov =
                centered_in.transpose() * centered_in / std::max(1.0, static_cast<double>(telemetry.inputs.rows() - 1));
            raw_cov.diagonal().setZero();
            // Scale factor for normalizing covariance entries.
            const double scale = std::max(1e-9, raw_cov.norm());
            cov4.topLeftCorner(used_eig, used_eig) = raw_cov / scale;
            eig_vals.head(used_eig) =
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(raw_cov).eigenvalues().reverse() / scale;
        }

        // Per-layer numerical rank and narrow flags.
        std::vector<double> layer_rank_norm(num_layers, 0.0);
        // Per-layer narrow indicator.
        std::vector<double> layer_narrow(num_layers, 0.0);
        // Per-layer nonpositive sample share.
        std::vector<double> layer_nonpos(num_layers, 0.0);
        // True when all layers are narrow.
        bool network_narrow = true;
        for (std::size_t l = 0; l < num_layers; ++l) {
            // Layer activation matrix.
            const Eigen::MatrixXd& act = telemetry.layer_activations[l];
            // Centered layer activation matrix.
            const Eigen::MatrixXd centered = act.rowwise() - act.colwise().mean();
            // Centered singular values.
            const Eigen::VectorXd sv = Eigen::BDCSVD<Eigen::MatrixXd>(centered).singularValues();
            // Numerical rank of layer l.
            const std::size_t n_rank =
                static_cast<std::size_t>((sv.array() > config.numerical_rank_tolerance).count());
            layer_rank_norm[l] = std::min(1.0, static_cast<double>(n_rank) / 8.0);
            layer_narrow[l] = n_rank <= config.narrow_rank_limit ? 1.0 : 0.0;
            network_narrow = network_narrow && (n_rank <= config.narrow_rank_limit);
            // Nonpositive observation count in layer l.
            std::size_t nonpos = 0;
            for (Eigen::Index i = 0; i < act.rows(); ++i) {
                if (act.row(i).maxCoeff() <= 0.0) {
                    ++nonpos;
                }
            }
            layer_nonpos[l] = static_cast<double>(nonpos) / static_cast<double>(std::max<Eigen::Index>(1, act.rows()));
        }

        // Input association scores with final hidden layer.
        std::vector<double> assoc_scores(static_cast<std::size_t>(d_in), 0.0);
        for (Eigen::Index c = 0; c < d_in; ++c) {
            // Maximum absolute Pearson correlation with final hidden units.
            double best = 0.0;
            for (Eigen::Index k = 0; k < last_hidden.cols(); ++k) {
                best = std::max(best, std::abs(StatisticsOps::pearson(telemetry.inputs.col(c), last_hidden.col(k))));
            }
            assoc_scores[static_cast<std::size_t>(c)] = best;
        }
        // Maximum association across all inputs.
        const double max_assoc =
            assoc_scores.empty() ? 0.0 : *std::max_element(assoc_scores.begin(), assoc_scores.end());

        // Current row cursor in `expanded`.
        Eigen::Index row = 0;
        for (Eigen::Index c = 0; c < d_in; ++c) {
            expanded(row, 25) = is_one_layer;
            expanded(row, 26) = is_two_layer;
            expanded(row, 27) = network_narrow ? 1.0 : 0.0;
            expanded(row, 28) = assoc_scores[static_cast<std::size_t>(c)];
            expanded(row, 29) = max_assoc > 1e-9 ? assoc_scores[static_cast<std::size_t>(c)] / max_assoc : 0.0;
            if (c < 4) {
                expanded(row, 30) = std::tanh(cov4(c, 0));
                expanded(row, 31) = std::tanh(cov4(c, 1));
                expanded(row, 32) = std::tanh(cov4(c, 2));
                expanded(row, 33) = std::tanh(cov4(c, 3));
                expanded(row, 34) = std::tanh(eig_vals(c));
                expanded(row, 35) = eig_vals(c) >= 0.0 ? 1.0 : -1.0;
            }
            ++row;
        }
        for (std::size_t l = 0; l < num_layers; ++l) {
            // Layer width.
            const Eigen::Index w = telemetry.layer_activations[l].cols();
            for (Eigen::Index u = 0; u < w; ++u) {
                expanded(row, 25) = is_one_layer;
                expanded(row, 26) = is_two_layer;
                expanded(row, 27) = network_narrow ? 1.0 : 0.0;
                expanded(row, 28) = l == 0 ? 1.0 : 0.0;
                expanded(row, 29) = l == 1 ? 1.0 : 0.0;
                expanded(row, 30) = layer_rank_norm[l];
                expanded(row, 31) = layer_narrow[l];
                expanded(row, 32) = layer_nonpos[l];
                ++row;
            }
        }
        // Number of output coordinates.
        const Eigen::Index d_out = telemetry.targets.cols();
        for (Eigen::Index k = 0; k < d_out; ++k) {
            // Target column k.
            const Eigen::VectorXd t = telemetry.targets.col(k);
            // Prediction column k.
            const Eigen::VectorXd p = telemetry.predictions.col(k);
            // Target range.
            const double t_range = t.size() > 0 ? (t.maxCoeff() - t.minCoeff()) : 0.0;
            // Prediction-to-target range ratio.
            const double ratio = t_range > 1e-12 ? (p.maxCoeff() - p.minCoeff()) / t_range : 1.0;
            // Smaller-range indicator.
            const double is_smaller = ratio < config.range_ratio_low ? 1.0 : 0.0;
            // Larger-range indicator.
            const double is_larger = ratio > config.range_ratio_high ? 1.0 : 0.0;
            // Comparable-range indicator.
            const double is_comparable = (is_smaller == 0.0 && is_larger == 0.0) ? 1.0 : 0.0;
            // Sign agreement share on column k.
            double sign_share = 0.0;
            for (Eigen::Index i = 0; i < t.size(); ++i) {
                if (((t(i) > 0.0) - (t(i) < 0.0)) == ((p(i) > 0.0) - (p(i) < 0.0))) {
                    sign_share += 1.0;
                }
            }
            sign_share /= static_cast<double>(std::max<Eigen::Index>(1, t.size()));

            for (int trip = 0; trip < 3; ++trip) {
                expanded(row, 25) = is_one_layer;
                expanded(row, 26) = is_two_layer;
                expanded(row, 27) = network_narrow ? 1.0 : 0.0;
                expanded(row, 33) = is_smaller;
                expanded(row, 34) = is_larger;
                expanded(row, 35) = is_comparable;
                expanded(row, 36) = sign_share;
                expanded(row, 37) = std::tanh(ratio - 1.0);
                ++row;
            }
        }
        return expanded;
    }

    static Eigen::MatrixXd build_sample_tokens(const ForwardTelemetryTensor& telemetry,
                                               const DistillerConfig& config) {
        // Total observations in the batch.
        const std::size_t n = telemetry.num_samples();
        // Bounded number of sample tokens to emit.
        const std::size_t s_count = std::min(n, kMaxSampleTokens);
        // Sample token matrix (s_count x 16).
        Eigen::MatrixXd tokens =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(s_count), static_cast<Eigen::Index>(kSampleTokenDim));

        for (std::size_t s = 0; s < s_count; ++s) {
            // Source observation index (strided when n > kMaxSampleTokens).
            const Eigen::Index i = static_cast<Eigen::Index>((s * n) / s_count);
            // Input row for observation i.
            const Eigen::RowVectorXd in_row = telemetry.inputs.row(i);
            // First hidden layer row for observation i.
            const Eigen::RowVectorXd l1_row = telemetry.layer_activations.front().row(i);
            // Final hidden layer row for observation i.
            const Eigen::RowVectorXd last_row = telemetry.layer_activations.back().row(i);
            // Target row for observation i.
            const Eigen::RowVectorXd t_row = telemetry.targets.row(i);
            // Prediction row for observation i.
            const Eigen::RowVectorXd p_row = telemetry.predictions.row(i);
            // Residual row for observation i.
            const Eigen::RowVectorXd r_row = telemetry.residuals.row(i);

            // Share of outputs where prediction and target signs agree on observation i.
            double row_sign_agree = 0.0;
            for (Eigen::Index k = 0; k < t_row.size(); ++k) {
                if (((t_row(k) > 0.0) - (t_row(k) < 0.0)) == ((p_row(k) > 0.0) - (p_row(k) < 0.0))) {
                    row_sign_agree += 1.0;
                }
            }
            row_sign_agree /= static_cast<double>(std::max<Eigen::Index>(1, t_row.size()));

            // Row index in `tokens`.
            const Eigen::Index r = static_cast<Eigen::Index>(s);
            tokens(r, 0) = static_cast<double>(s) / static_cast<double>(std::max<std::size_t>(1, s_count - 1));
            tokens(r, 1) = std::tanh(in_row.mean());
            tokens(r, 2) = std::tanh(in_row.norm() / std::sqrt(static_cast<double>(std::max<Eigen::Index>(1, in_row.size()))));
            tokens(r, 3) = std::tanh(in_row.minCoeff());
            tokens(r, 4) = std::tanh(in_row.maxCoeff());
            tokens(r, 5) = std::tanh(l1_row.mean());
            tokens(r, 6) = std::tanh(l1_row.maxCoeff());
            tokens(r, 7) = l1_row.maxCoeff() <= 0.0 ? 1.0 : 0.0;
            tokens(r, 8) = l1_row.cwiseAbs().maxCoeff() <= config.dead_activation_threshold ? 1.0 : 0.0;
            tokens(r, 9) = std::tanh(last_row.mean());
            tokens(r, 10) = std::tanh(last_row.maxCoeff());
            tokens(r, 11) = last_row.maxCoeff() <= 0.0 ? 1.0 : 0.0;
            tokens(r, 12) = std::tanh(p_row.mean());
            tokens(r, 13) = std::tanh(t_row.mean());
            tokens(r, 14) = std::tanh(r_row.norm() / std::sqrt(static_cast<double>(std::max<Eigen::Index>(1, r_row.size()))));
            tokens(r, 15) = row_sign_agree;
        }
        return tokens;
    }

    static Eigen::MatrixXd build_relationship_tokens(const ForwardTelemetryTensor& telemetry,
                                                     const DistillerConfig& config) {
        // 12 relationship tokens:
        //   0..5 : the 6 off-diagonal pairs of the first 4 inputs (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        //   6..9 : the 4 spectral eigenmodes of the 4x4 zero-diagonal input covariance
        //   10   : input-to-final-hidden association summary
        //   11   : cross-layer and prediction-target functional relationship summary
        Eigen::MatrixXd tokens = Eigen::MatrixXd::Zero(12, static_cast<Eigen::Index>(kRelationshipTokenDim));

        // Number of available leading inputs (up to 4).
        const Eigen::Index used = std::min<Eigen::Index>(4, telemetry.inputs.cols());
        // Zero-diagonal covariance matrix of the first 4 inputs.
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(4, 4);
        // Descending eigenvalues of `cov`.
        Eigen::VectorXd evals = Eigen::VectorXd::Zero(4);
        // Corresponding eigenvectors of `cov` (columns in descending eigenvalue order).
        Eigen::MatrixXd evecs = Eigen::MatrixXd::Identity(4, 4);
        // Frobenius norm of the unscaled zero-diagonal covariance.
        double cov_norm = 1.0;

        if (used > 0 && telemetry.inputs.rows() > 0) {
            // Centered leading input matrix.
            const Eigen::MatrixXd centered =
                telemetry.inputs.leftCols(used).rowwise() - telemetry.inputs.leftCols(used).colwise().mean();
            // Sample covariance matrix.
            Eigen::MatrixXd sub_cov =
                centered.transpose() * centered / std::max(1.0, static_cast<double>(telemetry.inputs.rows() - 1));
            sub_cov.diagonal().setZero();
            cov_norm = std::max(1e-9, sub_cov.norm());
            cov.topLeftCorner(used, used) = sub_cov / cov_norm;
            // Self-adjoint eigendecomposition of `sub_cov`.
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sub_cov);
            for (Eigen::Index k = 0; k < used; ++k) {
                // Source index in ascending solver output corresponding to descending position k.
                const Eigen::Index src = used - 1 - k;
                evals(k) = solver.eigenvalues()(src) / cov_norm;
                evecs.col(k).head(used) = solver.eigenvectors().col(src);
            }
        }

        // Token rows 0..5: pairwise input relationships.
        Eigen::Index pair_row = 0;
        for (Eigen::Index a = 0; a < 4; ++a) {
            for (Eigen::Index b = a + 1; b < 4; ++b) {
                // Pearson correlation between input a and input b.
                const double corr = (a < used && b < used)
                                        ? StatisticsOps::pearson(telemetry.inputs.col(a), telemetry.inputs.col(b))
                                        : 0.0;
                // Spearman correlation between input a and input b.
                const double rank_corr = (a < used && b < used)
                                             ? StatisticsOps::spearman(telemetry.inputs.col(a), telemetry.inputs.col(b))
                                             : 0.0;
                tokens(pair_row, 0) = static_cast<double>(a) / 3.0;
                tokens(pair_row, 1) = static_cast<double>(b) / 3.0;
                tokens(pair_row, 2) = std::tanh(cov(a, b));
                tokens(pair_row, 3) = cov(a, b) >= 0.0 ? 1.0 : -1.0;
                tokens(pair_row, 4) = std::abs(cov(a, b));
                tokens(pair_row, 5) = corr;
                tokens(pair_row, 6) = rank_corr;
                tokens(pair_row, 7) = evals(0);
                tokens(pair_row, 8) = evals(1);
                tokens(pair_row, 9) = evals(2);
                tokens(pair_row, 10) = evals(3);
                tokens(pair_row, 11) = evals(1) >= 0.0 ? 1.0 : -1.0;
                tokens(pair_row, 12) = evals(2) >= 0.0 ? 1.0 : -1.0;
                tokens(pair_row, 13) = std::tanh(evals(1) * 10.0);
                tokens(pair_row, 14) = std::tanh(evals(2) * 10.0);
                tokens(pair_row, 15) = (a < used && b < used) ? 1.0 : 0.0;
                ++pair_row;
            }
        }

        // Token rows 6..9: the 4 spectral eigenmodes of the zero-diagonal covariance.
        for (Eigen::Index k = 0; k < 4; ++k) {
            // Row index in `tokens`.
            const Eigen::Index r = 6 + k;
            // Normalized eigenvalue k.
            const double ev = evals(k);
            // Sign of eigenvalue k (+1 when >= 0, -1 when < 0).
            const double sign_k = ev >= 0.0 ? 1.0 : -1.0;
            tokens(r, 0) = static_cast<double>(k) / 3.0;
            tokens(r, 1) = std::tanh(ev);
            tokens(r, 2) = std::tanh(ev * 5.0);
            tokens(r, 3) = std::tanh(ev * 25.0);
            tokens(r, 4) = sign_k;
            tokens(r, 5) = std::abs(ev);
            tokens(r, 6) = evecs(0, k);
            tokens(r, 7) = evecs(1, k);
            tokens(r, 8) = evecs(2, k);
            tokens(r, 9) = evecs(3, k);
            tokens(r, 10) = k > 0 ? std::tanh(evals(k - 1) - ev) : 0.0;
            tokens(r, 11) = k + 1 < 4 ? std::tanh(ev - evals(k + 1)) : 0.0;
            tokens(r, 12) = evals(1) >= 0.0 ? 1.0 : -1.0;
            tokens(r, 13) = evals(2) >= 0.0 ? 1.0 : -1.0;
            tokens(r, 14) = std::tanh(cov_norm);
            tokens(r, 15) = k < used ? 1.0 : 0.0;
        }

        // Token row 10: input-to-final-hidden association relationship token.
        const Eigen::MatrixXd& last_hidden = telemetry.layer_activations.back();
        // Sorted association strengths.
        std::vector<double> assoc;
        for (Eigen::Index c = 0; c < telemetry.inputs.cols(); ++c) {
            // Best absolute Pearson correlation for input c.
            double best = 0.0;
            for (Eigen::Index u = 0; u < last_hidden.cols(); ++u) {
                best = std::max(best, std::abs(StatisticsOps::pearson(telemetry.inputs.col(c), last_hidden.col(u))));
            }
            assoc.push_back(best);
        }
        std::sort(assoc.begin(), assoc.end(), std::greater<double>());
        tokens(10, 0) = assoc.empty() ? 0.0 : assoc[0];
        tokens(10, 1) = assoc.size() >= 2 ? assoc[1] : 0.0;
        tokens(10, 2) = assoc.size() >= 2 ? assoc[assoc.size() - 2] : 0.0;
        tokens(10, 3) = assoc.empty() ? 0.0 : assoc.back();
        tokens(10, 4) = (!assoc.empty() && assoc[0] > config.dead_activation_threshold) ? 1.0 : 0.0;

        // Token row 11: functional output-target relationship token.
        const Eigen::Index d_out = telemetry.targets.cols();
        // Reciprocal of output dimension.
        const double inv_out = 1.0 / static_cast<double>(std::max<Eigen::Index>(1, d_out));
        // Smaller, larger, comparable counts.
        std::size_t smaller = 0;
        // Larger-range output count.
        std::size_t larger = 0;
        // Comparable-range output count.
        std::size_t comparable = 0;
        // Sign agreement entry count.
        std::size_t sign_agree = 0;
        for (Eigen::Index k = 0; k < d_out; ++k) {
            // Target column k.
            const Eigen::VectorXd t = telemetry.targets.col(k);
            // Prediction column k.
            const Eigen::VectorXd p = telemetry.predictions.col(k);
            // Target range.
            const double tr = t.size() > 0 ? (t.maxCoeff() - t.minCoeff()) : 0.0;
            // Prediction-to-target range ratio.
            const double ratio = tr > 1e-12 ? (p.maxCoeff() - p.minCoeff()) / tr : 1.0;
            if (ratio < config.range_ratio_low) {
                ++smaller;
            } else if (ratio > config.range_ratio_high) {
                ++larger;
            } else {
                ++comparable;
            }
            for (Eigen::Index i = 0; i < t.size(); ++i) {
                if (((t(i) > 0.0) - (t(i) < 0.0)) == ((p(i) > 0.0) - (p(i) < 0.0))) {
                    ++sign_agree;
                }
            }
        }
        tokens(11, 0) = static_cast<double>(smaller) * inv_out;
        tokens(11, 1) = static_cast<double>(larger) * inv_out;
        tokens(11, 2) = static_cast<double>(comparable) * inv_out;
        tokens(11, 3) = static_cast<double>(sign_agree) /
                        static_cast<double>(std::max<Eigen::Index>(1, telemetry.targets.size()));
        tokens(11, 4) = telemetry.layer_activations.size() == 1 ? 1.0 : 0.0;
        tokens(11, 5) = telemetry.layer_activations.size() >= 2 ? 1.0 : 0.0;

        return tokens;
    }

    static Eigen::MatrixXd build_layer_tokens(const ForwardTelemetryTensor& telemetry,
                                              const DistillerConfig& config,
                                              Eigen::VectorXd& valid_mask) {
        // Two structural layer token slots (Layer 1 and Layer 2).
        Eigen::MatrixXd tokens = Eigen::MatrixXd::Zero(2, static_cast<Eigen::Index>(kLayerTokenDim));
        valid_mask = Eigen::VectorXd::Zero(2);
        // Total hidden layers present.
        const std::size_t num_layers = telemetry.layer_activations.size();

        for (std::size_t l = 0; l < std::min<std::size_t>(2, num_layers); ++l) {
            valid_mask(static_cast<Eigen::Index>(l)) = 1.0;
            // Activation matrix of layer l.
            const Eigen::MatrixXd& act = telemetry.layer_activations[l];
            // Centered activation matrix of layer l.
            const Eigen::MatrixXd centered = act.rowwise() - act.colwise().mean();
            // Singular values of centered activations.
            const Eigen::VectorXd sv = Eigen::BDCSVD<Eigen::MatrixXd>(centered).singularValues();
            // Numerical rank (> numerical_rank_tolerance).
            const std::size_t num_rank =
                static_cast<std::size_t>((sv.array() > config.numerical_rank_tolerance).count());
            // Effective rank (>= spectral_rank_tolerance * sv(0)).
            const std::size_t eff_rank =
                (sv.size() > 0 && sv(0) > 1e-12)
                    ? static_cast<std::size_t>((sv.array() >= config.spectral_rank_tolerance * sv(0)).count())
                    : 0;
            // Dead and always-active unit counts.
            std::size_t dead = 0;
            // Always-active unit count.
            std::size_t active = 0;
            // Absolute activation magnitudes.
            const Eigen::ArrayXXd mag = act.array().abs();
            for (Eigen::Index u = 0; u < act.cols(); ++u) {
                if (mag.col(u).maxCoeff() <= config.dead_activation_threshold) {
                    ++dead;
                } else if (mag.col(u).minCoeff() > config.dead_activation_threshold) {
                    ++active;
                }
            }
            // Nonpositive observation count.
            std::size_t nonpos = 0;
            for (Eigen::Index i = 0; i < act.rows(); ++i) {
                if (act.row(i).maxCoeff() <= 0.0) {
                    ++nonpos;
                }
            }
            // Width and sample count as doubles.
            const double w = static_cast<double>(std::max<Eigen::Index>(1, act.cols()));
            // Sample count as double.
            const double n = static_cast<double>(std::max<Eigen::Index>(1, act.rows()));
            // Sum of singular values.
            const double sv_sum = std::max(1e-12, sv.sum());
            // Row index in `tokens`.
            const Eigen::Index r = static_cast<Eigen::Index>(l);

            tokens(r, 0) = l == 0 ? 1.0 : 0.0;
            tokens(r, 1) = l == 1 ? 1.0 : 0.0;
            tokens(r, 2) = num_layers == 1 ? 1.0 : 0.0;
            tokens(r, 3) = num_layers >= 2 ? 1.0 : 0.0;
            tokens(r, 4) = std::min(1.0, w / 48.0);
            tokens(r, 5) = static_cast<double>(dead) / w;
            tokens(r, 6) = static_cast<double>(active) / w;
            tokens(r, 7) = static_cast<double>(nonpos) / n;
            tokens(r, 8) = std::min(1.0, static_cast<double>(num_rank) / 16.0);
            tokens(r, 9) = std::min(1.0, static_cast<double>(eff_rank) / 16.0);
            tokens(r, 10) = num_rank <= config.narrow_rank_limit ? 1.0 : 0.0;
            tokens(r, 11) = sv.size() > 0 ? sv(0) / sv_sum : 0.0;
            tokens(r, 12) = sv.size() > 1 ? sv(1) / sv_sum : 0.0;
            tokens(r, 13) = sv.size() > 2 ? sv(2) / sv_sum : 0.0;
            tokens(r, 14) = sv.size() > 2 ? std::tanh((sv(2) - config.numerical_rank_tolerance) * 50.0) : -1.0;
            tokens(r, 15) = 1.0;
        }
        return tokens;
    }

    static Eigen::MatrixXd build_global_tokens(const ForwardTelemetryTensor& telemetry,
                                               const DistillerConfig& config,
                                               const FeatureExtractorRegistry& registry) {
        // One global token per mathematical AnalysisDomain (6 rows x 16 cols).
        Eigen::MatrixXd tokens = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(AnalysisDomainTraits::kDomainCount),
            static_cast<Eigen::Index>(kGlobalTokenDim));

        for (AnalysisDomain dom : AnalysisDomainTraits::all()) {
            // Row index corresponding to this mathematical domain.
            const Eigen::Index r = static_cast<Eigen::Index>(AnalysisDomainTraits::index(dom));
            // Domain one-hot marker in the first 6 columns.
            tokens(r, r) = 1.0;
            // Extracted descriptors for this domain.
            const std::vector<FeatureDescriptor> descs = registry.extract_domain(dom, telemetry, config);
            // Column cursor for packing domain values into columns 6..15.
            Eigen::Index col = 6;
            for (const FeatureDescriptor& d : descs) {
                for (double v : d.values) {
                    if (col < static_cast<Eigen::Index>(kGlobalTokenDim)) {
                        tokens(r, col++) = d.valid ? v : d.missing_fill_value;
                    }
                }
            }
        }
        return tokens;
    }
};

}  // namespace forward_analysis
