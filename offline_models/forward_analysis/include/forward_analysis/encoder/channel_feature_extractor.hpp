#pragma once

#include "forward_analysis/analysis/observation_extractor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/encoder/channel_role.hpp"
#include "forward_analysis/math/spectral_ops.hpp"
#include "forward_analysis/math/statistics_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace forward_analysis {

// Turns telemetry of any shape into a variable-length set of fixed-width channel tokens.
// Row order: inputs, hidden units layer by layer, then for each output its target, prediction
// and residual. Every feature is scale-bounded so networks of different sizes share one space.
struct ChannelFeatureExtractor {
    // Width of a channel token.
    static constexpr std::size_t kFeatureDim = 25;

    static Eigen::MatrixXd extract(const ForwardTelemetryTensor& telemetry,
                                   const DistillerConfig& config = DistillerConfig{}) {
        if (telemetry.num_samples() == 0 || telemetry.layer_activations.empty()) {
            throw std::invalid_argument("ChannelFeatureExtractor requires samples and at least one hidden layer");
        }
        const TelemetryShape shape = telemetry.shape();
        const Eigen::MatrixXd& first = telemetry.layer_activations.front();
        const Eigen::MatrixXd scores = SpectralOps::principal_scores(first, 2);
        const Eigen::VectorXd closed = closed_indicator(first, config);
        const double ray = ObservationExtractor::observe_layer(first, config).lies_on_ray ? 1.0 : 0.0;
        const double log_n = std::log(static_cast<double>(shape.num_samples)) / std::log(16.0);
        const double layer_span = static_cast<double>(shape.hidden_widths.size() + 1);

        Eigen::MatrixXd features =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(shape.channel_count()), kFeatureDim);
        Eigen::Index row = 0;
        const auto emit = [&](const Eigen::VectorXd& column, ChannelRole role, double depth, std::size_t index,
                              std::size_t group_width, const Eigen::VectorXd* paired) {
            fill_row(features.row(row++), column, role, depth, index, group_width, paired, scores, closed, ray,
                     log_n);
        };

        const Eigen::Index in_dim = telemetry.inputs.cols();
        for (Eigen::Index c = 0; c < in_dim; ++c) {
            emit(telemetry.inputs.col(c), ChannelRole::Input, 0.0, static_cast<std::size_t>(c),
                 static_cast<std::size_t>(in_dim), nullptr);
        }
        for (std::size_t l = 0; l < telemetry.layer_activations.size(); ++l) {
            const Eigen::MatrixXd& layer = telemetry.layer_activations[l];
            const double depth = static_cast<double>(l + 1) / layer_span;
            for (Eigen::Index u = 0; u < layer.cols(); ++u) {
                emit(layer.col(u), ChannelRole::Hidden, depth, static_cast<std::size_t>(u),
                     static_cast<std::size_t>(layer.cols()), nullptr);
            }
        }
        const Eigen::Index out_dim = telemetry.targets.cols();
        for (Eigen::Index k = 0; k < out_dim; ++k) {
            const Eigen::VectorXd t = telemetry.targets.col(k);
            const Eigen::VectorXd p = telemetry.predictions.col(k);
            const Eigen::VectorXd r = telemetry.residuals.col(k);
            const std::size_t idx = static_cast<std::size_t>(k);
            const std::size_t w = static_cast<std::size_t>(out_dim);
            emit(t, ChannelRole::Target, 1.0, idx, w, &p);
            emit(p, ChannelRole::Prediction, 1.0, idx, w, &t);
            emit(r, ChannelRole::Residual, 1.0, idx, w, &t);
        }
        return features;
    }

private:
    static Eigen::VectorXd closed_indicator(const Eigen::MatrixXd& first, const DistillerConfig& config) {
        Eigen::VectorXd closed(first.rows());
        for (Eigen::Index i = 0; i < first.rows(); ++i) {
            closed(i) = first.row(i).cwiseAbs().maxCoeff() <= config.dead_activation_threshold ? 1.0 : 0.0;
        }
        return closed;
    }

    template <typename Row>
    static void fill_row(Row out,
                         const Eigen::VectorXd& v,
                         ChannelRole role,
                         double depth,
                         std::size_t index,
                         std::size_t group_width,
                         const Eigen::VectorXd* paired,
                         const Eigen::MatrixXd& scores,
                         const Eigen::VectorXd& closed,
                         double ray,
                         double log_n) {
        constexpr double kTwoPi = 6.283185307179586;
        const double n = static_cast<double>(v.size());
        const double std_v = StatisticsOps::standard_deviation(v);

        out(static_cast<Eigen::Index>(role)) = 1.0;
        out(5) = depth;
        const double phase = kTwoPi * static_cast<double>(index) / static_cast<double>(group_width);
        out(6) = std::sin(phase);
        out(7) = std::cos(phase);
        out(8) = static_cast<double>(index) / static_cast<double>(group_width);
        out(9) = std::tanh(v.mean());
        out(10) = std::tanh(std_v);
        out(11) = std::tanh(v.minCoeff());
        out(12) = std::tanh(v.maxCoeff());
        out(13) = (v.array().abs() <= 1e-9).cast<double>().sum() / n;
        out(14) = (v.array() > 1e-9).cast<double>().sum() / n;
        out(15) = (v.array() < -1e-9).cast<double>().sum() / n;
        out(16) = StatisticsOps::pearson(v, scores.col(0));
        out(17) = StatisticsOps::pearson(v, scores.col(1));
        out(18) = StatisticsOps::pearson(v, closed);
        if (paired != nullptr) {
            out(19) = StatisticsOps::pearson(v, *paired);
            if (role == ChannelRole::Residual) {
                out(20) = std::tanh(std_v / std::max(1e-9, StatisticsOps::standard_deviation(*paired)));
            } else {
                out(20) = (v.array() * paired->array() > 0.0).cast<double>().sum() / n;
            }
        }
        out(21) = std::log1p(static_cast<double>(group_width)) / std::log(32.0);
        out(22) = log_n;
        out(23) = ray;
        out(24) = std_v <= 1e-9 ? 1.0 : 0.0;
    }
};

}  // namespace forward_analysis
