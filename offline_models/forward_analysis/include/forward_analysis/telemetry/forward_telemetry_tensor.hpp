#pragma once

#include "forward_analysis/telemetry/telemetry_shape.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace forward_analysis {

// Per-sample, per-unit 0/1 flags (N x units), stored compactly as bytes.
using MaskMatrix = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

struct ForwardTelemetryTensor {
    // Inputs fed to the observed network (N samples x D_in).
    Eigen::MatrixXd inputs;
    // Desired outputs for each sample (N x D_out).
    Eigen::MatrixXd targets;
    // Outputs the observed network actually produced (N x D_out).
    Eigen::MatrixXd predictions;
    // predictions - targets for each sample (N x D_out).
    Eigen::MatrixXd residuals;
    // Post-activation values of every hidden layer, in forward order; layer l is N x width_l.
    std::vector<Eigen::MatrixXd> layer_activations;
    // Per layer: 1 where |activation| <= dead_activation_threshold (unit not firing on that sample).
    std::vector<MaskMatrix> dead_masks;

    std::size_t num_samples() const noexcept {
        return static_cast<std::size_t>(inputs.rows());
    }

    std::size_t num_layers() const noexcept {
        return layer_activations.size();
    }

    std::size_t input_dim() const noexcept {
        return static_cast<std::size_t>(inputs.cols());
    }

    std::size_t output_dim() const noexcept {
        return static_cast<std::size_t>(targets.cols());
    }

    TelemetryShape shape() const {
        TelemetryShape out;
        out.num_samples = num_samples();
        out.input_dim = input_dim();
        out.output_dim = output_dim();
        for (const Eigen::MatrixXd& layer : layer_activations) {
            out.hidden_widths.push_back(static_cast<std::size_t>(layer.cols()));
        }
        return out;
    }
};

}  // namespace forward_analysis
