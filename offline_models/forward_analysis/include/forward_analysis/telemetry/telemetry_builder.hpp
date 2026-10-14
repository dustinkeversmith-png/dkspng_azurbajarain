#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/math/matrix_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace forward_analysis {

struct TelemetryBuilder {
    static ForwardTelemetryTensor build(Eigen::MatrixXd inputs,
                                        Eigen::MatrixXd targets,
                                        Eigen::MatrixXd predictions,
                                        std::vector<Eigen::MatrixXd> layer_activations,
                                        const DistillerConfig& config = DistillerConfig{}) {
        const Eigen::Index n = inputs.rows();
        if (n == 0 || inputs.cols() == 0) {
            throw std::invalid_argument("Telemetry inputs must be a non-empty N x D_in matrix");
        }
        require_rows(targets, n, "targets");
        require_rows(predictions, n, "predictions");
        if (targets.cols() == 0 || predictions.cols() != targets.cols()) {
            throw std::invalid_argument("Telemetry predictions and targets must share a non-zero width");
        }
        if (layer_activations.empty()) {
            throw std::invalid_argument("Telemetry requires at least one hidden layer activation matrix");
        }
        for (std::size_t l = 0; l < layer_activations.size(); ++l) {
            require_rows(layer_activations[l], n, "layer_activations[" + std::to_string(l) + "]");
            if (layer_activations[l].cols() == 0) {
                throw std::invalid_argument("Telemetry layer_activations[" + std::to_string(l) + "] has no units");
            }
            MatrixOps::require_finite(layer_activations[l]);
        }
        MatrixOps::require_finite(inputs);
        MatrixOps::require_finite(targets);
        MatrixOps::require_finite(predictions);

        ForwardTelemetryTensor telemetry;
        telemetry.residuals = predictions - targets;
        telemetry.dead_masks.reserve(layer_activations.size());
        for (const Eigen::MatrixXd& activations : layer_activations) {
            telemetry.dead_masks.push_back(
                (activations.array().abs() <= config.dead_activation_threshold).cast<std::uint8_t>().matrix());
        }
        telemetry.inputs = std::move(inputs);
        telemetry.targets = std::move(targets);
        telemetry.predictions = std::move(predictions);
        telemetry.layer_activations = std::move(layer_activations);
        return telemetry;
    }

private:
    static void require_rows(const Eigen::MatrixXd& m, Eigen::Index n, const std::string& name) {
        if (m.rows() != n) {
            throw std::invalid_argument("Telemetry " + name + " has " + std::to_string(m.rows()) +
                                        " rows, expected " + std::to_string(n));
        }
    }
};

}  // namespace forward_analysis
