#pragma once

#include "forward_analysis/analysis/gating_observation.hpp"
#include "forward_analysis/analysis/input_driver.hpp"
#include "forward_analysis/analysis/layer_observation.hpp"
#include "forward_analysis/analysis/output_observation.hpp"
#include "forward_analysis/telemetry/telemetry_shape.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace forward_analysis {

struct NetworkObservations {
    // Batch size, input width, hidden widths and output width of the observed telemetry.
    TelemetryShape shape;
    // One entry per hidden layer, in forward order.
    std::vector<LayerObservation> layers;
    // Half-space gating of the first layer, if any samples are fully closed.
    GatingObservation gating;
    // Input coordinates strongly correlated with the first layer's leading or secondary principal axis, strongest first.
    std::vector<InputDriver> drivers;
    // Input coordinates that are neither the gate nor a driver but correlate with some target column at or above
    // the driver threshold, strongest first (inputs the targets depend on, whether or not the network uses them).
    std::vector<std::size_t> target_related_coordinates;
    // One entry per network output.
    std::vector<OutputObservation> outputs;
    // True when there are several outputs and every pair of prediction columns is perfectly (+/-) correlated.
    bool outputs_move_in_lockstep = false;
    // Signs of the first eigen_sign_dim entries of the first layer's dominant covariance eigenvector.
    Eigen::VectorXd eigen_signs;

    std::vector<std::size_t> driver_coordinates(std::size_t axis) const {
        std::vector<std::size_t> out;
        for (const InputDriver& d : drivers) {
            if (d.axis == axis) {
                out.push_back(d.coordinate);
            }
        }
        return out;
    }
};

}  // namespace forward_analysis
