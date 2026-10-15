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
    // Signs (+1 / -1, zero counted as +1) of the eigenvalues, in descending order, of the sample covariance of the first
    // eigen_sign_dim input coordinates with its diagonal set to zero (inputs centered, divisor max(1, N - 1)).
    Eigen::VectorXd eigen_signs;
    // The two input coordinates with the largest max_k |Pearson(input, final hidden unit k)|, strongest first.
    std::vector<std::size_t> strongest_inputs;
    // The two input coordinates with the smallest such association, weakest first.
    std::vector<std::size_t> weakest_inputs;
    // Largest input / final-hidden-unit |Pearson| over all pairs (0 when the final layer does not vary).
    double strongest_association = 0.0;
    // Outputs whose range_ratio is below range_ratio_low.
    std::size_t outputs_with_smaller_range = 0;
    // Outputs whose range_ratio is above range_ratio_high.
    std::size_t outputs_with_larger_range = 0;
    // Outputs whose range_ratio lies in [range_ratio_low, range_ratio_high].
    std::size_t outputs_with_comparable_range = 0;
    // Sum of sign_agreements over all outputs.
    std::size_t sign_agreements = 0;
    // Number of prediction entries (samples x outputs).
    std::size_t sign_entries = 0;
    // True when every hidden layer's numerical rank is at most narrow_rank_limit.
    bool narrow_representation = false;

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
