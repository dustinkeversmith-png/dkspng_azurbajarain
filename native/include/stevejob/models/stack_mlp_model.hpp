#pragma once

#include "stevejob/core/helpers.hpp"
#include "stevejob/messaging/compressor.hpp"
#include "stevejob/models/regression_mlp_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

// Hidden layers use ReLU. The last layer is linear. Input width is the first
// layer's row count. A 2-input network keeps the sin(x0)+cos(x1) grid.
class StackMLPModel {
public:
    explicit StackMLPModel(stevejob::messaging::ModelData model) : model_(std::move(model)) { validate(); }

    RegressionScore evaluate_mse(int sample_count = 50) const {
        if (sample_count < 1 || sample_count > 50) {
            throw std::invalid_argument("regression evaluation uses 1 to 50 samples");
        }
        constexpr double kPi = 3.141592653589793;
        const std::size_t in_dim = model_.layers.front().weights.size();
        const std::size_t out_dim = model_.layers.back().biases.size();
        const double count = static_cast<double>(sample_count);
        double squared = 0.0;
        double max_residual = 0.0;
        double residual_sum = 0.0;
        std::vector<double> residuals(static_cast<std::size_t>(sample_count));
        for (int index = 0; index < sample_count; ++index) {
            std::vector<double> input(in_dim, 0.0);
            for (std::size_t axis = 0; axis < in_dim; ++axis) {
                if (axis == 0) {
                    input[axis] = (static_cast<double>(index) / count) * kPi;
                } else if (axis == 1) {
                    input[axis] = (static_cast<double>(sample_count - index) / count) * kPi;
                } else {
                    const int phase = (index + static_cast<int>(axis) * 3) % sample_count;
                    input[axis] = (static_cast<double>(phase) / count) * kPi;
                }
            }
            const std::vector<double> predicted = forward_one(input);
            double sample_square = 0.0;
            double sample_peak = 0.0;
            for (std::size_t coord = 0; coord < out_dim; ++coord) {
                const double target = coord == 0 ? scalar_target(input) : std::sin(input[coord % in_dim]);
                const double residual = std::abs(predicted[coord] - target);
                sample_square += residual * residual;
                sample_peak = std::max(sample_peak, residual);
            }
            residuals[static_cast<std::size_t>(index)] = sample_peak;
            squared += sample_square / static_cast<double>(out_dim);
            residual_sum += sample_peak;
            max_residual = std::max(max_residual, sample_peak);
        }
        const double mean_residual = residual_sum / count;
        double moment = 0.0;
        for (double residual : residuals) {
            const double delta = residual - mean_residual;
            moment += delta * delta;
        }
        RegressionScore score;
        score.loss = squared / count;
        score.max_residual = max_residual;
        score.std_residual = std::sqrt(moment / count);
        return score;
    }

private:
    stevejob::messaging::ModelData model_;

    static double scalar_target(const std::vector<double>& input) {
        double total = 0.0;
        for (std::size_t axis = 0; axis < input.size(); ++axis) {
            total += (axis % 2 == 0) ? std::sin(input[axis]) : std::cos(input[axis]);
        }
        return total;
    }

    std::vector<double> forward_one(const std::vector<double>& input) const {
        std::vector<double> current = input;
        const std::size_t last = model_.layers.size() - 1;
        for (std::size_t index = 0; index < model_.layers.size(); ++index) {
            const stevejob::messaging::LayerRecord& layer = model_.layers[index];
            const bool relu = index != last;
            std::vector<double> next(layer.biases.size(), 0.0);
            for (std::size_t col = 0; col < layer.biases.size(); ++col) {
                double sum = layer.biases[col];
                for (std::size_t row = 0; row < current.size(); ++row) {
                    sum += current[row] * layer.weights[row][col];
                }
                stevejob::core::require_finite(sum);
                next[col] = relu ? std::max(0.0, sum) : sum;
            }
            current = std::move(next);
        }
        return current;
    }

    void validate() const {
        if (model_.layers.empty()) {
            throw std::invalid_argument("a stack needs at least one layer");
        }
        for (std::size_t index = 0; index < model_.layers.size(); ++index) {
            const stevejob::messaging::LayerRecord& layer = model_.layers[index];
            if (layer.weights.empty() || layer.weights.front().empty()) {
                throw std::invalid_argument(layer.name + " is empty");
            }
            if (layer.biases.size() != layer.weights.front().size()) {
                throw std::invalid_argument(layer.name + " bias width does not match its columns");
            }
            for (const std::vector<double>& row : layer.weights) {
                if (row.size() != layer.biases.size()) {
                    throw std::invalid_argument(layer.name + " has a ragged weight row");
                }
            }
            if (index > 0 && model_.layers[index - 1].biases.size() != layer.weights.size()) {
                throw std::invalid_argument(layer.name + " does not match the previous layer width");
            }
        }
    }
};

}  // namespace stevejob::models
