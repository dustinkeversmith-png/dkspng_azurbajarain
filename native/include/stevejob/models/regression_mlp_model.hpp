#pragma once

#include "stevejob/core/helpers.hpp"
#include "stevejob/models/base_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

struct RegressionScore {
    double loss = 0.0;
    double max_residual = 0.0;
    double std_residual = 0.0;
};

//  D_in -> H (ReLU) -> D_out (linear). The live task is 2 -> 4 -> 1.
class RegressionMLPModel : public BaseModel {
public:
    RegressionMLPModel(std::vector<std::vector<double>> first_weights,
                       std::vector<double> first_bias,
                       std::vector<std::vector<double>> second_weights,
                       std::vector<double> second_bias)
        : first_weights_(std::move(first_weights)),
          first_bias_(std::move(first_bias)),
          second_weights_(std::move(second_weights)),
          second_bias_(std::move(second_bias)) {
        validate();
    }

    std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& inputs) const override {
        const std::size_t hidden_dim = first_bias_.size();
        const std::size_t out_dim = second_bias_.size();
        std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(out_dim, 0.0));
        for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
            if (inputs[sample].size() != first_weights_.size()) {
                throw std::invalid_argument("each input row must match the layer-1 row count");
            }
            std::vector<double> hidden(hidden_dim, 0.0);
            for (std::size_t unit = 0; unit < hidden_dim; ++unit) {
                double sum = first_bias_[unit];
                for (std::size_t input = 0; input < inputs[sample].size(); ++input) {
                    stevejob::core::require_finite(inputs[sample][input]);
                    sum += inputs[sample][input] * first_weights_[input][unit];
                }
                hidden[unit] = std::max(0.0, sum);
            }
            for (std::size_t coord = 0; coord < out_dim; ++coord) {
                double sum = second_bias_[coord];
                for (std::size_t unit = 0; unit < hidden_dim; ++unit) {
                    sum += hidden[unit] * second_weights_[unit][coord];
                }
                stevejob::core::require_finite(sum);
                outputs[sample][coord] = sum;
            }
        }
        return outputs;
    }

    // Fixed grid, 50 rows by default: y = sin(x0) + cos(x1). Loss is mean squared error.
    RegressionScore evaluate_mse(int sample_count = 50) const {
        if (sample_count < 1 || sample_count > 50) {
            throw std::invalid_argument("regression evaluation uses 1 to 50 samples");
        }
        if (second_bias_.size() != 1 || first_weights_.size() != 2) {
            throw std::invalid_argument("evaluate_mse expects a 2-input, 1-output network");
        }
        constexpr double kPi = 3.141592653589793;
        const double count = static_cast<double>(sample_count);
        double squared = 0.0;
        double max_residual = 0.0;
        double residual_sum = 0.0;
        std::vector<double> residuals(static_cast<std::size_t>(sample_count));
        for (int index = 0; index < sample_count; ++index) {
            const double x0 = (static_cast<double>(index) / count) * kPi;
            const double x1 = (static_cast<double>(sample_count - index) / count) * kPi;
            const double target = std::sin(x0) + std::cos(x1);
            const std::vector<std::vector<double>> input = {{x0, x1}};
            const double predicted = forward(input)[0][0];
            const double residual = std::abs(predicted - target);
            residuals[static_cast<std::size_t>(index)] = residual;
            squared += residual * residual;
            residual_sum += residual;
            max_residual = std::max(max_residual, residual);
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
    std::vector<std::vector<double>> first_weights_;
    std::vector<double> first_bias_;
    std::vector<std::vector<double>> second_weights_;
    std::vector<double> second_bias_;

    static void require_matrix(const std::vector<std::vector<double>>& matrix,
                               std::size_t rows,
                               std::size_t cols,
                               const char* label) {
        if (matrix.size() != rows) {
            throw std::invalid_argument(std::string(label) + " has the wrong number of rows");
        }
        for (const std::vector<double>& row : matrix) {
            if (row.size() != cols) {
                throw std::invalid_argument(std::string(label) + " has the wrong number of columns");
            }
            for (double value : row) {
                stevejob::core::require_finite(value);
            }
        }
    }

    void validate() const {
        if (first_bias_.empty() || second_bias_.empty() || first_weights_.empty()) {
            throw std::invalid_argument("regression layers must be non-empty");
        }
        require_matrix(first_weights_, first_weights_.size(), first_bias_.size(), "layer1 weights");
        require_matrix(second_weights_, first_bias_.size(), second_bias_.size(), "layer2 weights");
        for (double value : first_bias_) {
            stevejob::core::require_finite(value);
        }
        for (double value : second_bias_) {
            stevejob::core::require_finite(value);
        }
    }
};

}  // namespace stevejob::models
