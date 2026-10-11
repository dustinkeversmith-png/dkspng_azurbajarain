#pragma once

#include "stevejob/core/helpers.hpp"
#include "stevejob/math/linear_algebra.hpp"
#include "stevejob/models/base_model.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

struct Gradients {
    std::vector<std::vector<double>> layer1_weights;
    std::vector<double> layer1_biases;
    std::vector<std::vector<double>> layer2_weights;
    std::vector<double> layer2_biases;
};

struct RingScore {
    double loss = 0.0;
    double mean_radius = 0.0;
    double min_radius = 0.0;
    double max_radius = 0.0;
    double std_radius = 0.0;
};

class RingMLPModel : public BaseModel {
public:
    RingMLPModel(std::vector<std::vector<double>> first_weights,
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
        const std::vector<std::vector<double>>& noise) const override {
        for (const std::vector<double>& row : noise) {
            if (row.size() != 2) {
                throw std::invalid_argument("each noise row must have 2 values");
            }
            stevejob::core::require_finite(row[0]);
            stevejob::core::require_finite(row[1]);
        }
        std::vector<std::vector<double>> generated(noise.size(), std::vector<double>(2));
        for (std::size_t sample = 0; sample < noise.size(); ++sample) {
            double hidden[4];
            for (int unit = 0; unit < 4; ++unit) {
                const double pre =
                    noise[sample][0] * first_weights_[0][static_cast<std::size_t>(unit)] +
                    noise[sample][1] * first_weights_[1][static_cast<std::size_t>(unit)] +
                    first_bias_[static_cast<std::size_t>(unit)];
                hidden[unit] = std::tanh(pre);
            }
            for (int coord = 0; coord < 2; ++coord) {
                double total = second_bias_[static_cast<std::size_t>(coord)];
                for (int unit = 0; unit < 4; ++unit) {
                    total += hidden[unit] *
                             second_weights_[static_cast<std::size_t>(unit)][static_cast<std::size_t>(coord)];
                }
                generated[sample][static_cast<std::size_t>(coord)] = total;
            }
        }
        return generated;
    }

    RingScore evaluate_ring(int seed, int samples) const {
        if (samples < 1 || samples > 100) {
            throw std::invalid_argument("ring evaluation uses 1 to 100 samples");
        }
        std::vector<std::vector<double>> noise(static_cast<std::size_t>(samples), std::vector<double>(2));
        stevejob::core::Rng rng(static_cast<std::uint64_t>(seed));
        for (int sample = 0; sample < samples; ++sample) {
            double left = 0.0;
            double right = 0.0;
            rng.normal_pair(left, right);
            noise[static_cast<std::size_t>(sample)][0] = left;
            noise[static_cast<std::size_t>(sample)][1] = right;
        }
        const std::vector<std::vector<double>> generated = forward(noise);
        double sum = 0.0;
        double square_error = 0.0;
        double min_radius = 0.0;
        double max_radius = 0.0;
        std::vector<double> radii(generated.size());
        for (std::size_t index = 0; index < generated.size(); ++index) {
            const double radius = std::hypot(generated[index][0], generated[index][1]);
            radii[index] = radius;
            sum += radius;
            const double delta = radius - 1.0;
            square_error += delta * delta;
            if (index == 0 || radius < min_radius) {
                min_radius = radius;
            }
            if (index == 0 || radius > max_radius) {
                max_radius = radius;
            }
        }
        const double count = static_cast<double>(radii.size());
        const double mean = sum / count;
        double moment = 0.0;
        for (double radius : radii) {
            const double delta = radius - mean;
            moment += delta * delta;
        }
        const double deviation = std::sqrt(moment / count);
        const double spread_penalty = deviation < 0.2 ? 0.2 - deviation : 0.0;
        RingScore score;
        score.loss = square_error / count + spread_penalty;
        score.mean_radius = mean;
        score.min_radius = min_radius;
        score.max_radius = max_radius;
        score.std_radius = deviation;
        return score;
    }

    void apply_gradients(const Gradients& grads, double learning_rate, double decay) {
        first_weights_ = stevejob::math::apply_gradient_step(
            first_weights_, grads.layer1_weights, learning_rate, decay);
        second_weights_ = stevejob::math::apply_gradient_step(
            second_weights_, grads.layer2_weights, learning_rate, decay);
        if (!first_bias_.empty()) {
            first_bias_ = stevejob::math::apply_gradient_step(
                {first_bias_}, {grads.layer1_biases}, learning_rate, decay)[0];
        }
        if (!second_bias_.empty()) {
            second_bias_ = stevejob::math::apply_gradient_step(
                {second_bias_}, {grads.layer2_biases}, learning_rate, decay)[0];
        }
    }

    const std::vector<std::vector<double>>& first_weights() const { return first_weights_; }
    const std::vector<double>& first_bias() const { return first_bias_; }
    const std::vector<std::vector<double>>& second_weights() const { return second_weights_; }
    const std::vector<double>& second_bias() const { return second_bias_; }

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
        require_matrix(first_weights_, 2, 4, "layer1 weights");
        require_matrix(second_weights_, 4, 2, "layer2 weights");
        if (first_bias_.size() != 4 || second_bias_.size() != 2) {
            throw std::invalid_argument("layer biases must be length 4 then 2");
        }
        for (double value : first_bias_) {
            stevejob::core::require_finite(value);
        }
        for (double value : second_bias_) {
            stevejob::core::require_finite(value);
        }
    }
};

}  // namespace stevejob::models
