#pragma once

#include "stevejob/core/helpers.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::math {

inline std::vector<double> matvec(const std::vector<std::vector<double>>& matrix,
                                  const std::vector<double>& vector) {
    std::vector<double> product(matrix.size(), 0.0);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        double total = 0.0;
        for (std::size_t column = 0; column < matrix[row].size(); ++column) {
            stevejob::core::require_finite(matrix[row][column]);
            const double activation = column < vector.size() ? vector[column] : 0.0;
            stevejob::core::require_finite(activation);
            total += matrix[row][column] * activation;
        }
        product[row] = total;
    }
    return product;
}

inline std::vector<std::vector<double>> gemm(const std::vector<std::vector<double>>& left,
                                             const std::vector<std::vector<double>>& right) {
    if (left.empty() || right.empty()) {
        throw std::invalid_argument("gemm requires non-empty matrices");
    }
    const std::size_t shared = right.size();
    const std::size_t columns = right[0].size();
    for (const std::vector<double>& row : left) {
        if (row.size() != shared) {
            throw std::invalid_argument("gemm inner dimensions must match");
        }
        for (double value : row) {
            stevejob::core::require_finite(value);
        }
    }
    for (const std::vector<double>& row : right) {
        if (row.size() != columns) {
            throw std::invalid_argument("gemm right-hand rows must be rectangular");
        }
        for (double value : row) {
            stevejob::core::require_finite(value);
        }
    }
    std::vector<std::vector<double>> product(left.size(), std::vector<double>(columns, 0.0));
    for (std::size_t row = 0; row < left.size(); ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            double total = 0.0;
            for (std::size_t inner = 0; inner < shared; ++inner) {
                total += left[row][inner] * right[inner][column];
            }
            product[row][column] = total;
        }
    }
    return product;
}

inline std::vector<std::vector<double>> apply_gradient_step(
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<double>>& gradients,
    double learning_rate,
    double weight_decay,
    double dampening = 1.0) {
    stevejob::core::require_finite(learning_rate);
    stevejob::core::require_finite(weight_decay);
    stevejob::core::require_finite(dampening);
    if (learning_rate < 0.0) {
        throw std::invalid_argument("learning_rate must be zero or positive");
    }
    if (weight_decay < 0.0 || dampening < 0.0) {
        throw std::invalid_argument("weight_decay and dampening must be zero or positive");
    }

    const double step = learning_rate * dampening;
    std::vector<std::vector<double>> updated = weights;
    for (std::size_t row = 0; row < updated.size(); ++row) {
        for (std::size_t column = 0; column < updated[row].size(); ++column) {
            stevejob::core::require_finite(updated[row][column]);
            double gradient = 0.0;
            if (row < gradients.size() && column < gradients[row].size()) {
                gradient = gradients[row][column];
                stevejob::core::require_finite(gradient);
            }
            const double weight = updated[row][column];
            updated[row][column] = weight - (step * gradient) - (step * weight_decay * weight);
        }
    }
    return updated;
}

inline std::vector<std::vector<double>> linear_outputs(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& weights,
    const std::vector<double>& biases) {
    std::vector<std::vector<double>> outputs(inputs.size());
    for (std::size_t sample = 0; sample < inputs.size(); ++sample) {
        outputs[sample].resize(weights.size());
        for (std::size_t unit = 0; unit < weights.size(); ++unit) {
            double total = unit < biases.size() ? biases[unit] : 0.0;
            stevejob::core::require_finite(total);
            const std::size_t width = weights[unit].size();
            for (std::size_t feature = 0; feature < width; ++feature) {
                stevejob::core::require_finite(weights[unit][feature]);
                const double activation =
                    feature < inputs[sample].size() ? inputs[sample][feature] : 0.0;
                stevejob::core::require_finite(activation);
                total += weights[unit][feature] * activation;
            }
            outputs[sample][unit] = total;
        }
    }
    return outputs;
}

}  // namespace stevejob::math
