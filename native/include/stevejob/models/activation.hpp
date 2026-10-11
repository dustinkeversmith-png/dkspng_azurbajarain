#pragma once

#include <algorithm>
#include <cmath>

namespace stevejob::models {

enum class ActivationType { Linear, ReLU, Tanh, Sigmoid };

inline double apply_activation(double value, ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return std::max(0.0, value);
        case ActivationType::Tanh:
            return std::tanh(value);
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-value));
        case ActivationType::Linear:
        default:
            return value;
    }
}

}  // namespace stevejob::models
