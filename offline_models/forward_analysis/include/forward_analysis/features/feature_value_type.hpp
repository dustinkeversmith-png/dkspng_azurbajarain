#pragma once

#include <cstddef>

namespace forward_analysis {

// Mathematical value types produced by typed feature extractors.
enum class FeatureValueType : std::size_t {
    Scalar = 0,
    Boolean = 1,
    Integer = 2,
    Vector = 3,
    Matrix = 4,
    Distribution = 5,
    Categorical = 6
};

}  // namespace forward_analysis
