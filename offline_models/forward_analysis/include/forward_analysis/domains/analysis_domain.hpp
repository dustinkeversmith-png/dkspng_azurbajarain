#pragma once

#include <cstddef>

namespace forward_analysis {

// Categories of mathematical analysis supported by the numerical reasoning architecture.
enum class AnalysisDomain : std::size_t {
    Statistical = 0,
    Spectral = 1,
    Geometric = 2,
    Activation = 3,
    Functional = 4,
    Structural = 5
};

}  // namespace forward_analysis
