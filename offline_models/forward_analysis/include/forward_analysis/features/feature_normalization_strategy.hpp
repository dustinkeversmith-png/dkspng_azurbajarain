#pragma once

#include <cstddef>

namespace forward_analysis {

// Normalization transformation applied by a typed feature extractor to keep features bounded across shapes.
enum class FeatureNormalizationStrategy : std::size_t {
    Identity = 0,
    UnitInterval = 1,
    SignedUnitInterval = 2,
    TanhSquash = 3,
    LogScale = 4,
    StandardScore = 5
};

}  // namespace forward_analysis
