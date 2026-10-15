#pragma once

#include <cstddef>

namespace forward_analysis {

// Five-stage feature and token representation hierarchy for ablation and training.
enum class FeatureAblationLevel : std::size_t {
    // 1. Existing 25 per-channel features only.
    kBaseline25 = 0,
    // 2. Expanded 38-feature channel tokens with structural, spectral, and association context.
    kExpandedChannel = 1,
    // 3. Expanded channel tokens plus observation-level sample tokens.
    kChannelPlusSample = 2,
    // 4. Expanded channel tokens plus pairwise/spectral relationship tokens.
    kChannelPlusRelationship = 3,
    // 5. Complete architecture: channel + sample + relationship + layer + global domain tokens.
    kCompleteArchitecture = 4
};

}  // namespace forward_analysis
