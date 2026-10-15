#pragma once

#include <cstddef>
#include <vector>

namespace forward_analysis {

struct LayerObservation {
    // Number of units in the layer.
    std::size_t width = 0;
    // Units that output zero on every sample in the batch.
    std::vector<std::size_t> dead_units;
    // Units that are non-zero on every sample in the batch.
    std::vector<std::size_t> always_active_units;
    // Number of covariance singular values above spectral_rank_tolerance of the largest (directions the batch spreads along).
    std::size_t effective_rank = 0;
    // Number of singular values of the centered activation matrix above numerical_rank_tolerance (absolute).
    std::size_t numerical_rank = 0;
    // Samples on which no unit of this layer is positive (every activation <= 0).
    std::size_t nonpositive_samples = 0;
    // True when every activation row is a non-negative multiple of one fixed vector (rank-1 through the origin).
    bool lies_on_ray = false;

    std::size_t live_count() const noexcept {
        return width - dead_units.size();
    }
};

}  // namespace forward_analysis
