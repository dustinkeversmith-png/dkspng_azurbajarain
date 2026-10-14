#pragma once

#include "forward_analysis/training/loss_breakdown.hpp"

#include <cstddef>

namespace forward_analysis {

// Losses of one fine-tuning epoch.
struct EpochRecord {
    // 1-based epoch number.
    std::size_t epoch = 0;
    // Mean training losses over that epoch's steps.
    LossBreakdown train;
    // Held-out losses measured after the epoch.
    LossBreakdown eval;
    // True when this epoch's eval total was the lowest so far (its weights were snapshotted).
    bool best_so_far = false;
};

}  // namespace forward_analysis
