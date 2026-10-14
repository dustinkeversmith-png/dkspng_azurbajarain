#pragma once

#include "forward_analysis/training/epoch_record.hpp"

#include <cstddef>
#include <vector>

namespace forward_analysis {

// Result of a fine-tuning phase with early stopping and best-checkpoint restore.
struct FinetuneOutcome {
    // One record per epoch actually run.
    std::vector<EpochRecord> history;
    // 1-based epoch whose weights were restored (lowest eval total); 0 when no epoch ran.
    std::size_t best_epoch = 0;
    // Eval losses of the restored epoch.
    LossBreakdown best_eval;
    // True when training stopped before max_epochs because eval token CE had not decreased for patience epochs.
    bool stopped_early = false;
    // Upper bound on epochs requested.
    std::size_t max_epochs = 0;
};

}  // namespace forward_analysis
