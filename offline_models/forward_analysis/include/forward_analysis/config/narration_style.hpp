#pragma once

namespace forward_analysis {

// Which deterministic narrator writes the measured-facts text (the fallback for unverifiable decoder fields).
enum class NarrationStyle {
    // ObservationNarrator: dead units, rays, gating, drivers and output fidelity in descriptive prose.
    kMeasuredFacts,
    // BatchStatisticsNarrator: per-layer zero coordinates and non-positive observations, input associations,
    // numerical ranks, range ratios and sign agreement, in the corrected dataset's phrasing.
    kBatchStatistics,
};

}  // namespace forward_analysis
