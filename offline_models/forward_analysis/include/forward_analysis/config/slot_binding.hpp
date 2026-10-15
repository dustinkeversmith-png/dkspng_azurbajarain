#pragma once

namespace forward_analysis {

// How numbers in the training text become slot tokens the decoder writes and the measured telemetry fills.
enum class SlotBinding {
    // ObservationDelexicalizer: shape and index references matched by value against layer widths, dead / live
    // counts, gating splits, driver coordinates and index lists.
    kShapeFacts,
    // BatchStatementDelexicalizer: every number in a batch-statistic statement becomes the slot of the quantity
    // that statement position reports (N, W_l, ZERO_Ll, NONPOS_Ll, STRONG_k, WEAK_k, RANK_Ll, D_OUT, range
    // counts, SIGN_AGREE, SIGN_ENTRIES), independent of its value.
    kBatchStatistics,
};

}  // namespace forward_analysis
