#pragma once

#include <cstddef>

namespace forward_analysis {

// Running sums of distillation metrics for one split ("train" or "eval"); divided by samples when reported.
struct SplitMetricTotals {
    // Records accumulated.
    std::size_t samples = 0;
    // Sum of per-record mean word F1 of the raw decoder text.
    double decoder_f1 = 0.0;
    // Sum of per-record mean word F1 of the narrator text.
    double narrated_f1 = 0.0;
    // Sum of per-record mean word F1 of the final output text.
    double output_f1 = 0.0;
    // Decoder fields that passed the consistency check.
    std::size_t consistent_fields = 0;
    // Eigen-sign components predicted correctly.
    std::size_t sign_matches = 0;
    // Eigen-sign components compared.
    std::size_t sign_total = 0;
    // Sum of per-record descriptor mean absolute error.
    double descriptor_mae = 0.0;
};

}  // namespace forward_analysis
