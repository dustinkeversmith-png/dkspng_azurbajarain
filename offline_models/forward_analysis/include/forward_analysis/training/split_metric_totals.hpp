#pragma once

#include <cstddef>
#include <vector>

namespace forward_analysis {

// Running sums of distillation metrics for one split (e.g. "train", "validation", "ood"); divided by samples when reported.
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
    // Narrator fields that passed the consistency check.
    std::size_t narrated_consistent_fields = 0;
    // Final output fields that passed the consistency check.
    std::size_t output_consistent_fields = 0;
    // Ground-truth fields that passed the consistency check.
    std::size_t ground_truth_consistent_fields = 0;
    // Per field: sum of decoder word F1.
    std::vector<double> field_decoder_f1;
    // Per field: decoder fields that passed the consistency check.
    std::vector<std::size_t> field_decoder_consistent;
    // Eigen-sign components predicted correctly.
    std::size_t sign_matches = 0;
    // Eigen-sign components compared.
    std::size_t sign_total = 0;
    // Per eigen-sign component: records where that component was predicted correctly.
    std::vector<std::size_t> sign_component_matches;
    // Records whose whole eigen-sign vector was predicted correctly.
    std::size_t sign_vector_matches = 0;
    // Sum of per-record descriptor mean absolute error.
    double descriptor_mae = 0.0;
};

}  // namespace forward_analysis
