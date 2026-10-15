#pragma once

#include "forward_analysis/domains/analysis_domain.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <string>

namespace forward_analysis {

// Loss and evaluation metrics computed by a NumericalPredictionHead.
struct LossResult {
    // Identifier of the head.
    std::string head_id;
    // Mathematical domain of the head.
    AnalysisDomain domain = AnalysisDomain::Statistical;
    // Unweighted mean loss over valid target entries.
    double unweighted_loss = 0.0;
    // Task-weighted loss contribution.
    double weighted_loss = 0.0;
    // Mean absolute error between calibrated predictions and valid targets.
    double mean_absolute_error = 0.0;
    // Classification accuracy over valid entries (for classification heads).
    double accuracy = 0.0;
    // Count of valid (unmasked) target entries evaluated.
    std::size_t valid_entries = 0;
};

}  // namespace forward_analysis
