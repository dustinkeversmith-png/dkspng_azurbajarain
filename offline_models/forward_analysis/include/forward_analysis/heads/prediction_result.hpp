#pragma once

#include "forward_analysis/domains/analysis_domain.hpp"

#include <Eigen/Core>

#include <string>

namespace forward_analysis {

// Output of a NumericalPredictionHead forward pass.
struct PredictionResult {
    // Identifier of the head that produced this prediction.
    std::string head_id;
    // Mathematical analysis domain of the head.
    AnalysisDomain domain = AnalysisDomain::Statistical;
    // Uncalibrated linear/MLP logits or raw regression outputs.
    Eigen::VectorXd raw_outputs;
    // Calibrated outputs (probabilities in [0, 1], bounded regressions in [0, 1] or [-1, 1]).
    Eigen::VectorXd calibrated_outputs;
    // Thresholded discrete predictions (e.g. +1/-1 signs or 0/1 binary indicators).
    Eigen::VectorXd discrete_predictions;
    // Mean prediction confidence in [0, 1].
    double confidence = 1.0;
};

}  // namespace forward_analysis
