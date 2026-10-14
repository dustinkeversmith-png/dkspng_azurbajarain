#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/dataset/ground_truth_targets.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <string>

namespace forward_analysis {

struct TrainingSample {
    // Identifier copied from the JSONL record.
    std::string sample_id;
    // Validated telemetry with derived residuals and dead masks.
    ForwardTelemetryTensor telemetry;
    // Eigen signs and observation text from the dataset file.
    GroundTruthTargets ground_truth_targets;
    // Facts measured directly from the telemetry at load time (Aux1 target source and narrator input).
    NetworkObservations measured;
};

}  // namespace forward_analysis
