#pragma once

#include "forward_analysis/description/observation_text.hpp"

#include <Eigen/Core>

namespace forward_analysis {

struct GroundTruthTargets {
    // Expected signs (+1/-1) of the dominant first-layer covariance eigenvector components; Aux2 target.
    Eigen::VectorXd eigen_signs;
    // The four qualitative observation fields the decoder learns to produce.
    ObservationText observations;
};

}  // namespace forward_analysis
