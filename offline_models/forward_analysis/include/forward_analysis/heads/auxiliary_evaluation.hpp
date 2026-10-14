#pragma once

#include <Eigen/Core>

namespace forward_analysis {

struct AuxiliaryEvaluation {
    // Descriptor predicted from the latent bottleneck (same layout as ObservationDescriptor).
    Eigen::VectorXd predicted_observation_descriptor;
    // Predicted dominant-eigenvector signs (+1 / -1).
    Eigen::VectorXd predicted_eigen_signs;
    // Aux1: half mean squared error between predicted and measured descriptor.
    double aux1_observation_loss = 0.0;
    // Aux2: binary cross-entropy between predicted and target eigenvector signs.
    double aux2_eigen_sign_loss = 0.0;
};

}  // namespace forward_analysis
