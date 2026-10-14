#pragma once

#include <Eigen/Core>

namespace forward_analysis {

struct SpectralProfile {
    // Singular values of the activation covariance matrix, largest first.
    Eigen::VectorXd singular_values;
    // Principal axes (right singular vectors) as columns, each sign-pinned so its largest entry is positive.
    Eigen::MatrixXd principal_axes;
    // Sign (+1 / -1) of each component of the dominant principal axis.
    Eigen::VectorXd dominant_eigen_signs;
    // sqrt(1 - (s1/s0)^2): 0 when the top two directions carry equal variance, 1 when one direction dominates.
    double eccentricity = 0.0;
};

}  // namespace forward_analysis
