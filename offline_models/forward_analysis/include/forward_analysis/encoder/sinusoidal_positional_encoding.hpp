#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstddef>

namespace forward_analysis {

struct SinusoidalPositionalEncoding {
    static Eigen::MatrixXd encoding(std::size_t num_positions, std::size_t d_model) {
        const Eigen::Index n = static_cast<Eigen::Index>(num_positions);
        const Eigen::Index d = static_cast<Eigen::Index>(d_model);
        Eigen::MatrixXd pe(n, d);
        if (n == 0 || d == 0) {
            return pe;
        }
        const Eigen::ArrayXd positions = Eigen::ArrayXd::LinSpaced(n, 1.0, static_cast<double>(n));
        for (Eigen::Index dim = 0; dim < d; ++dim) {
            const Eigen::Index pair_idx = (dim / 2) * 2;
            const double exponent = static_cast<double>(pair_idx) / static_cast<double>(d);
            const Eigen::ArrayXd angle = positions / std::pow(10000.0, exponent);
            if (dim % 2 == 0) {
                pe.col(dim) = angle.sin().matrix();
            } else {
                pe.col(dim) = angle.cos().matrix();
            }
        }
        return pe;
    }

    static void add_in_place(Eigen::MatrixXd& sequence) {
        if (sequence.size() == 0) {
            return;
        }
        sequence += encoding(static_cast<std::size_t>(sequence.rows()),
                             static_cast<std::size_t>(sequence.cols()));
    }
};

}  // namespace forward_analysis
