#pragma once

#include "forward_analysis/math/deterministic_rng.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

class LinearInputAdapter {
public:
    LinearInputAdapter() = default;

    explicit LinearInputAdapter(std::size_t d_model,
                                std::size_t initial_d_total = 32,
                                std::uint64_t seed = 1337ULL)
        : d_model_(d_model),
          seed_(seed),
          weights_(static_cast<Eigen::Index>(d_model), 0),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d_model))) {
        ensure_input_capacity(initial_d_total);
    }

    void ensure_input_capacity(std::size_t required_d_total) {
        const Eigen::Index required = static_cast<Eigen::Index>(required_d_total);
        const Eigen::Index current_cols = weights_.cols();
        if (required <= current_cols) {
            return;
        }
        DeterministicRng rng(seed_ + static_cast<std::uint64_t>(current_cols) * 7919ULL);
        const double scale = 1.0 / std::sqrt(static_cast<double>(d_model_ + required_d_total));
        weights_.conservativeResize(Eigen::NoChange, required);
        for (Eigen::Index r = 0; r < weights_.rows(); ++r) {
            for (Eigen::Index c = current_cols; c < required; ++c) {
                weights_(r, c) = rng.uniform(-scale, scale);
            }
        }
    }

    Eigen::MatrixXd project(const Eigen::MatrixXd& probes) {
        ensure_input_capacity(static_cast<std::size_t>(probes.cols()));
        Eigen::MatrixXd projected = probes * weights_.leftCols(probes.cols()).transpose();
        projected.rowwise() += biases_.transpose();
        return projected;
    }

    void update_from_sequence_grad(const Eigen::MatrixXd& probes,
                                   const Eigen::MatrixXd& d_projected,
                                   double learning_rate) {
        if (probes.rows() == 0 || d_projected.rows() == 0 || learning_rate <= 0.0) {
            return;
        }
        if (d_projected.cols() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("LinearInputAdapter gradient width must equal d_model");
        }
        const Eigen::Index n = std::min(probes.rows(), d_projected.rows());
        ensure_input_capacity(static_cast<std::size_t>(probes.cols()));

        const Eigen::MatrixXd grad =
            d_projected.topRows(n).cwiseMax(-5.0).cwiseMin(5.0) * (learning_rate / static_cast<double>(n));
        biases_ -= grad.colwise().sum().transpose();
        weights_.leftCols(probes.cols()).noalias() -= grad.transpose() * probes.topRows(n);
    }

    std::size_t d_model() const noexcept { return d_model_; }
    const Eigen::MatrixXd& weights() const noexcept { return weights_; }
    const Eigen::VectorXd& biases() const noexcept { return biases_; }

private:
    // Output width of the projection (the encoder's model width).
    std::size_t d_model_ = 256;
    // Base seed for initializing newly added weight columns when the probe width grows.
    std::uint64_t seed_ = 1337ULL;
    // Projection matrix (d_model x probe width); grows column-wise to fit wider telemetry.
    Eigen::MatrixXd weights_;
    // Bias added to every projected probe row (d_model).
    Eigen::VectorXd biases_;
};

}  // namespace forward_analysis
