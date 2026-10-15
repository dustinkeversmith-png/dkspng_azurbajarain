#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstdint>

namespace forward_analysis {

// First/second moment estimates for one parameter tensor updated with Adam.
class AdamMoment {
public:
    // Grows the moment buffers to rows x cols, zero-filling new entries (used when a vocabulary grows).
    void ensure_shape(Eigen::Index rows, Eigen::Index cols) {
        if (first_.rows() == rows && first_.cols() == cols) {
            return;
        }
        const Eigen::Index old_rows = first_.rows();
        const Eigen::Index old_cols = first_.cols();
        first_.conservativeResize(rows, cols);
        second_.conservativeResize(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) {
                if (r >= old_rows || c >= old_cols) {
                    first_(r, c) = 0.0;
                    second_(r, c) = 0.0;
                }
            }
        }
    }

    const Eigen::MatrixXd& first() const noexcept { return first_; }
    const Eigen::MatrixXd& second() const noexcept { return second_; }

    // Replaces both moment estimates, e.g. with the copies a device has been updating.
    void assign(const Eigen::MatrixXd& first, const Eigen::MatrixXd& second) {
        first_ = first;
        second_ = second;
    }

    static constexpr double beta1() noexcept { return kBeta1; }
    static constexpr double beta2() noexcept { return kBeta2; }
    static constexpr double epsilon() noexcept { return kEpsilon; }

    // param -= lr * m_hat / (sqrt(v_hat) + eps); step is the 1-based global Adam step count.
    template <typename Param, typename Grad>
    void apply(Param& param, const Grad& grad, double learning_rate, std::uint64_t step) {
        ensure_shape(param.rows(), param.cols());
        first_ = kBeta1 * first_ + (1.0 - kBeta1) * grad;
        second_ = kBeta2 * second_ + (1.0 - kBeta2) * grad.cwiseProduct(grad);
        const double t = static_cast<double>(step);
        const double correction1 = 1.0 - std::pow(kBeta1, t);
        const double correction2 = 1.0 - std::pow(kBeta2, t);
        param -= (learning_rate * (first_.array() / correction1) /
                  ((second_.array() / correction2).sqrt() + kEpsilon))
                     .matrix();
    }

private:
    static constexpr double kBeta1 = 0.9;
    static constexpr double kBeta2 = 0.999;
    static constexpr double kEpsilon = 1e-8;

    // Running mean of gradients.
    Eigen::MatrixXd first_;
    // Running mean of squared gradients.
    Eigen::MatrixXd second_;
};

}  // namespace forward_analysis
