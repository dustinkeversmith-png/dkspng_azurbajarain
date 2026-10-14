#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace forward_analysis {

struct MatrixOps {
    template <typename Derived>
    static void require_finite(const Eigen::DenseBase<Derived>& values) {
        if (!values.allFinite()) {
            throw std::invalid_argument("Non-finite floating-point value encountered");
        }
    }

    static double clamp(double value, double low, double high) noexcept {
        return std::max(low, std::min(high, value));
    }

    // Row-major sample convention: X is N x D_in, W is D_out x D_in, result is N x D_out.
    static Eigen::MatrixXd affine_rows(const Eigen::MatrixXd& X,
                                       const Eigen::MatrixXd& W,
                                       const Eigen::VectorXd& b) {
        Eigen::MatrixXd out = X * W.transpose();
        out.rowwise() += b.transpose();
        return out;
    }

    static Eigen::VectorXd softmax(const Eigen::VectorXd& logits) {
        if (logits.size() == 0) {
            return logits;
        }
        const Eigen::ArrayXd shifted = (logits.array() - logits.maxCoeff()).cwiseMax(-60.0).exp();
        const double sum = shifted.sum();
        if (!(sum > 0.0)) {
            return Eigen::VectorXd::Constant(logits.size(), 1.0 / static_cast<double>(logits.size()));
        }
        return (shifted / sum).matrix();
    }

    static Eigen::MatrixXd softmax_rows(const Eigen::MatrixXd& logits) {
        if (logits.size() == 0) {
            return logits;
        }
        const Eigen::VectorXd row_max = logits.rowwise().maxCoeff();
        Eigen::ArrayXXd shifted = (logits.colwise() - row_max).array().cwiseMax(-60.0).exp();
        const Eigen::ArrayXd row_sum = shifted.rowwise().sum();
        shifted.colwise() /= row_sum;
        return shifted.matrix();
    }

    static Eigen::VectorXd layer_norm(const Eigen::VectorXd& x, double eps = 1e-6) {
        if (x.size() == 0) {
            return x;
        }
        const Eigen::ArrayXd centered = x.array() - x.mean();
        const double inv_std = 1.0 / std::sqrt(centered.square().mean() + eps);
        return (centered * inv_std).matrix();
    }

    static Eigen::MatrixXd layer_norm_rows(const Eigen::MatrixXd& X, double eps = 1e-6) {
        if (X.size() == 0) {
            return X;
        }
        const Eigen::MatrixXd centered = X.colwise() - X.rowwise().mean();
        return row_inverse_std(X, eps).asDiagonal() * centered;
    }

    // 1 / sqrt(variance + eps) of each row, the scale layer_norm_rows applies.
    static Eigen::VectorXd row_inverse_std(const Eigen::MatrixXd& X, double eps = 1e-6) {
        const Eigen::MatrixXd centered = X.colwise() - X.rowwise().mean();
        return (centered.array().square().rowwise().mean() + eps).rsqrt().matrix();
    }

    // Gradient through layer_norm_rows given its output and the per-row scale it used.
    static Eigen::MatrixXd layer_norm_rows_backward(const Eigen::MatrixXd& normalized,
                                                    const Eigen::VectorXd& inv_std,
                                                    const Eigen::MatrixXd& d_normalized) {
        const Eigen::VectorXd mean_grad = d_normalized.rowwise().mean();
        const Eigen::VectorXd mean_grad_dot =
            (d_normalized.array() * normalized.array()).rowwise().mean().matrix();
        Eigen::MatrixXd d_input = d_normalized.colwise() - mean_grad;
        d_input -= mean_grad_dot.asDiagonal() * normalized;
        return inv_std.asDiagonal() * d_input;
    }

    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
        const Eigen::ArrayXd clamped = x.array().cwiseMax(-40.0).cwiseMin(40.0);
        return (1.0 + (-clamped).exp()).inverse().matrix();
    }
};

}  // namespace forward_analysis
