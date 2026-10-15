#pragma once

#include "forward_analysis/heads/numerical_prediction_head.hpp"
#include "forward_analysis/math/adam_moment.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace forward_analysis {

// Two-layer residual MLP prediction head with Adam optimization, missing-label masking, and class-balanced weighting.
class DomainMlpHeadBase : public NumericalPredictionHead {
public:
    // Width of the internal tanh hidden layer.
    static constexpr std::size_t kHiddenDim = 64;

    DomainMlpHeadBase(std::string head_id,
                      AnalysisDomain domain,
                      std::size_t d_model,
                      std::size_t out_dim,
                      double weight,
                      bool binary_classification,
                      std::uint64_t seed)
        : id_(std::move(head_id)),
          domain_(domain),
          d_model_(d_model),
          out_dim_(out_dim),
          loss_weight_(weight),
          binary_classification_(binary_classification),
          b1_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(kHiddenDim))),
          b2_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(out_dim))),
          pos_weights_(Eigen::VectorXd::Ones(static_cast<Eigen::Index>(out_dim))),
          neg_weights_(Eigen::VectorXd::Ones(static_cast<Eigen::Index>(out_dim))) {
        // Parameter initialization RNG.
        DeterministicRng rng(seed);
        w1_ = rng.xavier_matrix(kHiddenDim, d_model_);
        w2_ = rng.xavier_matrix(out_dim_, kHiddenDim);
        w_skip_ = rng.xavier_matrix(out_dim_, d_model_) * 0.5;
    }

    std::string id() const override { return id_; }
    AnalysisDomain domain() const override { return domain_; }
    std::size_t output_dim() const noexcept override { return out_dim_; }
    double loss_weight() const noexcept override { return loss_weight_; }

    void set_class_weights(const Eigen::VectorXd& pos_w, const Eigen::VectorXd& neg_w) {
        pos_weights_ = pos_w;
        neg_weights_ = neg_w;
    }

    PredictionResult forward(const EncodedRepresentation& representation) const override {
        // Domain-routed latent vector (d_model).
        const Eigen::VectorXd x = representation.domain_vector(domain_);
        // Hidden layer activations (kHiddenDim).
        const Eigen::VectorXd h = (w1_ * x + b1_).array().tanh().matrix();
        // Raw pre-activation output logits (out_dim).
        const Eigen::VectorXd raw = w2_ * h + w_skip_ * x + b2_;
        // Sigmoid-calibrated outputs in [0, 1].
        const Eigen::VectorXd cal = MatrixOps::sigmoid(raw);

        // Assembled prediction result.
        PredictionResult res;
        res.head_id = id_;
        res.domain = domain_;
        res.raw_outputs = raw;
        res.calibrated_outputs = cal;
        res.discrete_predictions = cal.unaryExpr([](double p) { return p >= 0.5 ? 1.0 : 0.0; });
        res.confidence = (2.0 * (cal.array() - 0.5).abs()).mean();
        return res;
    }

    LossResult compute_loss(const PredictionResult& prediction,
                            const NumericalTargets& targets) const override {
        // Assembled loss result.
        LossResult out;
        out.head_id = id_;
        out.domain = domain_;
        if (!targets.has(id_)) {
            return out;
        }
        // Target vector for this head.
        const Eigen::VectorXd& y = targets.targets.at(id_);
        // Validity mask for this head.
        const Eigen::VectorXd mask =
            targets.masks.count(id_) > 0 ? targets.masks.at(id_) : Eigen::VectorXd::Ones(y.size());
        // Number of valid entries.
        const Eigen::Index k = std::min(prediction.calibrated_outputs.size(), y.size());

        // Accumulated loss over unmasked entries.
        double total_loss = 0.0;
        // Accumulated absolute error over unmasked entries.
        double total_mae = 0.0;
        // Count of matching discrete classifications.
        std::size_t correct = 0;
        // Count of active unmasked entries.
        std::size_t valid_count = 0;

        for (Eigen::Index i = 0; i < k; ++i) {
            if (mask(i) <= 0.0) {
                continue;
            }
            ++valid_count;
            // Calibrated prediction entry clamped away from 0 and 1.
            const double p = std::min(1.0 - 1e-12, std::max(1e-12, prediction.calibrated_outputs(i)));
            // Ground-truth target entry.
            const double t = y(i);
            total_mae += std::abs(p - t);
            if ((p >= 0.5) == (t >= 0.5)) {
                ++correct;
            }
            if (binary_classification_) {
                // Class-balanced binary cross-entropy.
                const double wp = pos_weights_(i);
                // Negative-class weight.
                const double wn = neg_weights_(i);
                total_loss += -(wp * t * std::log(p) + wn * (1.0 - t) * std::log(1.0 - p));
            } else {
                // Squared error + BCE calibration hybrid for bounded [0, 1] regression.
                const double diff = p - t;
                total_loss += 0.5 * diff * diff;
            }
        }
        if (valid_count > 0) {
            // Reciprocal of valid entry count.
            const double inv = 1.0 / static_cast<double>(valid_count);
            out.unweighted_loss = total_loss * inv;
            out.weighted_loss = out.unweighted_loss * loss_weight_;
            out.mean_absolute_error = total_mae * inv;
            out.accuracy = static_cast<double>(correct) * inv;
            out.valid_entries = valid_count;
        }
        return out;
    }

    LossResult compute_loss_and_step(const EncodedRepresentation& representation,
                                     const NumericalTargets& targets,
                                     double learning_rate,
                                     Eigen::MatrixXd& d_latents_z_accum) override {
        // Forward pass prediction.
        const PredictionResult pred = forward(representation);
        // Loss evaluation before parameter update.
        const LossResult loss = compute_loss(pred, targets);
        if (loss.valid_entries == 0 || !targets.has(id_)) {
            return loss;
        }

        // Domain-routed input vector (d_model).
        const Eigen::VectorXd x = representation.domain_vector(domain_);
        // Hidden layer activations (kHiddenDim).
        const Eigen::VectorXd h = (w1_ * x + b1_).array().tanh().matrix();
        // Ground-truth target vector.
        const Eigen::VectorXd& y = targets.targets.at(id_);
        // Per-entry validity mask.
        const Eigen::VectorXd mask =
            targets.masks.count(id_) > 0 ? targets.masks.at(id_) : Eigen::VectorXd::Ones(y.size());
        // Number of valid entries.
        const double inv_valid = 1.0 / static_cast<double>(loss.valid_entries);

        // Gradient with respect to raw pre-sigmoid logits (out_dim).
        Eigen::VectorXd d_raw = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(out_dim_));
        for (Eigen::Index i = 0; i < std::min<Eigen::Index>(static_cast<Eigen::Index>(out_dim_), y.size()); ++i) {
            if (mask(i) <= 0.0) {
                continue;
            }
            // Calibrated output p_i in (0, 1).
            const double p = std::min(1.0 - 1e-9, std::max(1e-9, pred.calibrated_outputs(i)));
            // Target value t_i.
            const double t = y(i);
            if (binary_classification_) {
                // Derivative of -(wp * t * log(p) + wn * (1 - t) * log(1 - p)) wrt raw logit z_i.
                const double wp = pos_weights_(i);
                // Negative-class weight.
                const double wn = neg_weights_(i);
                d_raw(i) = (wp * t * (p - 1.0) + wn * (1.0 - t) * p) * inv_valid;
            } else {
                // Smooth logit gradient (p - t) for bounded [0, 1] targets (avoids sigmoid saturation vanishing).
                d_raw(i) = (p - t) * inv_valid;
            }
        }

        // Gradient routed back into the Perceiver latents.
        const Eigen::VectorXd d_h = w2_.transpose() * d_raw;
        // Pre-tanh hidden gradient.
        const Eigen::VectorXd d_pre_h = d_h.array() * (1.0 - h.array().square());
        // Gradient with respect to the domain vector x.
        const Eigen::VectorXd d_x = (w1_.transpose() * d_pre_h + w_skip_.transpose() * d_raw) * loss_weight_;
        EncodedRepresentation::accumulate_domain_gradient(domain_, d_x, d_latents_z_accum);

        if (learning_rate > 0.0) {
            ++adam_step_;
            // Adam step size for head parameters.
            const double adam_lr = std::min(0.003, learning_rate * 0.06);
            // Weight and bias gradients.
            const Eigen::MatrixXd g_w2 = d_raw * h.transpose();
            // Skip-connection weight gradient.
            const Eigen::MatrixXd g_skip = d_raw * x.transpose();
            // First-layer weight gradient.
            const Eigen::MatrixXd g_w1 = d_pre_h * x.transpose();

            m_w2_.apply(w2_, g_w2, adam_lr, adam_step_);
            m_b2_.apply(b2_, d_raw, adam_lr, adam_step_);
            m_skip_.apply(w_skip_, g_skip, adam_lr, adam_step_);
            m_w1_.apply(w1_, g_w1, adam_lr, adam_step_);
            m_b1_.apply(b1_, d_pre_h, adam_lr, adam_step_);
        }
        return loss;
    }

private:
    // Head identifier string.
    std::string id_;
    // Mathematical analysis domain of this head.
    AnalysisDomain domain_;
    // Input vector dimension (d_model).
    std::size_t d_model_ = 256;
    // Output vector dimension.
    std::size_t out_dim_ = 1;
    // Task loss weight.
    double loss_weight_ = 1.0;
    // True for binary classification heads, false for bounded regression heads.
    bool binary_classification_ = false;
    // Adam step counter.
    std::uint64_t adam_step_ = 0;
    // First-layer MLP weights (kHiddenDim x d_model).
    Eigen::MatrixXd w1_;
    // First-layer MLP biases (kHiddenDim).
    Eigen::VectorXd b1_;
    // Second-layer MLP weights (out_dim x kHiddenDim).
    Eigen::MatrixXd w2_;
    // Second-layer MLP biases (out_dim).
    Eigen::VectorXd b2_;
    // Direct linear skip weights (out_dim x d_model).
    Eigen::MatrixXd w_skip_;
    // Positive-class weights per output component.
    Eigen::VectorXd pos_weights_;
    // Negative-class weights per output component.
    Eigen::VectorXd neg_weights_;
    // Adam moments for each parameter tensor.
    AdamMoment m_w1_;
    AdamMoment m_b1_;
    AdamMoment m_w2_;
    AdamMoment m_b2_;
    AdamMoment m_skip_;
};

}  // namespace forward_analysis
