#pragma once
#include "stevejob/core/metric_descriptor.hpp"
#include <variant>
#include <vector>
#include <string>

namespace stevejob::models {

struct ModelEvaluation {
    double primary_loss = 0.0;
    std::vector<core::DescribedField> metrics; // E.g., max_residual, sparsity, reconstruction_drift

    double get_metric_value(const std::string& key, double fallback = 0.0) const {
        for (const auto& m : metrics) {
            if (m.key == key && std::holds_alternative<double>(m.value)) {
                return std::get<double>(m.value);
            }
        }
        return fallback;
    }

    static ModelEvaluation from_grid(double loss, double max_residual, double std_residual) {
        ModelEvaluation score;
        score.primary_loss = loss;
        score.metrics = {
            core::DescribedField{
                "loss",
                core::FieldType::Scalar,
                loss,
                "eval_metric",
                "Fixed-point loss, four decimal places.",
                "Lower is better. A checkpoint write requires this value to fall by more than 0.0001."},
            core::DescribedField{
                "max_residual",
                core::FieldType::Scalar,
                max_residual,
                "eval_metric",
                "Peak absolute residual on the eval grid.",
                "A large peak means one sample dominates the error. Prefer a step that shrinks it without raising loss."},
            core::DescribedField{
                "std_residual",
                core::FieldType::Scalar,
                std_residual,
                "eval_metric",
                "Residual standard deviation on the eval grid.",
                "A large spread means the error is uneven across samples."},
        };
        return score;
    }
};

} // namespace stevejob::models
