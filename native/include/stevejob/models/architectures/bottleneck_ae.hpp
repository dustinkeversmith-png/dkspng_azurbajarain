#pragma once

#include "stevejob/models/model_template.hpp"

#include <algorithm>
#include <cmath>

namespace stevejob::models {

class BottleneckAutoencoder : public ModelTemplate {
public:
    BottleneckAutoencoder() : net_("BottleneckAutoencoder") {
        net_.architecture = "bottleneck_ae";
        net_.use_architecture_slices();
        net_.add_layer("encode", 8, 2, ActivationType::ReLU)
            .add_layer("decode", 2, 8, ActivationType::Linear)
            .set_loss(LossType::MeanSquaredError);
        net_.initialize_default_weights();
    }

    void adopt(NeuralModel model) { net_ = std::move(model); }

    NeuralModel& network() override { return net_; }
    const NeuralModel& network() const override { return net_; }

    std::vector<double> forward(const std::vector<double>& input) const override { return net_.forward(input); }

    RegressionScore evaluate(int sample_count = 50) const override { return score_network(net_, sample_count); }

    RegressionScore score_network(const NeuralModel& model, int sample_count = 50) const override {
        if (sample_count < 1 || sample_count > 50) {
            throw std::invalid_argument("reconstruction evaluation uses 1 to 50 samples");
        }
        const std::size_t in_dim = model.layers.front().in_dim;
        const double count = static_cast<double>(sample_count);
        double squared = 0.0;
        double max_residual = 0.0;
        double residual_sum = 0.0;
        std::vector<double> residuals(static_cast<std::size_t>(sample_count));
        for (int index = 0; index < sample_count; ++index) {
            const std::vector<double> input = sample_input(index, sample_count, in_dim);
            const std::vector<double> predicted = model.forward(input);
            double sample_square = 0.0;
            double sample_peak = 0.0;
            for (std::size_t coord = 0; coord < in_dim; ++coord) {
                const double residual = std::abs(predicted[coord] - input[coord]);
                sample_square += residual * residual;
                sample_peak = std::max(sample_peak, residual);
            }
            residuals[static_cast<std::size_t>(index)] = sample_peak;
            squared += sample_square / static_cast<double>(in_dim);
            residual_sum += sample_peak;
            max_residual = std::max(max_residual, sample_peak);
        }
        const double mean_residual = residual_sum / count;
        double moment = 0.0;
        for (double residual : residuals) {
            const double delta = residual - mean_residual;
            moment += delta * delta;
        }
        RegressionScore score;
        score.loss = squared / count;
        score.max_residual = max_residual;
        score.std_residual = std::sqrt(moment / count);
        return score;
    }

    ModelDirective get_directives() const override {
        return ModelDirective{
            "8-in reconstruction, 2-wide ReLU bottleneck, MSE.",
            "BottleneckAutoencoder",
            "c1 step 1-3. c2 follow Grad and raise sparsity. c3 g .05-.08 ints -25..25. Keep the bottleneck nonzero."};
    }

private:
    NeuralModel net_;
};

}  // namespace stevejob::models
