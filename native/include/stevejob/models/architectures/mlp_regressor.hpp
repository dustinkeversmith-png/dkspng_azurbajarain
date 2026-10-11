#pragma once

#include "stevejob/models/model_template.hpp"

namespace stevejob::models {

class MlpRegressor : public ModelTemplate {
public:
    MlpRegressor() : net_("MlpRegressor") {
        net_.architecture = "mlp_regressor";
        net_.add_layer("layer1", 4, 8, ActivationType::ReLU)
            .add_layer("layer2", 8, 4, ActivationType::ReLU)
            .add_layer("head", 4, 1, ActivationType::Linear)
            .set_loss(LossType::MeanSquaredError);
        net_.initialize_default_weights();
    }

    void adopt(NeuralModel model) { net_ = std::move(model); }

    NeuralModel& network() override { return net_; }
    const NeuralModel& network() const override { return net_; }

    std::vector<double> forward(const std::vector<double>& input) const override { return net_.forward(input); }

    RegressionScore evaluate(int sample_count = 50) const override { return net_.evaluate_mse(sample_count); }

    RegressionScore score_network(const NeuralModel& model, int sample_count = 50) const override {
        return model.evaluate_mse(sample_count);
    }

    ModelDirective get_directives() const override {
        return ModelDirective{
            "4-in regression. ReLU hidden, linear head.",
            "MlpRegressor",
            "c1 g .02-.04 step 1-3. c2 follow Grad. c3 g .04-.07 step<=8 ints -25..25."};
    }

private:
    NeuralModel net_;
};

}  // namespace stevejob::models
