#pragma once

#include "bottleneck_ae.hpp"
#include "demo_scale_model.hpp"
#include "forward_analyzed_model.hpp"
#include "mlp_regressor.hpp"

#include <utility>

namespace stevejob::models {

class ArchitectureCatalog {
public:
    template <typename Fn>
    static decltype(auto) make(const std::string& kind, Fn&& fn) {
        if (kind == "bottleneck_ae") {
            BottleneckAutoencoder model;
            return std::forward<Fn>(fn)(model);
        }
        if (kind == "demo_scale") {
            DemoScaleModel model;
            return std::forward<Fn>(fn)(model);
        }
        if (kind == "forward_analyzed_mlp" || kind == "forward_analyzed") {
            ForwardAnalyzedModel model;
            return std::forward<Fn>(fn)(model);
        }
        MlpRegressor model;
        return std::forward<Fn>(fn)(model);
    }

    template <typename Fn>
    static decltype(auto) visit(NeuralNetwork network, Fn&& fn) {
        return make(network.architecture, [&](auto& model) {
            model.adopt(std::move(network));
            return std::forward<Fn>(fn)(model);
        });
    }

    static NeuralNetwork prepare(NeuralNetwork network) {
        return visit(std::move(network), [](auto& model) { return NeuralNetwork(model.network()); });
    }
};

}  // namespace stevejob::models
