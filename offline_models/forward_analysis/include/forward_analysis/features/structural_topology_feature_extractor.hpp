#pragma once

#include "forward_analysis/features/domain_feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Domain F: Structural and topological analysis (depth, widths, layer boundaries, bottlenecks, channel roles).
class StructuralTopologyFeatureExtractor : public DomainFeatureExtractor {
public:
    std::string extractor_id() const override { return "structural_topology_analysis"; }

    AnalysisDomain domain() const override { return AnalysisDomain::Structural; }

    std::vector<std::string> required_tensors() const override {
        return {"inputs", "layer_activations", "targets"};
    }

    FeatureNormalizationStrategy normalization_strategy() const noexcept override {
        return FeatureNormalizationStrategy::UnitInterval;
    }

    double computational_cost() const noexcept override { return 0.5; }

    std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                           const DistillerConfig& /*config*/) const override {
        // Output feature list for the structural and topological domain.
        std::vector<FeatureDescriptor> out;
        if (!is_valid(telemetry)) {
            // Invalid placeholder descriptor.
            FeatureDescriptor invalid;
            invalid.id = "structural.invalid";
            invalid.domain = AnalysisDomain::Structural;
            invalid.valid = false;
            out.push_back(invalid);
            return out;
        }

        // Telemetry shape metadata.
        const TelemetryShape shape = telemetry.shape();
        // Number of hidden layers in the network.
        const std::size_t num_layers = shape.hidden_widths.size();
        // Width of the first hidden layer.
        const double w1 = num_layers >= 1 ? static_cast<double>(shape.hidden_widths[0]) : 0.0;
        // Width of the second hidden layer (0 when 1-layer).
        const double w2 = num_layers >= 2 ? static_cast<double>(shape.hidden_widths[1]) : 0.0;
        // Bottleneck ratio between final hidden layer and first hidden layer.
        const double bottleneck = w1 > 0.0 ? (num_layers >= 2 ? w2 / w1 : 1.0) : 1.0;

        // Topological and architectural boundary descriptor.
        FeatureDescriptor topo;
        topo.id = "structural.network_topology";
        topo.domain = AnalysisDomain::Structural;
        topo.type = FeatureValueType::Vector;
        topo.normalization = FeatureNormalizationStrategy::UnitInterval;
        topo.required_tensors = {"inputs", "layer_activations", "targets"};
        topo.shape = {8};
        topo.computational_cost = 0.5;
        topo.values = {
            num_layers == 1 ? 1.0 : 0.0,
            num_layers >= 2 ? 1.0 : 0.0,
            std::min(1.0, static_cast<double>(shape.num_samples) / 16.0),
            std::min(1.0, static_cast<double>(shape.input_dim) / 32.0),
            std::min(1.0, w1 / 48.0),
            std::min(1.0, w2 / 48.0),
            std::min(1.0, static_cast<double>(shape.output_dim) / 16.0),
            std::min(1.0, bottleneck)};
        out.push_back(topo);

        // Second-layer boundary descriptor (marked valid = false on 1-layer networks).
        FeatureDescriptor second_layer;
        second_layer.id = "structural.second_layer_topology";
        second_layer.domain = AnalysisDomain::Structural;
        second_layer.type = FeatureValueType::Vector;
        second_layer.normalization = FeatureNormalizationStrategy::UnitInterval;
        second_layer.required_tensors = {"layer_activations"};
        second_layer.shape = {2};
        second_layer.valid = num_layers >= 2;
        second_layer.missing_fill_value = 0.0;
        second_layer.computational_cost = 0.1;
        second_layer.values = {std::min(1.0, w2 / 48.0), std::min(1.0, bottleneck)};
        out.push_back(second_layer);

        return out;
    }
};

}  // namespace forward_analysis
