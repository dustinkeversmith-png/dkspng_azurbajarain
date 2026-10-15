#pragma once

#include "forward_analysis/features/domain_feature_extractor.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Example user-defined feature extractor registered into FeatureExtractorRegistry without modifying PerceiverIOEncoder.
class CustomProbeExtractor final : public DomainFeatureExtractor {
public:
    std::string extractor_id() const override { return "custom.probe"; }
    AnalysisDomain domain() const override { return AnalysisDomain::Statistical; }
    std::vector<std::string> required_tensors() const override { return {"inputs"}; }
    FeatureNormalizationStrategy normalization_strategy() const noexcept override {
        return FeatureNormalizationStrategy::LogScale;
    }
    double computational_cost() const noexcept override { return 1.0; }
    std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                           const DistillerConfig&) const override {
        // Mean squared input energy.
        const double energy = telemetry.inputs.squaredNorm() /
                              static_cast<double>(std::max<Eigen::Index>(1, telemetry.inputs.size()));
        // Descriptor for custom energy feature.
        FeatureDescriptor desc;
        desc.id = "custom.probe.energy";
        desc.domain = AnalysisDomain::Statistical;
        desc.type = FeatureValueType::Scalar;
        desc.normalization = FeatureNormalizationStrategy::LogScale;
        desc.required_tensors = {"inputs"};
        desc.values = {FeatureDescriptor::normalize_value(energy, FeatureNormalizationStrategy::LogScale)};
        desc.shape = {1};
        desc.valid = true;
        return {desc};
    }
};

}  // namespace forward_analysis
