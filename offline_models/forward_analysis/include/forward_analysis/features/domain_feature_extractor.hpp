#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/domains/analysis_domain.hpp"
#include "forward_analysis/features/feature_descriptor.hpp"
#include "forward_analysis/features/feature_normalization_strategy.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <string>
#include <vector>

namespace forward_analysis {

// Interface for modular, domain-specific feature extractors registered in FeatureExtractorRegistry.
class DomainFeatureExtractor {
public:
    virtual ~DomainFeatureExtractor() = default;

    // Unique identifier of this extractor module.
    virtual std::string extractor_id() const = 0;

    // Primary mathematical analysis domain of this extractor.
    virtual AnalysisDomain domain() const = 0;

    // Required telemetry tensors (e.g. "inputs", "layer_activations", "predictions", "targets").
    virtual std::vector<std::string> required_tensors() const = 0;

    // Default normalization strategy used by this extractor.
    virtual FeatureNormalizationStrategy normalization_strategy() const noexcept = 0;

    // Relative computational cost of running this extractor.
    virtual double computational_cost() const noexcept = 0;

    // Numerical validity check on the input telemetry before extraction.
    virtual bool is_valid(const ForwardTelemetryTensor& telemetry) const noexcept {
        return telemetry.num_samples() > 0 && !telemetry.layer_activations.empty() &&
               telemetry.inputs.allFinite() && telemetry.predictions.allFinite() &&
               telemetry.targets.allFinite();
    }

    // Extracts self-describing FeatureDescriptor objects from the telemetry.
    virtual std::vector<FeatureDescriptor> extract(const ForwardTelemetryTensor& telemetry,
                                                   const DistillerConfig& config) const = 0;
};

}  // namespace forward_analysis
