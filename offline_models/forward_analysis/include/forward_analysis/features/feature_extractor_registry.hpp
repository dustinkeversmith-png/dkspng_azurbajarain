#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/domains/analysis_domain.hpp"
#include "forward_analysis/domains/analysis_domain_traits.hpp"
#include "forward_analysis/features/activation_pathway_feature_extractor.hpp"
#include "forward_analysis/features/domain_feature_extractor.hpp"
#include "forward_analysis/features/feature_descriptor.hpp"
#include "forward_analysis/features/functional_mapping_feature_extractor.hpp"
#include "forward_analysis/features/geometric_feature_extractor.hpp"
#include "forward_analysis/features/spectral_feature_extractor.hpp"
#include "forward_analysis/features/statistical_feature_extractor.hpp"
#include "forward_analysis/features/structural_topology_feature_extractor.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace forward_analysis {

// Registry of modular, typed feature extractors spanning all mathematical analysis domains.
// Additional extractors can be registered at runtime without editing the Perceiver encoder.
class FeatureExtractorRegistry {
public:
    FeatureExtractorRegistry() {
        register_default_extractors();
    }

    // Registers the six core mathematical domain feature extractors.
    void register_default_extractors() {
        extractors_.clear();
        register_extractor(std::make_shared<StatisticalFeatureExtractor>());
        register_extractor(std::make_shared<SpectralFeatureExtractor>());
        register_extractor(std::make_shared<GeometricFeatureExtractor>());
        register_extractor(std::make_shared<ActivationPathwayFeatureExtractor>());
        register_extractor(std::make_shared<FunctionalMappingFeatureExtractor>());
        register_extractor(std::make_shared<StructuralTopologyFeatureExtractor>());
    }

    // Appends a custom or domain-specific feature extractor to the registry.
    void register_extractor(std::shared_ptr<DomainFeatureExtractor> extractor) {
        if (extractor != nullptr) {
            extractors_.push_back(std::move(extractor));
        }
    }

    // Number of registered feature extractors.
    std::size_t size() const noexcept { return extractors_.size(); }

    // Read-only access to the registered extractors.
    const std::vector<std::shared_ptr<DomainFeatureExtractor>>& extractors() const noexcept {
        return extractors_;
    }

    // Mathematical analysis domains that are valid and active on a given telemetry batch.
    std::vector<AnalysisDomain> active_domains(const ForwardTelemetryTensor& telemetry) const {
        // Ordered set of active domains.
        std::set<AnalysisDomain> domains;
        for (const std::shared_ptr<DomainFeatureExtractor>& ext : extractors_) {
            if (ext->is_valid(telemetry)) {
                domains.insert(ext->domain());
            }
        }
        return std::vector<AnalysisDomain>(domains.begin(), domains.end());
    }

    // Extracts all self-describing FeatureDescriptors across all registered extractors.
    std::vector<FeatureDescriptor> extract_all(const ForwardTelemetryTensor& telemetry,
                                               const DistillerConfig& config = DistillerConfig{}) const {
        // Accumulated feature descriptors across all registered extractors.
        std::vector<FeatureDescriptor> all;
        for (const std::shared_ptr<DomainFeatureExtractor>& ext : extractors_) {
            // Features produced by a single domain extractor.
            std::vector<FeatureDescriptor> part = ext->extract(telemetry, config);
            all.insert(all.end(), part.begin(), part.end());
        }
        return all;
    }

    // Extracts FeatureDescriptors belonging to one specific mathematical domain.
    std::vector<FeatureDescriptor> extract_domain(AnalysisDomain domain,
                                                  const ForwardTelemetryTensor& telemetry,
                                                  const DistillerConfig& config = DistillerConfig{}) const {
        // Filtered feature descriptors for `domain`.
        std::vector<FeatureDescriptor> filtered;
        for (const std::shared_ptr<DomainFeatureExtractor>& ext : extractors_) {
            if (ext->domain() == domain) {
                // Features produced by this matching extractor.
                std::vector<FeatureDescriptor> part = ext->extract(telemetry, config);
                filtered.insert(filtered.end(), part.begin(), part.end());
            }
        }
        return filtered;
    }

    // Flattens all extracted FeatureDescriptor values into one dense vector for inspection or testing.
    Eigen::VectorXd flatten_values(const ForwardTelemetryTensor& telemetry,
                                   const DistillerConfig& config = DistillerConfig{}) const {
        // All extracted feature descriptors.
        const std::vector<FeatureDescriptor> descriptors = extract_all(telemetry, config);
        // Flattened scalar entries.
        std::vector<double> flat;
        for (const FeatureDescriptor& d : descriptors) {
            for (double v : d.values) {
                flat.push_back(d.valid ? v : d.missing_fill_value);
            }
        }
        return Eigen::Map<const Eigen::VectorXd>(flat.data(), static_cast<Eigen::Index>(flat.size()));
    }

    // Total relative computational cost across all registered extractors.
    double total_computational_cost() const noexcept {
        // Sum of extractor cost estimates.
        double cost = 0.0;
        for (const std::shared_ptr<DomainFeatureExtractor>& ext : extractors_) {
            cost += ext->computational_cost();
        }
        return cost;
    }

private:
    // Ordered list of registered domain feature extractors.
    std::vector<std::shared_ptr<DomainFeatureExtractor>> extractors_;
};

}  // namespace forward_analysis
