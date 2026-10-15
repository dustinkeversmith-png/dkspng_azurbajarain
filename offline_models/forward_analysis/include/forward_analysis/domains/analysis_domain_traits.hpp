#pragma once

#include "forward_analysis/domains/analysis_domain.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace forward_analysis {

// Metadata and enumeration helpers for AnalysisDomain.
struct AnalysisDomainTraits {
    // Total number of mathematical analysis domains.
    static constexpr std::size_t kDomainCount = 6;

    static constexpr std::array<AnalysisDomain, kDomainCount> all() noexcept {
        return {AnalysisDomain::Statistical, AnalysisDomain::Spectral,   AnalysisDomain::Geometric,
                AnalysisDomain::Activation,  AnalysisDomain::Functional, AnalysisDomain::Structural};
    }

    static constexpr std::size_t index(AnalysisDomain domain) noexcept {
        return static_cast<std::size_t>(domain);
    }

    static std::string name(AnalysisDomain domain) {
        switch (domain) {
            case AnalysisDomain::Statistical:
                return "statistical";
            case AnalysisDomain::Spectral:
                return "spectral";
            case AnalysisDomain::Geometric:
                return "geometric";
            case AnalysisDomain::Activation:
                return "activation";
            case AnalysisDomain::Functional:
                return "functional";
            case AnalysisDomain::Structural:
                return "structural";
        }
        return "unknown";
    }
};

}  // namespace forward_analysis
