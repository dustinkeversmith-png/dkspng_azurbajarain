#pragma once

#include "forward_analysis/heads/domain_mlp_head_base.hpp"

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Domain A head: predicts input-to-hidden association strength, nonzero association indicator, and driver stats.
class FeatureAssociationHead : public DomainMlpHeadBase {
public:
    explicit FeatureAssociationHead(std::size_t d_model = 256,
                                    double weight = 0.25,
                                    std::uint64_t seed = 3106ULL)
        : DomainMlpHeadBase("feature_association",
                            AnalysisDomain::Statistical,
                            d_model,
                            5,
                            weight,
                            false,
                            seed) {}
};

}  // namespace forward_analysis
