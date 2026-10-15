#pragma once

#include "forward_analysis/heads/domain_mlp_head_base.hpp"

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Domain F head: predicts layer count (1-layer vs 2-layer) and normalized hidden/input/output dimensions.
class LayerStructureHead : public DomainMlpHeadBase {
public:
    explicit LayerStructureHead(std::size_t d_model = 256,
                                double weight = 0.35,
                                std::uint64_t seed = 3104ULL)
        : DomainMlpHeadBase("layer_structure",
                            AnalysisDomain::Structural,
                            d_model,
                            6,
                            weight,
                            false,
                            seed) {}
};

}  // namespace forward_analysis
