#pragma once

#include "forward_analysis/heads/domain_mlp_head_base.hpp"

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Domain B head: predicts per-layer normalized numerical rank and effective rank ratios.
class NumericalRankHead : public DomainMlpHeadBase {
public:
    explicit NumericalRankHead(std::size_t d_model = 256,
                               double weight = 0.25,
                               std::uint64_t seed = 3102ULL)
        : DomainMlpHeadBase("numerical_rank",
                            AnalysisDomain::Spectral,
                            d_model,
                            4,
                            weight,
                            false,
                            seed) {}
};

}  // namespace forward_analysis
