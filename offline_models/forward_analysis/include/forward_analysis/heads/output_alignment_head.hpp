#pragma once

#include "forward_analysis/heads/domain_mlp_head_base.hpp"

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Domain E head: predicts output sign preservation, order preservation, amplitude/range ratios, and correlations.
class OutputAlignmentHead : public DomainMlpHeadBase {
public:
    explicit OutputAlignmentHead(std::size_t d_model = 256,
                                 double weight = 0.30,
                                 std::uint64_t seed = 3105ULL)
        : DomainMlpHeadBase("output_alignment",
                            AnalysisDomain::Functional,
                            d_model,
                            9,
                            weight,
                            false,
                            seed) {}
};

}  // namespace forward_analysis
