#pragma once

#include "forward_analysis/heads/domain_mlp_head_base.hpp"

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Domain D head: predicts per-layer dead fraction, always-active fraction, nonpositive sample fraction, and gating.
class ActivationSparsityHead : public DomainMlpHeadBase {
public:
    explicit ActivationSparsityHead(std::size_t d_model = 256,
                                    double weight = 0.25,
                                    std::uint64_t seed = 3101ULL)
        : DomainMlpHeadBase("activation_sparsity",
                            AnalysisDomain::Activation,
                            d_model,
                            8,
                            weight,
                            false,
                            seed) {}
};

}  // namespace forward_analysis
