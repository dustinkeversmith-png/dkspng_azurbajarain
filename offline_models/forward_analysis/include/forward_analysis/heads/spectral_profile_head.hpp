#pragma once

#include "forward_analysis/heads/domain_mlp_head_base.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Domain B head: predicts normalized spectral profile and sign balance of the zero-diagonal input covariance.
class SpectralProfileHead : public DomainMlpHeadBase {
public:
    explicit SpectralProfileHead(std::size_t d_model = 256,
                                 double weight = 0.20,
                                 std::uint64_t seed = 3108ULL)
        : DomainMlpHeadBase("spectral_profile",
                            AnalysisDomain::Spectral,
                            d_model,
                            4,
                            weight,
                            false,
                            seed) {}
};

}  // namespace forward_analysis
