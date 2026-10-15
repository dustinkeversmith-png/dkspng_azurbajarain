#pragma once

#include "forward_analysis/heads/domain_mlp_head_base.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Domain C head: predicts narrow vs multidirectional representation and ray collapse with class-balanced weights.
class NarrowRepresentationHead : public DomainMlpHeadBase {
public:
    explicit NarrowRepresentationHead(std::size_t d_model = 256,
                                      double weight = 0.35,
                                      std::uint64_t seed = 3103ULL)
        : DomainMlpHeadBase("narrow_representation",
                            AnalysisDomain::Geometric,
                            d_model,
                            4,
                            weight,
                            true,
                            seed) {
        // Training split has ~23.7% narrow (1) vs ~76.3% multidirectional (0); balance positive class weight.
        Eigen::VectorXd pos_w(4);
        pos_w << 2.5, 2.5, 2.5, 2.0;
        // Negative-class weights.
        Eigen::VectorXd neg_w = Eigen::VectorXd::Ones(4);
        set_class_weights(pos_w, neg_w);
    }
};

}  // namespace forward_analysis
