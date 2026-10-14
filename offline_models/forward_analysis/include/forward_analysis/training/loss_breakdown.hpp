#pragma once

#include "forward_analysis/config/distiller_config.hpp"

namespace forward_analysis {

struct LossBreakdown {
    // Mean per-token cross-entropy of the decoder over the four observation fields (0 while the decoder is frozen).
    double token_cross_entropy = 0.0;
    // Aux1 observation-descriptor regression loss.
    double aux1_observation_loss = 0.0;
    // Aux2 eigen-sign loss.
    double aux2_eigen_sign_loss = 0.0;
    // token_cross_entropy + aux1_weight * aux1 + aux2_weight * aux2.
    double total_loss = 0.0;

    static LossBreakdown combine(double ce_loss,
                                 double aux1_loss,
                                 double aux2_loss,
                                 const DistillerConfig& config) noexcept {
        LossBreakdown result;
        result.token_cross_entropy = ce_loss;
        result.aux1_observation_loss = aux1_loss;
        result.aux2_eigen_sign_loss = aux2_loss;
        result.total_loss = ce_loss + (config.aux1_observation_weight * aux1_loss) +
                            (config.aux2_eigen_sign_weight * aux2_loss);
        return result;
    }
};

}  // namespace forward_analysis
