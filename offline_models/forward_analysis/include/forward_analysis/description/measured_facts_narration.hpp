#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/description/batch_statistics_narrator.hpp"
#include "forward_analysis/description/observation_narrator.hpp"
#include "forward_analysis/description/observation_text.hpp"

namespace forward_analysis {

// The narrator selected by config.narration_style.
struct MeasuredFactsNarration {
    static ObservationText describe(const NetworkObservations& obs, const DistillerConfig& config) {
        return config.narration_style == NarrationStyle::kBatchStatistics ? BatchStatisticsNarrator::describe(obs, config)
                                                                         : ObservationNarrator::describe(obs);
    }
};

}  // namespace forward_analysis
