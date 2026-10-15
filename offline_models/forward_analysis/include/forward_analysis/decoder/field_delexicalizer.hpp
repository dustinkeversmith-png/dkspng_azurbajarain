#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/slot_binding.hpp"
#include "forward_analysis/decoder/batch_statement_delexicalizer.hpp"
#include "forward_analysis/decoder/observation_delexicalizer.hpp"
#include "forward_analysis/decoder/shape_slot_lexicon.hpp"

#include <string>
#include <vector>

namespace forward_analysis {

// Slot-token lexemes of one training field under the selected SlotBinding.
struct FieldDelexicalizer {
    static std::vector<std::string> delexicalize(const std::string& text, const NetworkObservations& measured, SlotBinding binding) {
        return binding == SlotBinding::kBatchStatistics
                   ? BatchStatementDelexicalizer::delexicalize(text)
                   : ObservationDelexicalizer::delexicalize(text, ShapeSlotLexicon::from(measured));
    }
};

}  // namespace forward_analysis
