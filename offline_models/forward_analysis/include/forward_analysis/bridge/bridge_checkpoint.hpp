#pragma once

#include "forward_analysis/decoder/autoregressive_causal_decoder.hpp"
#include "forward_analysis/encoder/channel_sequence_formatter.hpp"
#include "forward_analysis/encoder/perceiver_io_encoder.hpp"
#include "forward_analysis/heads/auxiliary_alignment_heads.hpp"

namespace forward_analysis {

// Full copy of every trainable component of the bridge (weights and optimizer state) at one epoch.
struct BridgeCheckpoint {
    // Channel-feature adapter.
    ChannelSequenceFormatter formatter;
    // Perceiver IO encoder producing Z.
    PerceiverIOEncoder perceiver;
    // Aux1 / Aux2 heads.
    AuxiliaryAlignmentHeads aux_heads;
    // Cross-attending causal decoder with its vocabulary.
    AutoregressiveCausalDecoder decoder;
};

}  // namespace forward_analysis
