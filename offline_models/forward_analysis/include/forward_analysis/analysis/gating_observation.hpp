#pragma once

#include <cstddef>
#include <vector>

namespace forward_analysis {

struct GatingObservation {
    // Samples on which every first-layer unit outputs zero (empty unless some, but not all, samples are closed).
    std::vector<std::size_t> closed_samples;
    // Samples on which at least one first-layer unit is non-zero.
    std::vector<std::size_t> open_samples;
    // True when a single input coordinate is negative on exactly the closed samples.
    bool has_gate_coordinate = false;
    // That input coordinate (valid only when has_gate_coordinate).
    std::size_t gate_coordinate = 0;
    // True when every deeper layer shows one identical activation row across all closed samples.
    bool deeper_layers_constant = false;

    bool present() const noexcept {
        return !closed_samples.empty();
    }
};

}  // namespace forward_analysis
