#pragma once

#include <cstddef>

namespace forward_analysis {

// Token families projected into the Perceiver encoder sequence.
enum class TokenFamily : std::size_t {
    Channel = 0,
    Sample = 1,
    Relationship = 2,
    Layer = 3,
    Global = 4
};

}  // namespace forward_analysis
