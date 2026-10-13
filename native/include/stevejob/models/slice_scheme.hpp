#pragma once

#include <cstddef>

namespace stevejob::models {

struct SliceScheme {
    std::size_t block = 4;
    std::size_t long_side = 4;
    std::size_t short_side = 4;
};

}  // namespace stevejob::models
