#pragma once

#include <cstddef>
#include <string>

namespace stevejob::models {

struct SliceRef {
    std::string layer_name;
    std::string slice_id;
    std::size_t layer_index = 0;
    std::size_t row_offset = 0;
    std::size_t col_offset = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

}  // namespace stevejob::models
