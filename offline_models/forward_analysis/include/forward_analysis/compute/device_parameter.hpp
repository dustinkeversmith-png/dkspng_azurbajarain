#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/device_buffer.hpp"

#include <CL/cl.h>

#include <cstddef>
#include <utility>

namespace forward_analysis {

// One trainable tensor inside a DeviceParameterArena: where it starts in the packed buffers, its host shape, and
// views of its values and gradient (column-major, same layout as the host tensor).
class DeviceParameter {
public:
    // Records the tensor's place in the arena and drops any views of a previous layout.
    void place(std::size_t offset, std::size_t rows, std::size_t cols) {
        release_views();
        offset_ = offset;
        rows_ = rows;
        cols_ = cols;
    }

    // Takes the views of this tensor's value and gradient ranges.
    void attach(DeviceBuffer<double> value_view, DeviceBuffer<double> gradient_view) {
        value_ = std::move(value_view);
        gradient_ = std::move(gradient_view);
    }

    void release_views() noexcept {
        value_.reset();
        gradient_.reset();
    }

    cl_mem value() const noexcept { return value_.mem(); }
    cl_mem gradient() const noexcept { return gradient_.mem(); }
    std::size_t offset() const noexcept { return offset_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return rows_ * cols_; }

private:
    // First element of the tensor in the arena's packed buffers.
    std::size_t offset_ = 0;
    // Rows and columns of the host tensor (a vector has one column).
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    // View of the tensor's current values.
    DeviceBuffer<double> value_;
    // View of dLoss/dvalue of the latest training pass.
    DeviceBuffer<double> gradient_;
};

}  // namespace forward_analysis

#endif
