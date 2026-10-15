#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/device_buffer.hpp"
#include "forward_analysis/compute/device_parameter.hpp"
#include "forward_analysis/compute/opencl_runtime.hpp"
#include "forward_analysis/math/adam_moment.hpp"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace forward_analysis {

// Every trainable decoder tensor packed into four device buffers (values, gradients, Adam first and second
// moments). Each tensor starts at a multiple of the device's base address alignment so it can be viewed as a
// buffer of its own; the gaps stay zero. One kernel then updates all parameters, and gradients of a leading run
// of tensors are cleared with one fill.
class DeviceParameterArena {
public:
    // Lays the tensors out in the given order with the given (rows, cols) shapes and sizes the buffers; existing
    // contents are discarded and everything starts at zero.
    void layout(const OpenClRuntime& runtime,
                const std::vector<DeviceParameter*>& tensors,
                const std::vector<std::pair<std::size_t, std::size_t>>& shapes) {
        for (DeviceParameter* tensor : tensors_) {
            tensor->release_views();
        }
        tensors_ = tensors;
        const std::size_t align = runtime.alignment_bytes() / sizeof(double);
        std::size_t next = 0;
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            tensors[i]->place(next, shapes[i].first, shapes[i].second);
            next += tensors[i]->size();
            next = (next + align - 1) / align * align;
        }
        size_ = next;
        for (DeviceBuffer<double>* buffer : {&values_, &gradients_, &first_, &second_}) {
            buffer->reserve(runtime, size_);
            buffer->fill(runtime, 0.0, size_);
        }
        for (DeviceParameter* tensor : tensors_) {
            tensor->attach(DeviceBuffer<double>::region(values_, tensor->offset(), tensor->size()),
                           DeviceBuffer<double>::region(gradients_, tensor->offset(), tensor->size()));
        }
    }

    // Copies a host tensor and its Adam moments into the tensor's ranges.
    template <typename Dense>
    void write(const OpenClRuntime& runtime, const DeviceParameter& tensor, const Dense& value, AdamMoment& moment) const {
        moment.ensure_shape(static_cast<Eigen::Index>(tensor.rows()), static_cast<Eigen::Index>(tensor.cols()));
        values_.write(runtime, value.data(), tensor.size(), true, tensor.offset());
        first_.write(runtime, moment.first().data(), tensor.size(), true, tensor.offset());
        second_.write(runtime, moment.second().data(), tensor.size(), true, tensor.offset());
    }

    // Copies the tensor's values and Adam moments back into the host tensor and its AdamMoment.
    template <typename Dense>
    void read(const OpenClRuntime& runtime, const DeviceParameter& tensor, Dense& value, AdamMoment& moment) const {
        const Eigen::Index rows = static_cast<Eigen::Index>(tensor.rows());
        const Eigen::Index cols = static_cast<Eigen::Index>(tensor.cols());
        value.resize(rows, cols);
        values_.read(runtime, value.data(), tensor.size(), tensor.offset());
        Eigen::MatrixXd first(rows, cols);
        Eigen::MatrixXd second(rows, cols);
        first_.read(runtime, first.data(), tensor.size(), tensor.offset());
        second_.read(runtime, second.data(), tensor.size(), tensor.offset());
        moment.assign(first, second);
    }

    // Zeroes the gradients of every tensor from the start of the arena through last (inclusive).
    void zero_gradients_through(const OpenClRuntime& runtime, const DeviceParameter& last) const {
        gradients_.fill(runtime, 0.0, last.offset() + last.size());
    }

    // One Adam step on every tensor with bias corrections for the 1-based step count (as AdamMoment::apply, with
    // the corrections applied through their reciprocals), then decayed's values multiplied by decay_factor.
    void adam(const OpenClRuntime& runtime,
              double learning_rate,
              std::uint64_t step,
              const DeviceParameter& decayed,
              double decay_factor) const {
        const double t = static_cast<double>(step);
        const double correction1 = 1.0 - std::pow(AdamMoment::beta1(), t);
        const double correction2 = 1.0 - std::pow(AdamMoment::beta2(), t);
        runtime.launch("adam_update", size_, 1, size_, learning_rate, 1.0 / correction1, 1.0 / correction2, AdamMoment::beta1(),
                       AdamMoment::beta2(), AdamMoment::epsilon(), decayed.offset(), decayed.offset() + decayed.size(),
                       decay_factor, values_.mem(), gradients_.mem(), first_.mem(), second_.mem());
    }

private:
    // Packed buffers, size_ elements each: values, gradients of the latest pass, Adam first and second moments.
    DeviceBuffer<double> values_;
    DeviceBuffer<double> gradients_;
    DeviceBuffer<double> first_;
    DeviceBuffer<double> second_;
    // Elements in the packed layout, alignment gaps included.
    std::size_t size_ = 0;
    // Tensors placed by the latest layout() (owned by the caller), whose views are released before a new layout.
    std::vector<DeviceParameter*> tensors_;
};

}  // namespace forward_analysis

#endif
