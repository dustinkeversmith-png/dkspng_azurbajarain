#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/opencl_runtime.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

namespace forward_analysis {

// Device memory for count elements of T (cl_double or cl_int). Grows on reserve() and never shrinks, so step
// buffers sized for the longest field are reused by every shorter one. region() makes a view of part of another
// buffer (an OpenCL sub-buffer) that kernels and CLBlast use like a buffer of its own.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    // View of count elements of parent starting at element offset; offset * sizeof(T) must be a multiple of the
    // device's base address alignment. The view must be released before parent grows or is destroyed.
    static DeviceBuffer region(const DeviceBuffer& parent, std::size_t offset, std::size_t count) {
        const cl_buffer_region span{sizeof(T) * offset, sizeof(T) * std::max<std::size_t>(1, count)};
        cl_int status = CL_SUCCESS;
        DeviceBuffer view;
        view.mem_ = clCreateSubBuffer(parent.mem_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &span, &status);
        OpenClRuntime::check(status, "clCreateSubBuffer");
        view.capacity_ = count;
        return view;
    }

    // Releases the device memory (or view).
    void reset() noexcept {
        if (mem_ != nullptr) {
            clReleaseMemObject(mem_);
            mem_ = nullptr;
        }
        capacity_ = 0;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        swap(other);
        return *this;
    }
    ~DeviceBuffer() { reset(); }

    // Ensures room for count elements; existing contents are discarded when the buffer has to grow.
    void reserve(const OpenClRuntime& runtime, std::size_t count) {
        if (count <= capacity_ && mem_ != nullptr) {
            return;
        }
        if (mem_ != nullptr) {
            clReleaseMemObject(mem_);
            mem_ = nullptr;
        }
        cl_int status = CL_SUCCESS;
        mem_ = clCreateBuffer(runtime.context(), CL_MEM_READ_WRITE, sizeof(T) * std::max<std::size_t>(1, count), nullptr,
                              &status);
        OpenClRuntime::check(status, "clCreateBuffer");
        capacity_ = count;
    }

    // Copies count host elements to the buffer starting at element offset. A non-blocking write requires data to
    // stay valid until the next blocking read on the same queue.
    void write(const OpenClRuntime& runtime, const T* data, std::size_t count, bool blocking, std::size_t offset = 0) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueWriteBuffer(runtime.queue(), mem_, blocking ? CL_TRUE : CL_FALSE, sizeof(T) * offset,
                                                  sizeof(T) * count, data, 0, nullptr, nullptr),
                             "clEnqueueWriteBuffer");
    }

    // Blocking copy of count elements starting at element offset to the host; waits for every command queued before it.
    void read(const OpenClRuntime& runtime, T* data, std::size_t count, std::size_t offset = 0) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueReadBuffer(runtime.queue(), mem_, CL_TRUE, sizeof(T) * offset, sizeof(T) * count, data,
                                                 0, nullptr, nullptr),
                             "clEnqueueReadBuffer");
    }

    // Sets count elements starting at element offset to value.
    void fill(const OpenClRuntime& runtime, T value, std::size_t count, std::size_t offset = 0) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueFillBuffer(runtime.queue(), mem_, &value, sizeof(T), sizeof(T) * offset,
                                                 sizeof(T) * count, 0, nullptr, nullptr),
                             "clEnqueueFillBuffer");
    }

    // Device-side copy of the first count elements of source into this buffer.
    void copy_from(const OpenClRuntime& runtime, const DeviceBuffer& source, std::size_t count) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueCopyBuffer(runtime.queue(), source.mem_, mem_, 0, 0, sizeof(T) * count, 0, nullptr,
                                                 nullptr),
                             "clEnqueueCopyBuffer");
    }

    cl_mem mem() const noexcept { return mem_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    void swap(DeviceBuffer& other) noexcept {
        std::swap(mem_, other.mem_);
        std::swap(capacity_, other.capacity_);
    }

    // OpenCL buffer object, null until the first reserve().
    cl_mem mem_ = nullptr;
    // Number of T elements the buffer can hold.
    std::size_t capacity_ = 0;
};

}  // namespace forward_analysis

#endif
