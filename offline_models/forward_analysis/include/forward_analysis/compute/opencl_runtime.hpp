#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/decoder_kernel_source.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace forward_analysis {

// One OpenCL GPU with double precision: its context, an in-order command queue and the compiled decoder kernels.
// Commands are queued asynchronously; a blocking read is the only point where the host waits for the device.
class OpenClRuntime {
public:
    // The process-wide runtime, created on first use; null when no GPU with double precision is present.
    static std::shared_ptr<OpenClRuntime> shared() {
        static const std::shared_ptr<OpenClRuntime> instance = create();
        return instance;
    }

    OpenClRuntime(const OpenClRuntime&) = delete;
    OpenClRuntime& operator=(const OpenClRuntime&) = delete;

    ~OpenClRuntime() {
        for (auto& entry : kernels_) {
            clReleaseKernel(entry.second);
        }
        if (program_ != nullptr) {
            clReleaseProgram(program_);
        }
        if (queue_ != nullptr) {
            clReleaseCommandQueue(queue_);
        }
        if (context_ != nullptr) {
            clReleaseContext(context_);
        }
    }

    cl_context context() const noexcept { return context_; }
    cl_command_queue queue() const noexcept { return queue_; }
    // e.g. "NVIDIA GeForce RTX 4070 (NVIDIA CUDA, OpenCL 3.0 CUDA 12.x)"
    const std::string& description() const noexcept { return description_; }
    // Sub-buffer origins must be multiples of this many bytes (CL_DEVICE_MEM_BASE_ADDR_ALIGN), e.g. 512.
    std::size_t alignment_bytes() const noexcept { return alignment_bytes_; }

    static void check(cl_int status, const char* what) {
        if (status != CL_SUCCESS) {
            throw std::runtime_error(std::string("OpenCL error ") + std::to_string(status) + " in " + what);
        }
    }

    // Sets the kernel's arguments in order (cl_mem buffers, integers as cl_int, floating values as cl_double) and
    // queues it over a global range of global0 x global1 work-items (global1 = 1 for a 1-D range).
    template <typename... Args>
    void launch(const char* name, std::size_t global0, std::size_t global1, const Args&... args) const {
        const cl_kernel k = kernel(name);
        cl_uint index = 0;
        (set_arg(k, index++, args), ...);
        const std::size_t global[2] = {global0, global1};
        check(clEnqueueNDRangeKernel(queue_, k, global1 > 1 ? 2U : 1U, nullptr, global, nullptr, 0, nullptr, nullptr),
              name);
    }

    // Queues a row kernel: one work-group of DecoderKernelSource::kRowGroup work-items per row.
    template <typename... Args>
    void launch_rows(const char* name, std::size_t rows, const Args&... args) const {
        launch_groups(name, rows, 1, DecoderKernelSource::kRowGroup, 1, args...);
    }

    // Queues a kernel over groups0 x groups1 work-groups of local0 x local1 work-items.
    template <typename... Args>
    void launch_groups(const char* name,
                       std::size_t groups0,
                       std::size_t groups1,
                       std::size_t local0,
                       std::size_t local1,
                       const Args&... args) const {
        const cl_kernel k = kernel(name);
        cl_uint index = 0;
        (set_arg(k, index++, args), ...);
        const std::size_t global[2] = {groups0 * local0, groups1 * local1};
        const std::size_t local[2] = {local0, local1};
        check(clEnqueueNDRangeKernel(queue_, k, 2U, nullptr, global, local, 0, nullptr, nullptr), name);
    }

    void finish() const { check(clFinish(queue_), "clFinish"); }

private:
    OpenClRuntime() = default;

    static std::shared_ptr<OpenClRuntime> create() {
        cl_device_id device = nullptr;
        cl_platform_id platform = nullptr;
        if (!select_device(device, platform)) {
            return nullptr;
        }
        std::shared_ptr<OpenClRuntime> runtime(new OpenClRuntime());
        cl_int status = CL_SUCCESS;
        const cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                                    reinterpret_cast<cl_context_properties>(platform), 0};
        runtime->context_ = clCreateContext(properties, 1, &device, nullptr, nullptr, &status);
        check(status, "clCreateContext");
        runtime->queue_ = clCreateCommandQueue(runtime->context_, device, 0, &status);
        check(status, "clCreateCommandQueue");
        runtime->description_ = device_string(device, CL_DEVICE_NAME) + " (" +
                                platform_string(platform, CL_PLATFORM_NAME) + ", " +
                                device_string(device, CL_DEVICE_VERSION) + ")";
        cl_uint alignment_bits = 0;
        check(clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(alignment_bits), &alignment_bits, nullptr),
              "clGetDeviceInfo");
        runtime->alignment_bytes_ = std::max<std::size_t>(sizeof(cl_double), alignment_bits / 8);
        runtime->build(device);
        return runtime;
    }

    // Among all GPUs with double-precision support, the one with the most global memory.
    static bool select_device(cl_device_id& chosen, cl_platform_id& chosen_platform) {
        cl_uint platform_count = 0;
        if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
            return false;
        }
        std::vector<cl_platform_id> platforms(platform_count);
        check(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
        cl_ulong best_memory = 0;
        for (cl_platform_id platform : platforms) {
            cl_uint device_count = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) != CL_SUCCESS ||
                device_count == 0) {
                continue;
            }
            std::vector<cl_device_id> devices(device_count);
            check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr), "clGetDeviceIDs");
            for (cl_device_id device : devices) {
                cl_device_fp_config fp64 = 0;
                cl_ulong memory = 0;
                clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64), &fp64, nullptr);
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memory), &memory, nullptr);
                if (fp64 != 0 && memory > best_memory) {
                    best_memory = memory;
                    chosen = device;
                    chosen_platform = platform;
                }
            }
        }
        return chosen != nullptr;
    }

    void build(cl_device_id device) {
        cl_int status = CL_SUCCESS;
        const char* source = DecoderKernelSource::text();
        program_ = clCreateProgramWithSource(context_, 1, &source, nullptr, &status);
        check(status, "clCreateProgramWithSource");
        const std::string options = "-cl-std=CL1.2 -DROW_GROUP=" + std::to_string(DecoderKernelSource::kRowGroup);
        const cl_int built = clBuildProgram(program_, 1, &device, options.c_str(), nullptr, nullptr);
        if (built != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            throw std::runtime_error("OpenCL decoder kernels failed to compile (status " + std::to_string(built) +
                                     "):\n" + log);
        }
        cl_uint kernel_count = 0;
        check(clCreateKernelsInProgram(program_, 0, nullptr, &kernel_count), "clCreateKernelsInProgram");
        std::vector<cl_kernel> created(kernel_count);
        check(clCreateKernelsInProgram(program_, kernel_count, created.data(), nullptr), "clCreateKernelsInProgram");
        for (cl_kernel k : created) {
            std::size_t name_size = 0;
            clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &name_size);
            std::string name(name_size, '\0');
            clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, name_size, name.data(), nullptr);
            name.resize(name.find('\0') == std::string::npos ? name.size() : name.find('\0'));
            kernels_[name] = k;
        }
    }

    cl_kernel kernel(const char* name) const {
        const auto found = kernels_.find(name);
        if (found == kernels_.end()) {
            throw std::invalid_argument(std::string("unknown OpenCL kernel ") + name);
        }
        return found->second;
    }

    template <typename T>
    static void set_arg(cl_kernel k, cl_uint index, const T& value) {
        if constexpr (std::is_same_v<T, cl_mem>) {
            check(clSetKernelArg(k, index, sizeof(cl_mem), &value), "clSetKernelArg");
        } else if constexpr (std::is_floating_point_v<T>) {
            const cl_double v = static_cast<cl_double>(value);
            check(clSetKernelArg(k, index, sizeof(cl_double), &v), "clSetKernelArg");
        } else {
            static_assert(std::is_integral_v<T>, "kernel arguments are cl_mem, integers or floating values");
            const cl_int v = static_cast<cl_int>(value);
            check(clSetKernelArg(k, index, sizeof(cl_int), &v), "clSetKernelArg");
        }
    }

    static std::string device_string(cl_device_id device, cl_device_info what) {
        std::size_t size = 0;
        clGetDeviceInfo(device, what, 0, nullptr, &size);
        std::string text(size, '\0');
        clGetDeviceInfo(device, what, size, text.data(), nullptr);
        return text.substr(0, text.find('\0'));
    }

    static std::string platform_string(cl_platform_id platform, cl_platform_info what) {
        std::size_t size = 0;
        clGetPlatformInfo(platform, what, 0, nullptr, &size);
        std::string text(size, '\0');
        clGetPlatformInfo(platform, what, size, text.data(), nullptr);
        return text.substr(0, text.find('\0'));
    }

    // OpenCL context owning every buffer, program and queue on the chosen device.
    cl_context context_ = nullptr;
    // In-order queue: commands run in submission order, so no explicit events are needed between kernels.
    cl_command_queue queue_ = nullptr;
    // Program built from DecoderKernelSource::text().
    cl_program program_ = nullptr;
    // Kernel handles by function name, e.g. "layer_norm_rows".
    std::map<std::string, cl_kernel> kernels_;
    // Device name, platform and OpenCL version, reported in run settings.
    std::string description_;
    // CL_DEVICE_MEM_BASE_ADDR_ALIGN in bytes.
    std::size_t alignment_bytes_ = sizeof(cl_double);
};

}  // namespace forward_analysis

#endif
