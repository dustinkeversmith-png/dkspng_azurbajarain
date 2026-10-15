#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/decoder/decoder_tensor_refs.hpp"

#if FORWARD_ANALYSIS_WITH_OPENCL
#include "forward_analysis/compute/opencl_runtime.hpp"
#include "forward_analysis/decoder/decoder_device_mirror.hpp"
#endif

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace forward_analysis {

// The decoder's optional device copy and which side holds the newest parameters. Copying a decoder copies host
// tensors only: a copy starts without device memory, and an assignment keeps the target's device memory but
// re-uploads before its next device pass. Copying while the device holds newer values than the host throws,
// because the host tensors being copied would be stale.
class DecoderDeviceSlot {
public:
    DecoderDeviceSlot() = default;
    DecoderDeviceSlot(const DecoderDeviceSlot& other) { other.require_host_current(); }
    DecoderDeviceSlot& operator=(const DecoderDeviceSlot& other) {
        other.require_host_current();
        if (this != &other) {
            host_newer_ = true;
            device_newer_ = false;
        }
        return *this;
    }
    DecoderDeviceSlot(DecoderDeviceSlot&&) noexcept = default;
    DecoderDeviceSlot& operator=(DecoderDeviceSlot&&) noexcept = default;

    // True when decoder passes run on a device: config.use_gpu is set, the library was built with
    // FORWARD_ANALYSIS_WITH_OPENCL and a GPU with double precision is present (probed once).
    bool active(const DistillerConfig& config) {
#if FORWARD_ANALYSIS_WITH_OPENCL
        if (!config.use_gpu) {
            return false;
        }
        if (mirror_ == nullptr && !probed_) {
            probed_ = true;
            const std::shared_ptr<OpenClRuntime> runtime = OpenClRuntime::shared();
            if (runtime != nullptr) {
                mirror_ = std::make_unique<DecoderDeviceMirror>(runtime, config);
                host_newer_ = true;
            }
        }
        return mirror_ != nullptr;
#else
        (void)config;
        return false;
#endif
    }

    // e.g. "NVIDIA GeForce RTX 4070 (NVIDIA CUDA, OpenCL 3.0 CUDA 12.x)", or "cpu" without a device copy.
    std::string description() const {
#if FORWARD_ANALYSIS_WITH_OPENCL
        if (mirror_ != nullptr) {
            return mirror_->device_description();
        }
#endif
        return "cpu";
    }

    // Records that the host tensors changed (vocabulary growth, assignment); the next device pass re-uploads.
    // Only valid when the host was current beforehand.
    void mark_host_newer() noexcept {
        host_newer_ = true;
        device_newer_ = false;
    }

    // Brings the host tensors up to date when the device holds newer values.
    void synchronize_host(const DecoderTensorRefs& refs) {
#if FORWARD_ANALYSIS_WITH_OPENCL
        if (device_newer_ && mirror_ != nullptr) {
            mirror_->download(refs);
        }
#else
        (void)refs;
#endif
        device_newer_ = false;
    }

    double eval_loss(const DecoderTensorRefs& refs,
                     const Eigen::MatrixXd& latents,
                     const Eigen::VectorXd& descriptor,
                     std::size_t field,
                     const std::vector<int>& ids) {
#if FORWARD_ANALYSIS_WITH_OPENCL
        upload_if_needed(refs);
        return mirror_->eval_loss(latents, descriptor, field, ids);
#else
        (void)refs, (void)latents, (void)descriptor, (void)field, (void)ids;
        throw std::logic_error("DecoderDeviceSlot used without OpenCL support");
#endif
    }

    double train_step(const DecoderTensorRefs& refs,
                      const Eigen::MatrixXd& latents,
                      const Eigen::VectorXd& descriptor,
                      std::size_t field,
                      const std::vector<int>& ids,
                      std::uint64_t dropout_state,
                      double dropout_probability,
                      double learning_rate,
                      std::uint64_t adam_step,
                      double output_weight_decay,
                      Eigen::MatrixXd& d_latents) {
#if FORWARD_ANALYSIS_WITH_OPENCL
        upload_if_needed(refs);
        const double loss = mirror_->train_step(latents, descriptor, field, ids, dropout_state, dropout_probability,
                                                learning_rate, adam_step, output_weight_decay, d_latents);
        device_newer_ = true;
        return loss;
#else
        (void)refs, (void)latents, (void)descriptor, (void)field, (void)ids, (void)dropout_state,
            (void)dropout_probability, (void)learning_rate, (void)adam_step, (void)output_weight_decay, (void)d_latents;
        throw std::logic_error("DecoderDeviceSlot used without OpenCL support");
#endif
    }

private:
    void require_host_current() const {
        if (device_newer_) {
            throw std::logic_error("decoder copied while its device copy holds newer parameters; call synchronize_host() first");
        }
    }

#if FORWARD_ANALYSIS_WITH_OPENCL
    void upload_if_needed(const DecoderTensorRefs& refs) {
        if (host_newer_ || mirror_->vocab() != static_cast<std::size_t>(refs.token_embeddings->rows())) {
            mirror_->upload(refs);
            host_newer_ = false;
        }
    }

    // Device copy of parameters, moments and step buffers; null until a device pass is first requested.
    std::unique_ptr<DecoderDeviceMirror> mirror_;
    // True once a device has been looked for, so a machine without one is probed only once per decoder.
    bool probed_ = false;
#endif
    // The host tensors changed since the last upload.
    bool host_newer_ = true;
    // A device pass changed the parameters since the last download.
    bool device_newer_ = false;
};

}  // namespace forward_analysis
