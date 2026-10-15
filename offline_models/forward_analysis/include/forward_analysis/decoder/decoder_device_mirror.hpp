#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/compute/device_buffer.hpp"
#include "forward_analysis/compute/device_gemm.hpp"
#include "forward_analysis/compute/device_parameter.hpp"
#include "forward_analysis/compute/device_parameter_arena.hpp"
#include "forward_analysis/compute/opencl_runtime.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/decoder/decoder_tensor_refs.hpp"
#include "forward_analysis/decoder/observation_tokenizer.hpp"

#include <CL/cl.h>
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace forward_analysis {

// Device-resident decoder. Parameters, Adam moments, gradients and every intermediate of a field pass stay in
// GPU memory; a pass is DeviceGemm products plus the DecoderKernelSource kernels, queued without waiting. Per
// pass the host writes the token ids and Z | d_hat (two writes), and one blocking read returns the per-step losses
// (with dLoss/dZ when training); dropout masks are drawn on the device. The arithmetic follows
// AutoregressiveCausalDecoder::forward / backward and AdamMoment::apply.
class DecoderDeviceMirror {
public:
    DecoderDeviceMirror(std::shared_ptr<OpenClRuntime> runtime, const DistillerConfig& config)
        : runtime_(std::move(runtime)),
          config_(config) {
        const OpenClRuntime& rt = *runtime_;
        const std::size_t steps = config_.max_field_tokens - 1;
        const std::size_t d = config_.d_model;
        const std::size_t h = config_.decoder_hidden_dim;
        ones_.reserve(rt, steps * std::max(d, h));
        ones_.fill(rt, 1.0, steps * std::max(d, h));
        id_segment_ = round_up(steps, rt.alignment_bytes() / sizeof(int));
        token_ids_.reserve(rt, 3 * id_segment_);
        previous_ = DeviceBuffer<int>::region(token_ids_, 0, steps);
        second_previous_ = DeviceBuffer<int>::region(token_ids_, id_segment_, steps);
        targets_ = DeviceBuffer<int>::region(token_ids_, 2 * id_segment_, steps);
        staged_ids_.assign(3 * id_segment_, 0);
        for (DeviceBuffer<double>* b : {&token_mask_, &second_mask_, &step_inputs_, &query_inputs_, &queries_, &mixed_,
                                        &normalized_, &d_normalized_, &d_mixed_, &d_queries_, &d_query_inputs_,
                                        &d_step_inputs_}) {
            b->reserve(rt, steps * d);
        }
        for (DeviceBuffer<double>* b : {&hidden_mask_, &hidden_, &dropped_, &d_hidden_, &d_pre_hidden_}) {
            b->reserve(rt, steps * h);
        }
        for (DeviceBuffer<double>* b : {&query_inverse_std_, &inverse_std_}) {
            b->reserve(rt, steps);
        }
    }

    const std::string& device_description() const noexcept { return runtime_->description(); }
    std::size_t vocab() const noexcept { return vocab_; }

    // Copies every host parameter and Adam moment to the device, laying the arena out for the current shapes.
    void upload(const DecoderTensorRefs& host) {
        const OpenClRuntime& rt = *runtime_;
        const auto shape = [](const auto& tensor) {
            return std::pair<std::size_t, std::size_t>(static_cast<std::size_t>(tensor.rows()),
                                                       static_cast<std::size_t>(tensor.cols()));
        };
        arena_.layout(rt,
                      {&token_embeddings_, &second_token_embeddings_, &field_embeddings_, &query_weights_, &key_weights_,
                       &value_weights_, &descriptor_weights_, &hidden_weights_, &hidden_biases_, &output_weights_,
                       &output_biases_},
                      {shape(*host.token_embeddings), shape(*host.second_token_embeddings), shape(*host.field_embeddings),
                       shape(*host.query_weights), shape(*host.key_weights), shape(*host.value_weights),
                       shape(*host.descriptor_weights), shape(*host.hidden_weights), shape(*host.hidden_biases),
                       shape(*host.output_weights), shape(*host.output_biases)});
        arena_.write(rt, token_embeddings_, *host.token_embeddings, *host.token_moment);
        arena_.write(rt, second_token_embeddings_, *host.second_token_embeddings, *host.token2_moment);
        arena_.write(rt, field_embeddings_, *host.field_embeddings, *host.field_moment);
        arena_.write(rt, query_weights_, *host.query_weights, *host.query_moment);
        arena_.write(rt, key_weights_, *host.key_weights, *host.key_moment);
        arena_.write(rt, value_weights_, *host.value_weights, *host.value_moment);
        arena_.write(rt, descriptor_weights_, *host.descriptor_weights, *host.descriptor_moment);
        arena_.write(rt, hidden_weights_, *host.hidden_weights, *host.hidden_weights_moment);
        arena_.write(rt, hidden_biases_, *host.hidden_biases, *host.hidden_biases_moment);
        arena_.write(rt, output_weights_, *host.output_weights, *host.output_weights_moment);
        arena_.write(rt, output_biases_, *host.output_biases, *host.output_biases_moment);
        position_rows_ = static_cast<std::size_t>(host.positions->rows());
        positions_.reserve(rt, static_cast<std::size_t>(host.positions->size()));
        positions_.write(rt, host.positions->data(), static_cast<std::size_t>(host.positions->size()), true);
        vocab_ = static_cast<std::size_t>(host.token_embeddings->rows());
        probabilities_.reserve(rt, (config_.max_field_tokens - 1) * vocab_);
    }

    // Copies every device parameter and Adam moment back into the host tensors.
    void download(const DecoderTensorRefs& host) const {
        const OpenClRuntime& rt = *runtime_;
        arena_.read(rt, token_embeddings_, *host.token_embeddings, *host.token_moment);
        arena_.read(rt, second_token_embeddings_, *host.second_token_embeddings, *host.token2_moment);
        arena_.read(rt, field_embeddings_, *host.field_embeddings, *host.field_moment);
        arena_.read(rt, query_weights_, *host.query_weights, *host.query_moment);
        arena_.read(rt, key_weights_, *host.key_weights, *host.key_moment);
        arena_.read(rt, value_weights_, *host.value_weights, *host.value_moment);
        arena_.read(rt, descriptor_weights_, *host.descriptor_weights, *host.descriptor_moment);
        arena_.read(rt, hidden_weights_, *host.hidden_weights, *host.hidden_weights_moment);
        arena_.read(rt, hidden_biases_, *host.hidden_biases, *host.hidden_biases_moment);
        arena_.read(rt, output_weights_, *host.output_weights, *host.output_weights_moment);
        arena_.read(rt, output_biases_, *host.output_biases, *host.output_biases_moment);
    }

    // Eval-mode (no dropout) mean token cross-entropy of one field; ids holds BOS ... EOS (at least 2 entries).
    double eval_loss(const Eigen::MatrixXd& latents,
                     const Eigen::VectorXd& descriptor,
                     std::size_t field,
                     const std::vector<int>& ids) {
        const std::size_t steps = ids.size() - 1;
        stage(latents, descriptor, ids);
        forward(steps, field, ones_.mem(), ones_.mem(), ones_.mem(), false);
        const std::size_t latent_size = latent_rows_ * config_.d_model;
        staged_results_.resize(steps);
        results_.read(*runtime_, staged_results_.data(), steps, latent_size);
        return mean_of(staged_results_.data(), steps);
    }

    // Teacher-forced pass, backward pass and one Adam update of every parameter (plus decoupled decay on W_out).
    // With dropout_probability > 0 the three masks (steps x d, steps x d, steps x hidden) are drawn on the device
    // from DeterministicRng state dropout_state, exactly as the host would draw them (the caller advances its
    // generator by steps * (2 d_model + decoder_hidden_dim) draws). Returns the pass loss and writes dLoss/dZ.
    double train_step(const Eigen::MatrixXd& latents,
                      const Eigen::VectorXd& descriptor,
                      std::size_t field,
                      const std::vector<int>& ids,
                      std::uint64_t dropout_state,
                      double dropout_probability,
                      double learning_rate,
                      std::uint64_t adam_step,
                      double output_weight_decay,
                      Eigen::MatrixXd& d_latents) {
        const OpenClRuntime& rt = *runtime_;
        const std::size_t steps = ids.size() - 1;
        stage(latents, descriptor, ids);
        if (dropout_probability > 0.0) {
            const std::size_t draws = steps * (2 * config_.d_model + config_.decoder_hidden_dim);
            rt.launch("dropout_masks", draws, 1, steps, config_.d_model, config_.decoder_hidden_dim,
                      static_cast<std::uint32_t>(dropout_state & 0xFFFFFFFFULL),
                      static_cast<std::uint32_t>(dropout_state >> 32), dropout_probability,
                      1.0 / (1.0 - dropout_probability), token_mask_.mem(), second_mask_.mem(), hidden_mask_.mem());
            forward(steps, field, token_mask_.mem(), second_mask_.mem(), hidden_mask_.mem(), true);
        } else {
            forward(steps, field, ones_.mem(), ones_.mem(), ones_.mem(), true);
        }
        backward(steps, field);
        arena_.adam(rt, learning_rate, adam_step, output_weights_, 1.0 - learning_rate * output_weight_decay);

        const std::size_t latent_size = latent_rows_ * config_.d_model;
        staged_results_.resize(latent_size + steps);
        results_.read(rt, staged_results_.data(), latent_size + steps);
        d_latents = Eigen::Map<const Eigen::MatrixXd>(staged_results_.data(), static_cast<Eigen::Index>(latent_rows_),
                                                      static_cast<Eigen::Index>(config_.d_model));
        return mean_of(staged_results_.data() + latent_size, steps);
    }

private:
    static constexpr double kSecondTokenWeight = 0.5;
    static constexpr double kLayerNormEpsilon = 1e-6;
    // Attention rows up to this many latents use the one-work-item-per-row softmax (a work-group per row spends
    // most of its time in the reduction tree when rows are this short).
    static constexpr std::size_t kNarrowSoftmaxColumns = 32;

    static std::size_t round_up(std::size_t count, std::size_t multiple) {
        const std::size_t step = std::max<std::size_t>(1, multiple);
        return (count + step - 1) / step * step;
    }

    // Sum of count losses in step order, divided by count.
    static double mean_of(const double* losses, std::size_t count) {
        double total = 0.0;
        for (std::size_t t = 0; t < count; ++t) {
            total += losses[t];
        }
        return total / static_cast<double>(count);
    }

    // Lays out Z | d_hat and dLoss/dZ | losses for m latent rows (views are rebuilt only when m changes).
    void ensure_latent_layout(std::size_t m) {
        if (m == latent_rows_ && latents_.mem() != nullptr) {
            return;
        }
        const OpenClRuntime& rt = *runtime_;
        const std::size_t latent_size = m * config_.d_model;
        latents_.reset();
        descriptor_.reset();
        descriptor_offset_ = round_up(latent_size, rt.alignment_bytes() / sizeof(double));
        inputs_.reserve(rt, descriptor_offset_ + ObservationDescriptor::dim());
        latents_ = DeviceBuffer<double>::region(inputs_, 0, latent_size);
        descriptor_ = DeviceBuffer<double>::region(inputs_, descriptor_offset_, ObservationDescriptor::dim());
        staged_inputs_.assign(descriptor_offset_ + ObservationDescriptor::dim(), 0.0);
        for (DeviceBuffer<double>* b : {&keys_, &values_, &d_values_, &d_keys_}) {
            b->reserve(rt, latent_size);
        }
        for (DeviceBuffer<double>* b : {&attention_, &d_attention_, &d_scores_}) {
            b->reserve(rt, (config_.max_field_tokens - 1) * m);
        }
        results_.reserve(rt, latent_size + config_.max_field_tokens - 1);
        latent_rows_ = m;
    }

    // Queues the writes of the token ids and Z | d_hat (non-blocking: the staged host copies stay valid until the
    // blocking read that ends the pass).
    void stage(const Eigen::MatrixXd& latents, const Eigen::VectorXd& descriptor, const std::vector<int>& ids) {
        const OpenClRuntime& rt = *runtime_;
        const std::size_t steps = ids.size() - 1;
        ensure_latent_layout(static_cast<std::size_t>(latents.rows()));
        for (std::size_t t = 0; t < steps; ++t) {
            staged_ids_[t] = ids[t];
            staged_ids_[id_segment_ + t] = t > 0 ? ids[t - 1] : ObservationTokenizer::kPadToken;
            staged_ids_[2 * id_segment_ + t] = ids[t + 1];
        }
        token_ids_.write(rt, staged_ids_.data(), 2 * id_segment_ + steps, false);
        const std::size_t latent_size = latent_rows_ * config_.d_model;
        std::copy(latents.data(), latents.data() + latent_size, staged_inputs_.begin());
        std::copy(descriptor.data(), descriptor.data() + descriptor.size(),
                  staged_inputs_.begin() + static_cast<std::ptrdiff_t>(descriptor_offset_));
        inputs_.write(rt, staged_inputs_.data(), staged_inputs_.size(), false);
    }

    // With write_gradient, probabilities_ ends up holding dLoss/dlogits for backward().
    void forward(std::size_t steps,
                 std::size_t field,
                 cl_mem token_mask,
                 cl_mem second_mask,
                 cl_mem hidden_mask,
                 bool write_gradient) {
        const OpenClRuntime& rt = *runtime_;
        const std::size_t s = steps;
        const std::size_t d = config_.d_model;
        const std::size_t h = config_.decoder_hidden_dim;
        const std::size_t m = latent_rows_;
        const std::size_t v = vocab_;
        const std::size_t dim = ObservationDescriptor::dim();
        const double scale = 1.0 / std::sqrt(static_cast<double>(d));

        rt.launch("build_step_inputs", s, d, s, d, v, field_embeddings_.rows(), position_rows_, field,
                  kSecondTokenWeight, previous_.mem(), second_previous_.mem(), token_embeddings_.value(),
                  second_token_embeddings_.value(), field_embeddings_.value(), positions_.mem(), token_mask, second_mask,
                  step_inputs_.mem());
        rt.launch_rows("layer_norm_rows", s, s, d, kLayerNormEpsilon, step_inputs_.mem(), query_inputs_.mem(),
                       query_inverse_std_.mem());
        DeviceGemm::multiply(rt, false, false, s, d, d, 1.0, query_inputs_.mem(), s, query_weights_.value(), d, 0.0,
                             queries_.mem(), s);
        DeviceGemm::multiply(rt, false, false, m, d, d, 1.0, latents_.mem(), m, key_weights_.value(), d, 0.0, keys_.mem(),
                             m);
        DeviceGemm::multiply(rt, false, false, m, d, d, 1.0, latents_.mem(), m, value_weights_.value(), d, 0.0,
                             values_.mem(), m);
        DeviceGemm::multiply(rt, false, true, s, m, d, scale, queries_.mem(), s, keys_.mem(), m, 0.0, attention_.mem(), s);
        if (m <= kNarrowSoftmaxColumns) {
            rt.launch("softmax_rows_narrow", s, 1, s, m, attention_.mem());
        } else {
            rt.launch_rows("softmax_rows", s, s, m, attention_.mem());
        }

        DeviceGemm::multiply(rt, false, false, s, d, m, 1.0, attention_.mem(), s, values_.mem(), m, 0.0, mixed_.mem(), s);
        rt.launch("add_inputs_and_conditioning", s, d, s, d, dim, mixed_.mem(), step_inputs_.mem(),
                  descriptor_weights_.value(), descriptor_.mem());
        rt.launch_rows("layer_norm_rows", s, s, d, kLayerNormEpsilon, mixed_.mem(), normalized_.mem(), inverse_std_.mem());

        DeviceGemm::multiply(rt, false, true, s, h, d, 1.0, normalized_.mem(), s, hidden_weights_.value(), h, 0.0,
                             hidden_.mem(), s);
        rt.launch("bias_tanh_mask", s, h, s, h, hidden_.mem(), hidden_biases_.value(), hidden_mask, dropped_.mem());
        DeviceGemm::multiply(rt, false, true, s, v, h, 1.0, dropped_.mem(), s, output_weights_.value(), v, 0.0,
                             probabilities_.mem(), s);
        rt.launch_rows("softmax_loss_rows", s, s, v, probabilities_.mem(), output_biases_.value(), targets_.mem(),
                       results_.mem(), m * d, write_gradient ? 1 : 0);
        hidden_mask_in_use_ = hidden_mask;
        token_mask_in_use_ = token_mask;
        second_mask_in_use_ = second_mask;
    }

    // Requires forward(..., write_gradient = true). Leaves every parameter gradient in the arena and dLoss/dZ at
    // the start of results_.
    void backward(std::size_t steps, std::size_t field) {
        const OpenClRuntime& rt = *runtime_;
        const std::size_t s = steps;
        const std::size_t d = config_.d_model;
        const std::size_t h = config_.decoder_hidden_dim;
        const std::size_t m = latent_rows_;
        const std::size_t v = vocab_;
        const std::size_t dim = ObservationDescriptor::dim();
        const double scale = 1.0 / std::sqrt(static_cast<double>(d));

        DeviceGemm::multiply(rt, true, false, v, h, s, 1.0, probabilities_.mem(), s, dropped_.mem(), s, 0.0,
                             output_weights_.gradient(), v);
        rt.launch("column_sums", v, 1, s, v, probabilities_.mem(), 1, 0, output_biases_.gradient());
        DeviceGemm::multiply(rt, false, false, s, h, v, 1.0, probabilities_.mem(), s, output_weights_.value(), v, 0.0,
                             d_hidden_.mem(), s);
        rt.launch_rows("tanh_backward_columns", h, s, h, d_hidden_.mem(), hidden_mask_in_use_, hidden_.mem(),
                       d_pre_hidden_.mem(), hidden_biases_.gradient());
        DeviceGemm::multiply(rt, true, false, h, d, s, 1.0, d_pre_hidden_.mem(), s, normalized_.mem(), s, 0.0,
                             hidden_weights_.gradient(), h);
        DeviceGemm::multiply(rt, false, false, s, d, h, 1.0, d_pre_hidden_.mem(), s, hidden_weights_.value(), h, 0.0,
                             d_normalized_.mem(), s);
        rt.launch_rows("layer_norm_backward", s, s, d, normalized_.mem(), inverse_std_.mem(), d_normalized_.mem(), 0,
                       d_normalized_.mem(), d_mixed_.mem());
        rt.launch("descriptor_gradient", d, dim, s, d, dim, d_mixed_.mem(), descriptor_.mem(),
                  descriptor_weights_.gradient());

        DeviceGemm::multiply(rt, false, true, s, m, d, 1.0, d_mixed_.mem(), s, values_.mem(), m, 0.0, d_attention_.mem(),
                             s);
        DeviceGemm::multiply(rt, true, false, m, d, s, 1.0, attention_.mem(), s, d_mixed_.mem(), s, 0.0, d_values_.mem(),
                             m);
        rt.launch("softmax_backward_rows", s, 1, s, m, attention_.mem(), d_attention_.mem(), d_scores_.mem());
        DeviceGemm::multiply(rt, false, false, s, d, m, scale, d_scores_.mem(), s, keys_.mem(), m, 0.0, d_queries_.mem(),
                             s);
        DeviceGemm::multiply(rt, true, false, m, d, s, scale, d_scores_.mem(), s, queries_.mem(), s, 0.0, d_keys_.mem(),
                             m);

        DeviceGemm::multiply(rt, true, false, d, d, s, 1.0, query_inputs_.mem(), s, d_queries_.mem(), s, 0.0,
                             query_weights_.gradient(), d);
        DeviceGemm::multiply(rt, true, false, d, d, m, 1.0, latents_.mem(), m, d_keys_.mem(), m, 0.0,
                             key_weights_.gradient(), d);
        DeviceGemm::multiply(rt, true, false, d, d, m, 1.0, latents_.mem(), m, d_values_.mem(), m, 0.0,
                             value_weights_.gradient(), d);
        DeviceGemm::multiply(rt, false, true, m, d, d, 1.0, d_keys_.mem(), m, key_weights_.value(), d, 0.0,
                             results_.mem(), m);
        DeviceGemm::multiply(rt, false, true, m, d, d, 1.0, d_values_.mem(), m, value_weights_.value(), d, 1.0,
                             results_.mem(), m);

        DeviceGemm::multiply(rt, false, true, s, d, d, 1.0, d_queries_.mem(), s, query_weights_.value(), d, 0.0,
                             d_query_inputs_.mem(), s);
        rt.launch_rows("layer_norm_backward", s, s, d, query_inputs_.mem(), query_inverse_std_.mem(),
                       d_query_inputs_.mem(), 1, d_mixed_.mem(), d_step_inputs_.mem());

        arena_.zero_gradients_through(rt, field_embeddings_);
        rt.launch("embedding_gradients", s, d, s, d, v, kSecondTokenWeight, field_embeddings_.rows(), field,
                  previous_.mem(), second_previous_.mem(), d_step_inputs_.mem(), token_mask_in_use_, second_mask_in_use_,
                  token_embeddings_.gradient(), second_token_embeddings_.gradient(), field_embeddings_.gradient());
    }

    // Device, context, queue and compiled kernels shared by every mirror in the process.
    std::shared_ptr<OpenClRuntime> runtime_;
    // d_model, decoder_hidden_dim, max_field_tokens and num_latents fix the buffer shapes.
    DistillerConfig config_;
    // Vocabulary size of the uploaded token tensors.
    std::size_t vocab_ = 0;
    // Rows of Z the latent-sized buffers are laid out for (num_latents); 0 before the first pass.
    std::size_t latent_rows_ = 0;

    // Packed values, gradients and Adam moments of the trainable tensors below, laid out E, E2, F first so their
    // accumulated gradients clear with one fill.
    DeviceParameterArena arena_;
    // Views of each trainable tensor in arena_ (same shapes as the host decoder's tensors).
    DeviceParameter token_embeddings_;
    DeviceParameter second_token_embeddings_;
    DeviceParameter field_embeddings_;
    DeviceParameter query_weights_;
    DeviceParameter key_weights_;
    DeviceParameter value_weights_;
    DeviceParameter descriptor_weights_;
    DeviceParameter hidden_weights_;
    DeviceParameter hidden_biases_;
    DeviceParameter output_weights_;
    DeviceParameter output_biases_;
    // Fixed sinusoidal step vectors (max_field_tokens x d, never trained) and their row count.
    DeviceBuffer<double> positions_;
    std::size_t position_rows_ = 0;

    // Token ids of a pass in three aligned segments of id_segment_ entries (previous, two back, target), with a
    // view of each segment.
    DeviceBuffer<int> token_ids_;
    std::size_t id_segment_ = 0;
    DeviceBuffer<int> previous_;
    DeviceBuffer<int> second_previous_;
    DeviceBuffer<int> targets_;
    // Z (M x d) at the start and d_hat (14) at descriptor_offset_, with a view of each.
    DeviceBuffer<double> inputs_;
    std::size_t descriptor_offset_ = 0;
    DeviceBuffer<double> latents_;
    DeviceBuffer<double> descriptor_;
    // Dropout masks drawn by dropout_masks; ones_ stands in for every mask in eval mode or with dropout 0.
    DeviceBuffer<double> token_mask_;
    DeviceBuffer<double> second_mask_;
    DeviceBuffer<double> hidden_mask_;
    DeviceBuffer<double> ones_;
    // Masks used by the latest forward pass, read again by backward.
    cl_mem token_mask_in_use_ = nullptr;
    cl_mem second_mask_in_use_ = nullptr;
    cl_mem hidden_mask_in_use_ = nullptr;

    // Forward intermediates: u, LN(u) and its row scales, queries, keys, values, attention (steps x M),
    // x = u + A V + W_aux d_hat, LN(x) and its row scales, tanh hidden, dropped hidden, and probabilities
    // (steps x vocab; dLoss/dlogits after a training forward pass).
    DeviceBuffer<double> step_inputs_;
    DeviceBuffer<double> query_inputs_;
    DeviceBuffer<double> query_inverse_std_;
    DeviceBuffer<double> queries_;
    DeviceBuffer<double> keys_;
    DeviceBuffer<double> values_;
    DeviceBuffer<double> attention_;
    DeviceBuffer<double> mixed_;
    DeviceBuffer<double> normalized_;
    DeviceBuffer<double> inverse_std_;
    DeviceBuffer<double> hidden_;
    DeviceBuffer<double> dropped_;
    DeviceBuffer<double> probabilities_;

    // Backward intermediates: dLoss/d(dropped hidden), d(pre-activation), d(LN(x)), dx, d(attention), d(values),
    // d(scores), d(queries), d(keys), d(LN(u)) and du.
    DeviceBuffer<double> d_hidden_;
    DeviceBuffer<double> d_pre_hidden_;
    DeviceBuffer<double> d_normalized_;
    DeviceBuffer<double> d_mixed_;
    DeviceBuffer<double> d_attention_;
    DeviceBuffer<double> d_values_;
    DeviceBuffer<double> d_scores_;
    DeviceBuffer<double> d_queries_;
    DeviceBuffer<double> d_keys_;
    DeviceBuffer<double> d_query_inputs_;
    DeviceBuffer<double> d_step_inputs_;
    // dLoss/dZ (M x d) followed by the per-step losses, read back in one transfer.
    DeviceBuffer<double> results_;

    // Host copies backing the non-blocking writes of one pass, and the results read back.
    std::vector<int> staged_ids_;
    std::vector<double> staged_inputs_;
    std::vector<double> staged_results_;
};

}  // namespace forward_analysis

#endif
