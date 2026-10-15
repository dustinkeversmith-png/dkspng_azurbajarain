#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/decoder/decoder_device_slot.hpp"
#include "forward_analysis/decoder/decoder_domain_guidance.hpp"
#include "forward_analysis/decoder/decoder_parameter_gradients.hpp"
#include "forward_analysis/decoder/decoder_sequence_cache.hpp"
#include "forward_analysis/decoder/decoder_tensor_refs.hpp"
#include "forward_analysis/decoder/observation_tokenizer.hpp"
#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/encoder/sinusoidal_positional_encoding.hpp"
#include "forward_analysis/math/adam_moment.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace forward_analysis {

// Field-conditioned causal word decoder that cross-attends over the latent bottleneck Z (M x d) and is
// conditioned on the Aux1-predicted observation descriptor d_hat. Each step t:
//   u_t   = drop(E[prev]) + 0.5 drop(E2[prev2]) + F[field] + P[t]
//   alpha = softmax(LN(u_t) W_Q (Z W_K)^T / sqrt(d)),  c_t = alpha (Z W_V)
//   x_t   = LN(u_t + c_t + W_aux d_hat)
//   h_t   = drop(tanh(W1 x_t + b1)),  logits = W_out h_t + b_out
// drop() is inverted dropout, active only in training steps.
// With config.use_gpu and an OpenCL GPU, training passes and evaluation losses run on the device (parameters and
// Adam state stay there between passes); generation, attention inspection and copies read host tensors, which are
// refreshed from the device first.
class AutoregressiveCausalDecoder {
public:
    explicit AutoregressiveCausalDecoder(const DistillerConfig& config = DistillerConfig{},
                                         std::uint64_t seed = 4000ULL)
        : config_(config),
          rng_(seed),
          dropout_rng_(seed + 1ULL) {
        config_.validate();
        const std::size_t d = config_.d_model;
        const std::size_t h = config_.decoder_hidden_dim;
        field_embeddings_ = random_normal(ObservationText::kFieldCount, d, kEmbeddingScale);
        positions_ = SinusoidalPositionalEncoding::encoding(config_.max_field_tokens, d) * kEmbeddingScale;
        query_weights_ = rng_.xavier_matrix(d, d);
        key_weights_ = rng_.xavier_matrix(d, d);
        value_weights_ = rng_.xavier_matrix(d, d);
        descriptor_weights_ = rng_.xavier_matrix(d, ObservationDescriptor::dim());
        hidden_weights_ = rng_.xavier_matrix(h, d);
        hidden_biases_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(h));
        grow_vocabulary();
    }

    void set_frozen(bool frozen) noexcept { frozen_ = frozen; }
    bool is_frozen() const noexcept { return frozen_; }

    // Registers every lexeme of the given field sequences so parameter shapes are fixed before training.
    void prepare_vocabulary(const std::vector<std::vector<std::string>>& field_lexemes) {
        for (const std::vector<std::string>& lexemes : field_lexemes) {
            tokenizer_.encode_lexemes(lexemes, config_.max_field_tokens, true);
        }
        grow_vocabulary_synchronized();
    }

    // Eval-mode (no dropout) mean token cross-entropy of one field.
    double compute_loss(const Eigen::MatrixXd& latents_z,
                        const Eigen::VectorXd& descriptor,
                        std::size_t field,
                        const std::vector<std::string>& lexemes) const {
        const std::vector<int> ids = tokenizer_.encode_known_lexemes(lexemes, config_.max_field_tokens);
        if (ids.size() < 2) {
            return 0.0;
        }
        if (device_.active(config_)) {
            validate_inputs(field, descriptor);
            return device_.eval_loss(tensor_refs(), latents_z, descriptor, field, ids);
        }
        return cross_entropy(forward(latents_z, descriptor, field, ids, nullptr), ids);
    }

    // Eval-mode loss and its exact gradient with respect to Z, without touching any weight.
    double loss_and_latent_gradient(const Eigen::MatrixXd& latents_z,
                                    const Eigen::VectorXd& descriptor,
                                    std::size_t field,
                                    const std::vector<std::string>& lexemes,
                                    Eigen::MatrixXd& d_latents) const {
        synchronize_host();
        const std::vector<int> ids = tokenizer_.encode_known_lexemes(lexemes, config_.max_field_tokens);
        d_latents = Eigen::MatrixXd::Zero(latents_z.rows(), latents_z.cols());
        if (ids.size() < 2) {
            return 0.0;
        }
        const DecoderSequenceCache cache = forward(latents_z, descriptor, field, ids, nullptr);
        d_latents = backward(cache, latents_z, descriptor, field, ids).latents;
        return cross_entropy(cache, ids);
    }

    // Attention weights over the latents at every teacher-forced step (T x M), eval mode.
    Eigen::MatrixXd attention_weights(const Eigen::MatrixXd& latents_z,
                                      const Eigen::VectorXd& descriptor,
                                      std::size_t field,
                                      const std::vector<std::string>& lexemes) const {
        synchronize_host();
        const std::vector<int> ids = tokenizer_.encode_known_lexemes(lexemes, config_.max_field_tokens);
        return forward(latents_z, descriptor, field, ids, nullptr).attention;
    }

    // Teacher-forced training pass over one field with dropout; returns that pass's loss, updates the decoder
    // with Adam (plus decoupled weight decay on W_out) unless frozen, and adds dLoss/dZ into d_latents_accum.
    double compute_loss_and_step(const Eigen::MatrixXd& latents_z,
                                 const Eigen::VectorXd& descriptor,
                                 std::size_t field,
                                 const std::vector<std::string>& lexemes,
                                 Eigen::MatrixXd& d_latents_accum) {
        const std::vector<int> ids = tokenizer_.encode_lexemes(lexemes, config_.max_field_tokens, true);
        grow_vocabulary_synchronized();
        if (ids.size() < 2) {
            return 0.0;
        }
        const bool on_device = device_.active(config_);
        if (frozen_) {
            return on_device ? compute_loss(latents_z, descriptor, field, lexemes)
                             : cross_entropy(forward(latents_z, descriptor, field, ids, nullptr), ids);
        }
        if (on_device) {
            validate_inputs(field, descriptor);
            // The device draws the three dropout masks from the generator's current state in the order dropout_mask
            // would (token, second-token, hidden; row by row), so the generator skips ahead by that many draws.
            const std::uint64_t dropout_state = dropout_rng_.state;
            if (config_.decoder_dropout > 0.0) {
                dropout_rng_.skip(static_cast<std::uint64_t>(ids.size() - 1) *
                                  (2 * config_.d_model + config_.decoder_hidden_dim));
            }
            ++adam_step_;
            Eigen::MatrixXd d_latents;
            const double loss = device_.train_step(tensor_refs(), latents_z, descriptor, field, ids, dropout_state,
                                                   config_.decoder_dropout, config_.decoder_learning_rate, adam_step_,
                                                   config_.decoder_output_weight_decay, d_latents);
            if (d_latents_accum.rows() != latents_z.rows() || d_latents_accum.cols() != latents_z.cols()) {
                d_latents_accum = Eigen::MatrixXd::Zero(latents_z.rows(), latents_z.cols());
            }
            d_latents_accum += d_latents;
            return loss;
        }
        const DecoderSequenceCache cache = forward(latents_z, descriptor, field, ids, &dropout_rng_);
        const double loss = cross_entropy(cache, ids);
        const DecoderParameterGradients g = backward(cache, latents_z, descriptor, field, ids);

        if (d_latents_accum.rows() != latents_z.rows() || d_latents_accum.cols() != latents_z.cols()) {
            d_latents_accum = Eigen::MatrixXd::Zero(latents_z.rows(), latents_z.cols());
        }
        d_latents_accum += g.latents;

        const double lr = config_.decoder_learning_rate;
        ++adam_step_;
        output_weights_moment_.apply(output_weights_, g.output_weights, lr, adam_step_);
        output_weights_ *= 1.0 - lr * config_.decoder_output_weight_decay;
        output_biases_moment_.apply(output_biases_, g.output_biases, lr, adam_step_);
        hidden_weights_moment_.apply(hidden_weights_, g.hidden_weights, lr, adam_step_);
        hidden_biases_moment_.apply(hidden_biases_, g.hidden_biases, lr, adam_step_);
        token_moment_.apply(token_embeddings_, g.token_embeddings, lr, adam_step_);
        token2_moment_.apply(second_token_embeddings_, g.second_token_embeddings, lr, adam_step_);
        field_moment_.apply(field_embeddings_, g.field_embeddings, lr, adam_step_);
        query_moment_.apply(query_weights_, g.query_weights, lr, adam_step_);
        key_moment_.apply(key_weights_, g.key_weights, lr, adam_step_);
        value_moment_.apply(value_weights_, g.value_weights, lr, adam_step_);
        descriptor_moment_.apply(descriptor_weights_, g.descriptor_weights, lr, adam_step_);
        return loss;
    }

    // Greedy generation of one field, at most budget tokens including BOS/EOS.
    std::vector<int> generate_field_tokens(const Eigen::MatrixXd& latents_z,
                                           const Eigen::VectorXd& descriptor,
                                           std::size_t field,
                                           std::size_t budget) const {
        synchronize_host();
        const std::size_t cap = std::min(std::max<std::size_t>(2, budget), config_.max_field_tokens);
        const Eigen::MatrixXd keys = latents_z * key_weights_;
        const Eigen::MatrixXd values = latents_z * value_weights_;
        const Eigen::RowVectorXd base = field_embeddings_.row(static_cast<Eigen::Index>(field));
        const Eigen::RowVectorXd conditioning = (descriptor_weights_ * descriptor).transpose();
        const double scale = 1.0 / std::sqrt(static_cast<double>(config_.d_model));
        std::vector<int> ids{ObservationTokenizer::kBosToken};
        while (ids.size() + 1 < cap) {
            const std::size_t t = ids.size() - 1;
            const int prev = ids.back();
            const int prev2 = ids.size() >= 2 ? ids[ids.size() - 2] : ObservationTokenizer::kPadToken;
            const Eigen::MatrixXd u = base + token_embeddings_.row(prev) +
                                      kSecondTokenWeight * second_token_embeddings_.row(prev2) +
                                      positions_.row(static_cast<Eigen::Index>(t));
            const Eigen::MatrixXd alpha =
                MatrixOps::softmax_rows(MatrixOps::layer_norm_rows(u) * query_weights_ * keys.transpose() * scale);
            const Eigen::MatrixXd x = u + alpha * values + conditioning;
            const Eigen::MatrixXd xn = MatrixOps::layer_norm_rows(x);
            const Eigen::VectorXd hidden = (hidden_weights_ * xn.transpose() + hidden_biases_).array().tanh().matrix();
            Eigen::VectorXd logits = output_weights_ * hidden + output_biases_;
            logits(ObservationTokenizer::kPadToken) = -1e30;
            logits(ObservationTokenizer::kBosToken) = -1e30;
            logits(ObservationTokenizer::kUnkToken) = -1e30;
            if (config_.enable_structural_decoder_constraints) {
                DecoderDomainGuidance::apply(field, ids, descriptor, tokenizer_, logits);
            }
            block_repeated_ngrams(ids, logits);
            Eigen::Index best = 0;
            logits.maxCoeff(&best);
            ids.push_back(static_cast<int>(best));
            if (best == ObservationTokenizer::kEosToken) {
                return ids;
            }
        }
        ids.push_back(ObservationTokenizer::kEosToken);
        return ids;
    }

    const ObservationTokenizer& tokenizer() const noexcept { return tokenizer_; }
    ObservationTokenizer& tokenizer() noexcept { return tokenizer_; }
    const Eigen::MatrixXd& output_weights() const {
        synchronize_host();
        return output_weights_;
    }

    // Copies parameters and Adam state back from the device when a device pass changed them (required before the
    // decoder is copied, e.g. into a checkpoint).
    void synchronize_host() const { device_.synchronize_host(tensor_refs()); }

    // Where training passes and evaluation losses run, e.g. "NVIDIA GeForce RTX 4070 (NVIDIA CUDA, OpenCL 3.0 ...)"
    // or "cpu".
    std::string accelerator() const {
        device_.active(config_);
        return device_.description();
    }

private:
    static constexpr double kEmbeddingScale = 0.5;
    static constexpr double kSecondTokenWeight = 0.5;

    void validate_inputs(std::size_t field, const Eigen::VectorXd& descriptor) const {
        if (field >= ObservationText::kFieldCount) {
            throw std::invalid_argument("AutoregressiveCausalDecoder field index out of range");
        }
        if (descriptor.size() != static_cast<Eigen::Index>(ObservationDescriptor::dim())) {
            throw std::invalid_argument("AutoregressiveCausalDecoder descriptor must have ObservationDescriptor::dim() entries");
        }
    }

    // Addresses of the host tensors and Adam state (mutable, so const accessors can refresh them from the device).
    DecoderTensorRefs tensor_refs() const {
        DecoderTensorRefs refs;
        refs.token_embeddings = &token_embeddings_;
        refs.token_moment = &token_moment_;
        refs.second_token_embeddings = &second_token_embeddings_;
        refs.token2_moment = &token2_moment_;
        refs.field_embeddings = &field_embeddings_;
        refs.field_moment = &field_moment_;
        refs.positions = &positions_;
        refs.query_weights = &query_weights_;
        refs.query_moment = &query_moment_;
        refs.key_weights = &key_weights_;
        refs.key_moment = &key_moment_;
        refs.value_weights = &value_weights_;
        refs.value_moment = &value_moment_;
        refs.descriptor_weights = &descriptor_weights_;
        refs.descriptor_moment = &descriptor_moment_;
        refs.hidden_weights = &hidden_weights_;
        refs.hidden_weights_moment = &hidden_weights_moment_;
        refs.hidden_biases = &hidden_biases_;
        refs.hidden_biases_moment = &hidden_biases_moment_;
        refs.output_weights = &output_weights_;
        refs.output_weights_moment = &output_weights_moment_;
        refs.output_biases = &output_biases_;
        refs.output_biases_moment = &output_biases_moment_;
        return refs;
    }

    // Adds rows for newly registered tokens, first pulling device-side updates so they are not overwritten.
    void grow_vocabulary_synchronized() {
        if (static_cast<Eigen::Index>(tokenizer_.vocab_size()) <= token_embeddings_.rows()) {
            return;
        }
        synchronize_host();
        grow_vocabulary();
        device_.mark_host_newer();
    }

    // Forbids any token that would repeat a four-token sequence already emitted in this field.
    static void block_repeated_ngrams(const std::vector<int>& ids, Eigen::VectorXd& logits) {
        const std::size_t n = ids.size();
        if (n < 4) {
            return;
        }
        for (std::size_t j = 0; j + 3 < n; ++j) {
            if (ids[j] == ids[n - 3] && ids[j + 1] == ids[n - 2] && ids[j + 2] == ids[n - 1]) {
                logits(ids[j + 3]) = -1e30;
            }
        }
    }

    Eigen::MatrixXd random_normal(std::size_t rows, std::size_t cols, double scale) {
        Eigen::MatrixXd m(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
        for (Eigen::Index r = 0; r < m.rows(); ++r) {
            for (Eigen::Index c = 0; c < m.cols(); ++c) {
                m(r, c) = rng_.normal(0.0, scale);
            }
        }
        return m;
    }

    // Inverted dropout mask: each entry is 0 with probability p, otherwise 1 / (1 - p). All ones when rng is null.
    Eigen::MatrixXd dropout_mask(Eigen::Index rows, Eigen::Index cols, DeterministicRng* rng) const {
        Eigen::MatrixXd mask = Eigen::MatrixXd::Ones(rows, cols);
        const double p = config_.decoder_dropout;
        if (rng == nullptr || p <= 0.0) {
            return mask;
        }
        const double keep = 1.0 / (1.0 - p);
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) {
                mask(r, c) = rng->uniform() < p ? 0.0 : keep;
            }
        }
        return mask;
    }

    void grow_vocabulary() {
        const Eigen::Index vocab = static_cast<Eigen::Index>(tokenizer_.vocab_size());
        const Eigen::Index old = token_embeddings_.rows();
        if (vocab <= old) {
            return;
        }
        const Eigen::Index added = vocab - old;
        const std::size_t a = static_cast<std::size_t>(added);
        token_embeddings_.conservativeResize(vocab, static_cast<Eigen::Index>(config_.d_model));
        second_token_embeddings_.conservativeResize(vocab, static_cast<Eigen::Index>(config_.d_model));
        output_weights_.conservativeResize(vocab, static_cast<Eigen::Index>(config_.decoder_hidden_dim));
        output_biases_.conservativeResize(vocab);
        token_embeddings_.bottomRows(added) = random_normal(a, config_.d_model, kEmbeddingScale);
        second_token_embeddings_.bottomRows(added) = random_normal(a, config_.d_model, kEmbeddingScale);
        output_weights_.bottomRows(added) = rng_.xavier_matrix(a, config_.decoder_hidden_dim);
        output_biases_.tail(added).setZero();
    }

    // Teacher-forced pass; rng non-null switches on dropout.
    DecoderSequenceCache forward(const Eigen::MatrixXd& latents_z,
                                 const Eigen::VectorXd& descriptor,
                                 std::size_t field,
                                 const std::vector<int>& ids,
                                 DeterministicRng* rng) const {
        validate_inputs(field, descriptor);
        const Eigen::Index steps = static_cast<Eigen::Index>(ids.size() - 1);
        const Eigen::Index d = static_cast<Eigen::Index>(config_.d_model);
        DecoderSequenceCache cache;
        cache.embedding_mask = dropout_mask(steps, d, rng);
        cache.second_embedding_mask = dropout_mask(steps, d, rng);
        cache.hidden_mask = dropout_mask(steps, static_cast<Eigen::Index>(config_.decoder_hidden_dim), rng);

        Eigen::MatrixXd u(steps, d);
        for (Eigen::Index t = 0; t < steps; ++t) {
            const int prev = ids[static_cast<std::size_t>(t)];
            const int prev2 = t > 0 ? ids[static_cast<std::size_t>(t - 1)] : ObservationTokenizer::kPadToken;
            cache.previous_tokens.push_back(prev);
            cache.second_previous_tokens.push_back(prev2);
            u.row(t) = token_embeddings_.row(prev).cwiseProduct(cache.embedding_mask.row(t)) +
                       kSecondTokenWeight * second_token_embeddings_.row(prev2).cwiseProduct(cache.second_embedding_mask.row(t)) +
                       field_embeddings_.row(static_cast<Eigen::Index>(field)) + positions_.row(t);
        }

        cache.query_inverse_std = MatrixOps::row_inverse_std(u);
        cache.query_inputs = MatrixOps::layer_norm_rows(u);
        cache.queries = cache.query_inputs * query_weights_;
        cache.keys = latents_z * key_weights_;
        cache.values = latents_z * value_weights_;
        cache.attention = MatrixOps::softmax_rows(cache.queries * cache.keys.transpose() /
                                                  std::sqrt(static_cast<double>(config_.d_model)));

        Eigen::MatrixXd x = u + cache.attention * cache.values;
        x.rowwise() += (descriptor_weights_ * descriptor).transpose();
        cache.inverse_std = MatrixOps::row_inverse_std(x);
        cache.normalized_inputs = MatrixOps::layer_norm_rows(x);
        cache.hidden = MatrixOps::affine_rows(cache.normalized_inputs, hidden_weights_, hidden_biases_).array().tanh().matrix();
        cache.probabilities = MatrixOps::softmax_rows(
            MatrixOps::affine_rows(cache.hidden.cwiseProduct(cache.hidden_mask), output_weights_, output_biases_));
        return cache;
    }

    DecoderParameterGradients backward(const DecoderSequenceCache& cache,
                                       const Eigen::MatrixXd& latents_z,
                                       const Eigen::VectorXd& descriptor,
                                       std::size_t field,
                                       const std::vector<int>& ids) const {
        const Eigen::Index steps = cache.probabilities.rows();
        const double scale = 1.0 / std::sqrt(static_cast<double>(config_.d_model));
        DecoderParameterGradients g;

        Eigen::MatrixXd d_logits = cache.probabilities;
        for (Eigen::Index t = 0; t < steps; ++t) {
            d_logits(t, ids[static_cast<std::size_t>(t + 1)]) -= 1.0;
        }
        d_logits /= static_cast<double>(steps);

        const Eigen::MatrixXd dropped_hidden = cache.hidden.cwiseProduct(cache.hidden_mask);
        g.output_weights = d_logits.transpose() * dropped_hidden;
        g.output_biases = d_logits.colwise().sum().transpose();
        const Eigen::MatrixXd d_pre_hidden = (d_logits * output_weights_)
                                                 .cwiseProduct(cache.hidden_mask)
                                                 .cwiseProduct((1.0 - cache.hidden.array().square()).matrix());
        g.hidden_weights = d_pre_hidden.transpose() * cache.normalized_inputs;
        g.hidden_biases = d_pre_hidden.colwise().sum().transpose();
        const Eigen::MatrixXd d_x = MatrixOps::layer_norm_rows_backward(
            cache.normalized_inputs, cache.inverse_std, d_pre_hidden * hidden_weights_);

        g.descriptor_weights = d_x.colwise().sum().transpose() * descriptor.transpose();

        const Eigen::MatrixXd d_context = d_x;
        const Eigen::MatrixXd d_attention = d_context * cache.values.transpose();
        const Eigen::MatrixXd d_values = cache.attention.transpose() * d_context;
        const Eigen::VectorXd row_dot = d_attention.cwiseProduct(cache.attention).rowwise().sum();
        const Eigen::MatrixXd d_scores =
            cache.attention.cwiseProduct(d_attention - row_dot.replicate(1, d_attention.cols()));
        const Eigen::MatrixXd d_queries = d_scores * cache.keys * scale;
        const Eigen::MatrixXd d_keys = d_scores.transpose() * cache.queries * scale;

        g.query_weights = cache.query_inputs.transpose() * d_queries;
        g.key_weights = latents_z.transpose() * d_keys;
        g.value_weights = latents_z.transpose() * d_values;
        g.latents = d_keys * key_weights_.transpose() + d_values * value_weights_.transpose();

        const Eigen::MatrixXd d_u = d_x + MatrixOps::layer_norm_rows_backward(
                                              cache.query_inputs, cache.query_inverse_std, d_queries * query_weights_.transpose());

        g.token_embeddings = Eigen::MatrixXd::Zero(token_embeddings_.rows(), token_embeddings_.cols());
        g.second_token_embeddings = Eigen::MatrixXd::Zero(token_embeddings_.rows(), token_embeddings_.cols());
        g.field_embeddings = Eigen::MatrixXd::Zero(field_embeddings_.rows(), field_embeddings_.cols());
        for (Eigen::Index t = 0; t < steps; ++t) {
            g.token_embeddings.row(cache.previous_tokens[static_cast<std::size_t>(t)]) +=
                d_u.row(t).cwiseProduct(cache.embedding_mask.row(t));
            g.second_token_embeddings.row(cache.second_previous_tokens[static_cast<std::size_t>(t)]) +=
                kSecondTokenWeight * d_u.row(t).cwiseProduct(cache.second_embedding_mask.row(t));
        }
        g.field_embeddings.row(static_cast<Eigen::Index>(field)) = d_u.colwise().sum();
        return g;
    }

    static double cross_entropy(const DecoderSequenceCache& cache, const std::vector<int>& ids) {
        double total = 0.0;
        const Eigen::Index steps = cache.probabilities.rows();
        for (Eigen::Index t = 0; t < steps; ++t) {
            total -= std::log(std::max(1e-12, cache.probabilities(t, ids[static_cast<std::size_t>(t + 1)])));
        }
        return total / static_cast<double>(steps);
    }

    // d_model, decoder_hidden_dim, max_field_tokens, decoder learning rate, dropout and output weight decay.
    DistillerConfig config_;
    // Source of initial weights, including rows added when the vocabulary grows.
    DeterministicRng rng_;
    // Source of dropout masks during training steps.
    DeterministicRng dropout_rng_;
    // When true, compute_loss_and_step reports loss without touching weights (pretraining phase).
    bool frozen_ = true;
    // Number of Adam updates applied so far (bias-correction step count).
    std::uint64_t adam_step_ = 0;
    // Lexeme vocabulary shared by all four fields, slot tokens included.
    ObservationTokenizer tokenizer_;
    // The tensors below are mutable because const accessors refresh them from the device copy when a device pass
    // has changed them; their logical value is the same on both sides.
    // Embedding of the previous token (vocab x d_model).
    mutable Eigen::MatrixXd token_embeddings_;
    // Embedding of the token two steps back (vocab x d_model).
    mutable Eigen::MatrixXd second_token_embeddings_;
    // One embedding per observation field (4 x d_model), selecting which field is being written.
    mutable Eigen::MatrixXd field_embeddings_;
    // Fixed sinusoidal step vectors (max_field_tokens x d_model).
    mutable Eigen::MatrixXd positions_;
    // Attention query projection W_Q (d_model x d_model).
    mutable Eigen::MatrixXd query_weights_;
    // Attention key projection W_K applied to each latent row (d_model x d_model).
    mutable Eigen::MatrixXd key_weights_;
    // Attention value projection W_V applied to each latent row (d_model x d_model).
    mutable Eigen::MatrixXd value_weights_;
    // Projects the predicted observation descriptor into the step input, W_aux (d_model x 14).
    mutable Eigen::MatrixXd descriptor_weights_;
    // Hidden layer weights (hidden x d_model).
    mutable Eigen::MatrixXd hidden_weights_;
    // Hidden layer biases (hidden).
    mutable Eigen::VectorXd hidden_biases_;
    // Vocabulary logits weights (vocab x hidden).
    mutable Eigen::MatrixXd output_weights_;
    // Vocabulary logit biases (vocab).
    mutable Eigen::VectorXd output_biases_;
    // Adam state for each parameter tensor above.
    mutable AdamMoment token_moment_;
    mutable AdamMoment token2_moment_;
    mutable AdamMoment field_moment_;
    mutable AdamMoment query_moment_;
    mutable AdamMoment key_moment_;
    mutable AdamMoment value_moment_;
    mutable AdamMoment descriptor_moment_;
    mutable AdamMoment hidden_weights_moment_;
    mutable AdamMoment hidden_biases_moment_;
    mutable AdamMoment output_weights_moment_;
    mutable AdamMoment output_biases_moment_;
    // Device copy of the tensors and Adam state with the step buffers, and which side is newer (empty on the CPU).
    mutable DecoderDeviceSlot device_;
};

}  // namespace forward_analysis
