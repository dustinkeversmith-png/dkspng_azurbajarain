#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/analysis/observation_extractor.hpp"
#include "forward_analysis/bridge/bridge_checkpoint.hpp"
#include "forward_analysis/bridge/distillation_result.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/dataset/training_sample.hpp"
#include "forward_analysis/decoder/autoregressive_causal_decoder.hpp"
#include "forward_analysis/decoder/field_delexicalizer.hpp"
#include "forward_analysis/decoder/observation_delexicalizer.hpp"
#include "forward_analysis/decoder/shape_slot_lexicon.hpp"
#include "forward_analysis/description/measured_facts_narration.hpp"
#include "forward_analysis/description/observation_consistency.hpp"
#include "forward_analysis/encoder/channel_sequence_formatter.hpp"
#include "forward_analysis/encoder/perceiver_io_encoder.hpp"
#include "forward_analysis/heads/auxiliary_alignment_heads.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"
#include "forward_analysis/training/finetune_outcome.hpp"
#include "forward_analysis/training/loss_breakdown.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace forward_analysis {

// Multi-family tokens (Channel, Sample, Relationship, Layer, Global) -> typed adapters -> Perceiver IO latents Z,
// which feed both the domain-aware NumericalPredictionHead registry / AuxiliaryAlignmentHeads and the
// field-conditioned causal decoder.
class DualObjectiveDistillationBridge {
public:
    explicit DualObjectiveDistillationBridge(const DistillerConfig& config = DistillerConfig{},
                                             std::uint64_t seed = 20261013ULL)
        : config_(config),
          formatter_(config, seed + 10ULL),
          perceiver_(config, seed + 20ULL),
          aux_heads_(config, seed + 30ULL),
          decoder_(config, seed + 40ULL) {
        config_.validate();
    }

    // Ground-truth field f of a sample with its shape and index references replaced by slot tokens bound to
    // that sample's measured facts, e.g. "<COUNT:DEAD_L1|W> of the <DIM:W_1|w> <ORD:LAYER_1|wl> units ...".
    static std::vector<std::string> target_lexemes(const TrainingSample& sample,
                                                   std::size_t field,
                                                   SlotBinding binding = SlotBinding::kShapeFacts) {
        return FieldDelexicalizer::delexicalize(
            sample.ground_truth_targets.observations.field(field), sample.measured, binding);
    }

    void prepare_vocabulary(const std::vector<TrainingSample>& dataset) {
        // Delexicalized field sequences across all samples.
        std::vector<std::vector<std::string>> fields;
        for (const TrainingSample& sample : dataset) {
            // Field index in [0, 4).
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                fields.push_back(target_lexemes(sample, f, config_.slot_binding));
            }
        }
        decoder_.prepare_vocabulary(fields);
    }

    LossBreakdown pretrain_representation_step(const TrainingSample& sample, double learning_rate) {
        decoder_.set_frozen(true);
        return train_step(sample, learning_rate, false);
    }

    LossBreakdown finetune_end_to_end_step(const TrainingSample& sample, double learning_rate) {
        decoder_.set_frozen(false);
        return train_step(sample, learning_rate, true);
    }

    std::vector<LossBreakdown> run_pretraining_phase(const std::vector<TrainingSample>& dataset,
                                                     std::size_t epochs,
                                                     double learning_rate) {
        // Per-epoch training loss history.
        std::vector<LossBreakdown> history;
        if (dataset.empty()) {
            return history;
        }
        // Pre-extracted multi-family token batches for `dataset`.
        const std::vector<MultiFamilyTokenBatch> batches = extract_batches(dataset);
        // Epoch index.
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            history.push_back(run_epoch_cached(dataset, batches, learning_rate, false));
        }
        return history;
    }

    // Up to max_epochs of end-to-end training. After every epoch the held-out losses are measured (on the
    // training set when held_out is empty); training stops once held-out token CE has not decreased for
    // config.early_stopping_patience consecutive epochs, and the weights of the epoch with the lowest
    // held-out total loss are restored.
    FinetuneOutcome run_finetuning_phase(const std::vector<TrainingSample>& dataset,
                                         std::size_t max_epochs,
                                         double learning_rate,
                                         const std::vector<TrainingSample>& held_out) {
        // Aggregated finetuning outcome.
        FinetuneOutcome outcome;
        outcome.max_epochs = max_epochs;
        if (dataset.empty()) {
            return outcome;
        }
        // Validation set reference (falls back to training set when held_out is empty).
        const std::vector<TrainingSample>& validation = held_out.empty() ? dataset : held_out;
        // Pre-extracted token batches for the training set.
        const std::vector<MultiFamilyTokenBatch> train_batches = extract_batches(dataset);
        // Pre-extracted token batches for the validation set.
        const std::vector<MultiFamilyTokenBatch> val_batches =
            held_out.empty() ? train_batches : extract_batches(validation);
        // Best held-out cross-entropy observed so far.
        double best_ce = std::numeric_limits<double>::infinity();
        // Best held-out total loss observed so far.
        double best_total = std::numeric_limits<double>::infinity();
        // Consecutive epochs without a decrease in held-out cross-entropy.
        std::size_t epochs_without_ce_decrease = 0;
        // Best model checkpoint.
        BridgeCheckpoint best = checkpoint();
        // 1-indexed epoch counter.
        for (std::size_t epoch = 1; epoch <= max_epochs; ++epoch) {
            // Record for the current epoch.
            EpochRecord record;
            record.epoch = epoch;
            record.train = run_epoch_cached(dataset, train_batches, learning_rate, true);
            record.eval = evaluate_dataset_cached(validation, val_batches);
            if (record.eval.total_loss < best_total) {
                best_total = record.eval.total_loss;
                best = checkpoint();
                outcome.best_epoch = epoch;
                outcome.best_eval = record.eval;
                record.best_so_far = true;
            }
            outcome.history.push_back(record);
            if (record.eval.token_cross_entropy < best_ce) {
                best_ce = record.eval.token_cross_entropy;
                epochs_without_ce_decrease = 0;
            } else if (++epochs_without_ce_decrease >= config_.early_stopping_patience) {
                outcome.stopped_early = epoch < max_epochs;
                break;
            }
        }
        restore(best);
        return outcome;
    }

    // Losses of one sample without updating anything (decoder in eval mode, conditioned on the domain heads).
    LossBreakdown evaluate_sample(const TrainingSample& sample) {
        return evaluate_sample_cached(sample, formatter_.extract_token_batch(sample.telemetry));
    }

    // Mean losses over a set of samples without updating anything (e.g. held-out records after each epoch).
    LossBreakdown evaluate_dataset(const std::vector<TrainingSample>& dataset) {
        return evaluate_dataset_cached(dataset, extract_batches(dataset));
    }

    Eigen::MatrixXd encode(const ForwardTelemetryTensor& telemetry) {
        return perceiver_.encode(formatter_.format_sequence(telemetry));
    }

    DistillationResult distill_telemetry(const ForwardTelemetryTensor& telemetry) {
        // Output distillation bundle.
        DistillationResult result;
        // Extracted multi-family token batch.
        const MultiFamilyTokenBatch batch = formatter_.extract_token_batch(telemetry);
        result.channel_count = static_cast<std::size_t>(batch.channel_tokens.rows());
        result.encoder_token_count = batch.total_tokens();
        result.latent_embeddings = perceiver_.encode(formatter_.format_sequence(batch));
        result.measured = ObservationExtractor::extract(telemetry, config_);

        // Predictions from all registered domain heads and legacy heads on Z.
        const AuxiliaryEvaluation aux = aux_heads_.evaluate(result.latent_embeddings);
        result.predicted_observation_descriptor = aux.predicted_observation_descriptor;
        result.decoder_conditioning_vector = aux.decoder_conditioning_vector;
        result.predicted_eigen_signs = aux.predicted_eigen_signs;
        result.domain_predictions = aux.domain_predictions;

        // Lexicon of measured shape/count slots for slot filling.
        const ShapeSlotLexicon slots = ShapeSlotLexicon::from(result.measured);
        result.narrated_observations = MeasuredFactsNarration::describe(result.measured, config_);
        // Reference to the fallback deterministic narration.
        const ObservationText& narrated = result.narrated_observations;
        // Field index in [0, 4).
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            // Generated token ids for field `f`.
            const std::vector<int> ids = decoder_.generate_field_tokens(
                result.latent_embeddings, result.decoder_conditioning_vector, f, config_.max_field_tokens);
            // Decoded lexeme strings.
            const std::vector<std::string> lexemes = decoder_.tokenizer().decode_lexemes(ids);
            result.decoder_token_counts[f] = ids.size();
            result.decoder_slot_observations.field(f) = LexemeSplitter::join(lexemes);
            // Relexicalized sentence string.
            std::string filled;
            result.decoder_slots_filled[f] = ObservationDelexicalizer::relexicalize(lexemes, slots, filled);
            result.decoder_observations.field(f) =
                result.decoder_slots_filled[f] ? filled : result.decoder_slot_observations.field(f);
            result.decoder_field_consistent[f] =
                result.decoder_slots_filled[f] && ObservationConsistency::verify_field(filled, result.measured);
            result.observations.field(f) =
                result.decoder_field_consistent[f] ? result.decoder_observations.field(f) : narrated.field(f);
        }
        return result;
    }

    const DistillerConfig& config() const noexcept { return config_; }
    const ChannelSequenceFormatter& formatter() const noexcept { return formatter_; }
    const PerceiverIOEncoder& perceiver() const noexcept { return perceiver_; }
    const AuxiliaryAlignmentHeads& auxiliary_heads() const noexcept { return aux_heads_; }
    const AutoregressiveCausalDecoder& decoder() const noexcept { return decoder_; }

private:
    BridgeCheckpoint checkpoint() const {
        decoder_.synchronize_host();
        return BridgeCheckpoint{formatter_, perceiver_, aux_heads_, decoder_};
    }

    void restore(const BridgeCheckpoint& saved) {
        formatter_ = saved.formatter;
        perceiver_ = saved.perceiver;
        aux_heads_ = saved.aux_heads;
        decoder_ = saved.decoder;
    }

    std::vector<MultiFamilyTokenBatch> extract_batches(const std::vector<TrainingSample>& dataset) const {
        // Pre-extracted token batches for all samples in `dataset`.
        std::vector<MultiFamilyTokenBatch> batches;
        batches.reserve(dataset.size());
        for (const TrainingSample& sample : dataset) {
            batches.push_back(formatter_.extract_token_batch(sample.telemetry));
        }
        return batches;
    }

    LossBreakdown evaluate_sample_cached(const TrainingSample& sample, const MultiFamilyTokenBatch& batch) {
        // Latent bottleneck Z (M x d_model).
        const Eigen::MatrixXd latents = perceiver_.encode(formatter_.format_sequence(batch));
        // Evaluation of all auxiliary and domain prediction heads.
        const AuxiliaryEvaluation aux = aux_heads_.evaluate(
            latents,
            ObservationDescriptor::encode(sample.measured),
            sample.ground_truth_targets.eigen_signs,
            &sample.measured);
        // Mean cross-entropy across the four observation fields.
        double ce = 0.0;
        // Field index in [0, 4).
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            ce += decoder_.compute_loss(
                latents, aux.decoder_conditioning_vector, f, target_lexemes(sample, f, config_.slot_binding));
        }
        ce /= static_cast<double>(ObservationText::kFieldCount);
        return LossBreakdown::combine(ce, aux.aux1_observation_loss, aux.aux2_eigen_sign_loss, config_);
    }

    LossBreakdown evaluate_dataset_cached(const std::vector<TrainingSample>& dataset,
                                          const std::vector<MultiFamilyTokenBatch>& batches) {
        // Accumulated cross-entropy.
        double ce = 0.0;
        // Accumulated Aux1 loss.
        double aux1 = 0.0;
        // Accumulated Aux2 loss.
        double aux2 = 0.0;
        // Sample index.
        for (std::size_t i = 0; i < dataset.size(); ++i) {
            // Sample loss breakdown.
            const LossBreakdown loss = evaluate_sample_cached(dataset[i], batches[i]);
            ce += loss.token_cross_entropy;
            aux1 += loss.aux1_observation_loss;
            aux2 += loss.aux2_eigen_sign_loss;
        }
        // Reciprocal sample count.
        const double inv_n = dataset.empty() ? 0.0 : 1.0 / static_cast<double>(dataset.size());
        return LossBreakdown::combine(ce * inv_n, aux1 * inv_n, aux2 * inv_n, config_);
    }

    LossBreakdown train_step(const TrainingSample& sample, double learning_rate, bool train_decoder) {
        return train_step_cached(sample, formatter_.extract_token_batch(sample.telemetry), learning_rate, train_decoder);
    }

    LossBreakdown train_step_cached(const TrainingSample& sample,
                                    const MultiFamilyTokenBatch& batch,
                                    double learning_rate,
                                    bool train_decoder) {
        // Forward cache of the Perceiver IO encoder.
        const PerceiverForwardCache cache = perceiver_.forward_with_cache(formatter_.format_sequence(batch));
        // Zero gradient matrix wrt Z (M x d_model).
        const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(config_.num_latents), static_cast<Eigen::Index>(config_.d_model));

        // Mean token cross-entropy across the four observation fields.
        double ce = 0.0;
        // Accumulated gradient wrt Z (M x d_model).
        Eigen::MatrixXd d_latents = zero;
        if (train_decoder) {
            // Domain-head conditioning vector d_hat (14-dim).
            const Eigen::VectorXd conditioning_descriptor =
                aux_heads_.evaluate(cache.latents_z).decoder_conditioning_vector;
            // Decoder gradient wrt Z accumulated over the four fields.
            Eigen::MatrixXd d_decoder = zero;
            // Field index in [0, 4).
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                ce += decoder_.compute_loss_and_step(cache.latents_z,
                                                     conditioning_descriptor,
                                                     f,
                                                     target_lexemes(sample, f, config_.slot_binding),
                                                     d_decoder);
            }
            ce /= static_cast<double>(ObservationText::kFieldCount);
            d_latents += d_decoder / static_cast<double>(ObservationText::kFieldCount);
        }

        // Auxiliary and domain head losses + parameter updates + gradient into d_latents.
        const AuxiliaryEvaluation aux = aux_heads_.compute_losses_and_step(
            cache.latents_z,
            ObservationDescriptor::encode(sample.measured),
            sample.ground_truth_targets.eigen_signs,
            learning_rate,
            d_latents,
            &sample.measured);

        // Gradient wrt the input token sequence from the Perceiver encoder backward pass.
        const Eigen::MatrixXd d_sequence = perceiver_.backward_and_update(cache, d_latents, learning_rate);
        formatter_.update_adapters(batch, d_sequence, learning_rate);
        return LossBreakdown::combine(ce, aux.aux1_observation_loss, aux.aux2_eigen_sign_loss, config_);
    }

    LossBreakdown run_epoch_cached(const std::vector<TrainingSample>& dataset,
                                   const std::vector<MultiFamilyTokenBatch>& batches,
                                   double learning_rate,
                                   bool train_decoder) {
        decoder_.set_frozen(!train_decoder);
        // Sum of token cross-entropies across the epoch.
        double ce = 0.0;
        // Sum of Aux1 losses across the epoch.
        double aux1 = 0.0;
        // Sum of Aux2 losses across the epoch.
        double aux2 = 0.0;
        // Sample index.
        for (std::size_t i = 0; i < dataset.size(); ++i) {
            // Single-sample step loss breakdown.
            const LossBreakdown step = train_step_cached(dataset[i], batches[i], learning_rate, train_decoder);
            ce += step.token_cross_entropy;
            aux1 += step.aux1_observation_loss;
            aux2 += step.aux2_eigen_sign_loss;
        }
        // Reciprocal dataset size.
        const double inv_n = 1.0 / static_cast<double>(dataset.size());
        return LossBreakdown::combine(ce * inv_n, aux1 * inv_n, aux2 * inv_n, config_);
    }

    // Shared model dimensions, decoder budget and learning rate, loss weights and measurement thresholds.
    DistillerConfig config_;
    // Turns telemetry of any shape into the multi-family token sequence (T x d_model).
    ChannelSequenceFormatter formatter_;
    // Compresses the token sequence into the latent bottleneck Z.
    PerceiverIOEncoder perceiver_;
    // Branch A: domain-specific prediction head registry, observation-descriptor head, and eigen-sign head on Z.
    AuxiliaryAlignmentHeads aux_heads_;
    // Branch B: generates the four observation fields from Z and the domain conditioning vector.
    AutoregressiveCausalDecoder decoder_;
};

}  // namespace forward_analysis
