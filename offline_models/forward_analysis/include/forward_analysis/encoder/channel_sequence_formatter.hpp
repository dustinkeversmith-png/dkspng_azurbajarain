#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/config/feature_ablation_level.hpp"
#include "forward_analysis/encoder/channel_feature_extractor.hpp"
#include "forward_analysis/encoder/linear_input_adapter.hpp"
#include "forward_analysis/encoder/multi_family_token_batch.hpp"
#include "forward_analysis/encoder/multi_family_token_extractor.hpp"
#include "forward_analysis/features/feature_extractor_registry.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>

namespace forward_analysis {

// Projects multi-family telemetry tokens (Channel, Sample, Relationship, Layer, Global) into the Perceiver's
// d_model sequence space using family-specific LinearInputAdapters and learned token-family type embeddings,
// while preserving the baseline 25-feature channel_features() and format_sequence(features) API.
class ChannelSequenceFormatter {
public:
    explicit ChannelSequenceFormatter(const DistillerConfig& config = DistillerConfig{},
                                      std::uint64_t seed = 1337ULL)
        : config_(config),
          adapter_(config.d_model, MultiFamilyTokenExtractor::kExpandedChannelDim, seed),
          sample_adapter_(config.d_model, MultiFamilyTokenExtractor::kSampleTokenDim, seed + 101ULL),
          relationship_adapter_(config.d_model, MultiFamilyTokenExtractor::kRelationshipTokenDim, seed + 202ULL),
          layer_adapter_(config.d_model, MultiFamilyTokenExtractor::kLayerTokenDim, seed + 303ULL),
          global_adapter_(config.d_model, MultiFamilyTokenExtractor::kGlobalTokenDim, seed + 404ULL) {
        config_.validate();
        // Random generator for token-family embeddings.
        DeterministicRng rng(seed + 505ULL);
        family_embeddings_ = rng.xavier_matrix(5, config.d_model) * 0.25;
    }

    // Baseline 25-feature per-channel matrix (C x 25).
    Eigen::MatrixXd channel_features(const ForwardTelemetryTensor& telemetry) const {
        return ChannelFeatureExtractor::extract(telemetry, config_);
    }

    // Multi-family token batch extracted according to config_.feature_ablation_level.
    MultiFamilyTokenBatch extract_token_batch(const ForwardTelemetryTensor& telemetry) const {
        return MultiFamilyTokenExtractor::extract(telemetry, config_, feature_registry_);
    }

    // Projects a single channel feature matrix (C x F) into (C x d_model).
    Eigen::MatrixXd format_sequence(const Eigen::MatrixXd& features) {
        return adapter_.project(features);
    }

    // Projects all active token families of a MultiFamilyTokenBatch into one sequence (T_total x d_model).
    Eigen::MatrixXd format_sequence(const MultiFamilyTokenBatch& batch) {
        if (config_.feature_ablation_level == FeatureAblationLevel::kBaseline25) {
            return adapter_.project(batch.channel_tokens);
        }
        // Total tokens across all active families.
        const Eigen::Index total = static_cast<Eigen::Index>(batch.total_tokens());
        // Model embedding width.
        const Eigen::Index d = static_cast<Eigen::Index>(config_.d_model);
        // Combined projected sequence matrix.
        Eigen::MatrixXd sequence(total, d);
        // Current row offset in `sequence`.
        Eigen::Index offset = 0;

        if (batch.channel_tokens.rows() > 0) {
            // Projected channel tokens.
            Eigen::MatrixXd proj = adapter_.project(batch.channel_tokens);
            proj.rowwise() += family_embeddings_.row(0);
            sequence.middleRows(offset, proj.rows()) = proj;
            offset += proj.rows();
        }
        if (batch.sample_tokens.rows() > 0) {
            // Projected sample tokens scaled to balance family size against channel tokens.
            Eigen::MatrixXd proj = sample_adapter_.project(batch.sample_tokens) * 2.0;
            proj.rowwise() += family_embeddings_.row(1);
            sequence.middleRows(offset, proj.rows()) = proj;
            offset += proj.rows();
        }
        if (batch.relationship_tokens.rows() > 0) {
            // Projected relationship tokens scaled to balance family size against channel tokens.
            Eigen::MatrixXd proj = relationship_adapter_.project(batch.relationship_tokens) * 3.0;
            proj.rowwise() += family_embeddings_.row(2);
            sequence.middleRows(offset, proj.rows()) = proj;
            offset += proj.rows();
        }
        if (batch.layer_tokens.rows() > 0) {
            // Projected layer tokens, masked by layer_valid_mask for missing layers.
            Eigen::MatrixXd proj = layer_adapter_.project(batch.layer_tokens) * 4.0;
            proj.rowwise() += family_embeddings_.row(3);
            for (Eigen::Index r = 0; r < proj.rows() && r < batch.layer_valid_mask.size(); ++r) {
                proj.row(r) *= batch.layer_valid_mask(r);
            }
            sequence.middleRows(offset, proj.rows()) = proj;
            offset += proj.rows();
        }
        if (batch.global_tokens.rows() > 0) {
            // Projected global domain summary tokens scaled to balance single-row cardinality.
            Eigen::MatrixXd proj = global_adapter_.project(batch.global_tokens) * 6.0;
            proj.rowwise() += family_embeddings_.row(4);
            sequence.middleRows(offset, proj.rows()) = proj;
            offset += proj.rows();
        }
        return sequence;
    }

    // Formats telemetry directly into the Perceiver input sequence according to config_.feature_ablation_level.
    Eigen::MatrixXd format_sequence(const ForwardTelemetryTensor& telemetry) {
        return format_sequence(extract_token_batch(telemetry));
    }

    // Updates the single channel adapter from a sequence gradient.
    void update_adapter(const Eigen::MatrixXd& features, const Eigen::MatrixXd& d_sequence, double learning_rate) {
        adapter_.update_from_sequence_grad(features, d_sequence, learning_rate);
    }

    // Updates all active family adapters and token-family embeddings from the Perceiver input gradient.
    void update_adapters(const MultiFamilyTokenBatch& batch,
                         const Eigen::MatrixXd& d_sequence,
                         double learning_rate) {
        if (learning_rate <= 0.0 || d_sequence.rows() == 0) {
            return;
        }
        if (config_.feature_ablation_level == FeatureAblationLevel::kBaseline25) {
            adapter_.update_from_sequence_grad(batch.channel_tokens, d_sequence, learning_rate);
            return;
        }
        // Current row offset in `d_sequence`.
        Eigen::Index offset = 0;
        // Clamped learning rate for family embeddings.
        const double emb_lr = learning_rate * 0.25;

        if (batch.channel_tokens.rows() > 0 && offset + batch.channel_tokens.rows() <= d_sequence.rows()) {
            // Gradient slice for channel tokens.
            const Eigen::MatrixXd grad = d_sequence.middleRows(offset, batch.channel_tokens.rows());
            adapter_.update_from_sequence_grad(batch.channel_tokens, grad, learning_rate);
            family_embeddings_.row(0) -= emb_lr * grad.colwise().mean();
            offset += batch.channel_tokens.rows();
        }
        if (batch.sample_tokens.rows() > 0 && offset + batch.sample_tokens.rows() <= d_sequence.rows()) {
            // Gradient slice for sample tokens.
            const Eigen::MatrixXd grad = d_sequence.middleRows(offset, batch.sample_tokens.rows());
            sample_adapter_.update_from_sequence_grad(batch.sample_tokens, grad * 2.0, learning_rate);
            family_embeddings_.row(1) -= emb_lr * grad.colwise().mean();
            offset += batch.sample_tokens.rows();
        }
        if (batch.relationship_tokens.rows() > 0 && offset + batch.relationship_tokens.rows() <= d_sequence.rows()) {
            // Gradient slice for relationship tokens.
            const Eigen::MatrixXd grad = d_sequence.middleRows(offset, batch.relationship_tokens.rows());
            relationship_adapter_.update_from_sequence_grad(batch.relationship_tokens, grad * 3.0, learning_rate);
            family_embeddings_.row(2) -= emb_lr * grad.colwise().mean();
            offset += batch.relationship_tokens.rows();
        }
        if (batch.layer_tokens.rows() > 0 && offset + batch.layer_tokens.rows() <= d_sequence.rows()) {
            // Gradient slice for layer tokens, masked for missing layers.
            Eigen::MatrixXd grad = d_sequence.middleRows(offset, batch.layer_tokens.rows());
            for (Eigen::Index r = 0; r < grad.rows() && r < batch.layer_valid_mask.size(); ++r) {
                grad.row(r) *= batch.layer_valid_mask(r);
            }
            layer_adapter_.update_from_sequence_grad(batch.layer_tokens, grad * 4.0, learning_rate);
            family_embeddings_.row(3) -= emb_lr * grad.colwise().mean();
            offset += batch.layer_tokens.rows();
        }
        if (batch.global_tokens.rows() > 0 && offset + batch.global_tokens.rows() <= d_sequence.rows()) {
            // Gradient slice for global domain tokens.
            const Eigen::MatrixXd grad = d_sequence.middleRows(offset, batch.global_tokens.rows());
            global_adapter_.update_from_sequence_grad(batch.global_tokens, grad * 6.0, learning_rate);
            family_embeddings_.row(4) -= emb_lr * grad.colwise().mean();
            offset += batch.global_tokens.rows();
        }
    }

    const LinearInputAdapter& adapter() const noexcept { return adapter_; }
    const FeatureExtractorRegistry& feature_registry() const noexcept { return feature_registry_; }
    FeatureExtractorRegistry& feature_registry() noexcept { return feature_registry_; }

private:
    // Thresholds, ablation level, and model width.
    DistillerConfig config_;
    // Registry of mathematical domain feature extractors.
    FeatureExtractorRegistry feature_registry_;
    // Learned projection from channel features (25 or 38) to d_model.
    LinearInputAdapter adapter_;
    // Learned projection from sample tokens (16) to d_model.
    LinearInputAdapter sample_adapter_;
    // Learned projection from relationship tokens (16) to d_model.
    LinearInputAdapter relationship_adapter_;
    // Learned projection from layer tokens (16) to d_model.
    LinearInputAdapter layer_adapter_;
    // Learned projection from global domain tokens (16) to d_model.
    LinearInputAdapter global_adapter_;
    // Learned embedding per TokenFamily (5 x d_model).
    Eigen::MatrixXd family_embeddings_;
};

}  // namespace forward_analysis
