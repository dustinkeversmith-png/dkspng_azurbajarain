#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/heads/activation_sparsity_head.hpp"
#include "forward_analysis/heads/eigen_sign_prediction_head.hpp"
#include "forward_analysis/heads/encoded_representation.hpp"
#include "forward_analysis/heads/feature_association_head.hpp"
#include "forward_analysis/heads/layer_structure_head.hpp"
#include "forward_analysis/heads/loss_result.hpp"
#include "forward_analysis/heads/narrow_representation_head.hpp"
#include "forward_analysis/heads/numerical_prediction_head.hpp"
#include "forward_analysis/heads/numerical_rank_head.hpp"
#include "forward_analysis/heads/numerical_targets.hpp"
#include "forward_analysis/heads/output_alignment_head.hpp"
#include "forward_analysis/heads/prediction_result.hpp"
#include "forward_analysis/heads/spectral_profile_head.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace forward_analysis {

// Registry of modular, domain-specific NumericalPredictionHeads across all six mathematical domains.
class NumericalPredictionHeadRegistry {
public:
    explicit NumericalPredictionHeadRegistry(const DistillerConfig& config = DistillerConfig{},
                                             std::uint64_t seed = 3100ULL)
        : config_(config) {
        register_default_heads(seed);
    }

    // Deep-copy constructor so checkpoint saving/restoring preserves trained domain head parameters.
    NumericalPredictionHeadRegistry(const NumericalPredictionHeadRegistry& other)
        : config_(other.config_) {
        copy_from(other);
    }

    NumericalPredictionHeadRegistry& operator=(const NumericalPredictionHeadRegistry& other) {
        if (this != &other) {
            config_ = other.config_;
            copy_from(other);
        }
        return *this;
    }

    void register_default_heads(std::uint64_t seed = 3100ULL) {
        heads_.clear();
        sparsity_head_ = std::make_shared<ActivationSparsityHead>(config_.d_model, 0.25, seed + 1ULL);
        rank_head_ = std::make_shared<NumericalRankHead>(config_.d_model, 0.25, seed + 2ULL);
        narrow_head_ = std::make_shared<NarrowRepresentationHead>(config_.d_model, 0.35, seed + 3ULL);
        structure_head_ = std::make_shared<LayerStructureHead>(config_.d_model, 0.35, seed + 4ULL);
        alignment_head_ = std::make_shared<OutputAlignmentHead>(config_.d_model, 0.30, seed + 5ULL);
        association_head_ = std::make_shared<FeatureAssociationHead>(config_.d_model, 0.25, seed + 6ULL);
        eigen_head_ = std::make_shared<EigenSignPredictionHead>(
            config_.d_model, config_.eigen_sign_dim, config_.aux2_eigen_sign_weight, seed + 7ULL);
        spectral_head_ = std::make_shared<SpectralProfileHead>(
            config_.d_model, config_.aux2_eigen_sign_weight > 0.0 ? 0.20 : 0.0, seed + 8ULL);

        register_head(sparsity_head_);
        register_head(rank_head_);
        register_head(narrow_head_);
        register_head(structure_head_);
        register_head(alignment_head_);
        register_head(association_head_);
        register_head(eigen_head_);
        register_head(spectral_head_);
    }

    void register_head(std::shared_ptr<NumericalPredictionHead> head) {
        if (head != nullptr) {
            heads_.push_back(std::move(head));
        }
    }

    std::size_t size() const noexcept { return heads_.size(); }

    const std::vector<std::shared_ptr<NumericalPredictionHead>>& heads() const noexcept { return heads_; }

    // Runs forward() on all registered domain heads.
    std::map<std::string, PredictionResult> forward_all(const EncodedRepresentation& representation) const {
        // Predictions keyed by head id.
        std::map<std::string, PredictionResult> out;
        for (const std::shared_ptr<NumericalPredictionHead>& h : heads_) {
            out[h->id()] = h->forward(representation);
        }
        return out;
    }

    // Evaluates losses on all registered domain heads without updating parameters.
    std::map<std::string, LossResult> evaluate_losses(const std::map<std::string, PredictionResult>& predictions,
                                                      const NumericalTargets& targets) const {
        // Loss results keyed by head id.
        std::map<std::string, LossResult> out;
        for (const std::shared_ptr<NumericalPredictionHead>& h : heads_) {
            // Prediction entry for head `h`.
            const auto it = predictions.find(h->id());
            if (it != predictions.end()) {
                out[h->id()] = h->compute_loss(it->second, targets);
            }
        }
        return out;
    }

    // Runs forward + backward + Adam update across all registered domain heads and accumulates dLoss/dZ.
    std::map<std::string, LossResult> compute_losses_and_step(const EncodedRepresentation& representation,
                                                              const NumericalTargets& targets,
                                                              double learning_rate,
                                                              Eigen::MatrixXd& d_latents_z_accum) {
        // Loss results keyed by head id.
        std::map<std::string, LossResult> out;
        for (const std::shared_ptr<NumericalPredictionHead>& h : heads_) {
            out[h->id()] = h->compute_loss_and_step(representation, targets, learning_rate, d_latents_z_accum);
        }
        return out;
    }

    // Synthesizes the 14-component ObservationDescriptor from the calibrated domain head predictions.
    static Eigen::VectorXd synthesize_descriptor(const std::map<std::string, PredictionResult>& preds) {
        // Synthesized 14-value descriptor.
        Eigen::VectorXd desc = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(ObservationDescriptor::dim()));
        // Helper to read a calibrated component from `preds`.
        const auto get = [&preds](const std::string& head_id, Eigen::Index idx, double fallback = 0.0) {
            const auto it = preds.find(head_id);
            if (it == preds.end() || idx >= it->second.calibrated_outputs.size()) {
                return fallback;
            }
            return it->second.calibrated_outputs(idx);
        };
        // Whether the network is predicted to have 2 hidden layers.
        const double p_two_layer = get("layer_structure", 0, 0.5);

        desc(0) = get("activation_sparsity", 0);
        // Last-layer dead fraction: blend L1 and L2 dead fraction according to predicted layer count.
        desc(1) = (1.0 - p_two_layer) * get("activation_sparsity", 0) + p_two_layer * get("activation_sparsity", 3);
        desc(2) = get("numerical_rank", 0);
        desc(3) = get("narrow_representation", 3);
        desc(4) = get("activation_sparsity", 6);
        desc(5) = get("activation_sparsity", 7);
        desc(6) = get("feature_association", 0);
        desc(7) = get("feature_association", 1);
        desc(8) = get("feature_association", 2);
        desc(9) = get("output_alignment", 0);
        desc(10) = get("output_alignment", 1);
        desc(11) = 2.0 * get("output_alignment", 2);
        desc(12) = 2.0 * get("output_alignment", 3) - 1.0;
        desc(13) = get("output_alignment", 4);
        return desc;
    }

    // Synthesizes the 14-dimensional decoder conditioning vector packing key structural, geometric, spectral, and functional predictions.
    static Eigen::VectorXd synthesize_decoder_conditioning(const std::map<std::string, PredictionResult>& preds,
                                                           const Eigen::VectorXd& base_descriptor) {
        // Conditioning vector of width 14 passed to AutoregressiveCausalDecoder (W_aux d_hat).
        Eigen::VectorXd cond = base_descriptor;
        // Helper to read a calibrated component from `preds`.
        const auto get = [&preds](const std::string& head_id, Eigen::Index idx, double fallback = 0.0) {
            const auto it = preds.find(head_id);
            if (it == preds.end() || idx >= it->second.calibrated_outputs.size()) {
                return fallback;
            }
            return it->second.calibrated_outputs(idx);
        };
        if (cond.size() >= 14) {
            // Embed structural and geometric signals directly into slots unused by the batch-statistics template
            // while preserving smooth continuity with base_descriptor:
            //   slot 3: narrow representation probability (Domain C)
            //   slot 4: two-layer network probability (Domain F)
            //   slot 5: L1 numerical rank ratio (Domain B)
            //   slot 6: L2 numerical rank ratio (Domain B)
            //   slot 7: nonzero input association probability (Domain A)
            //   slot 8: strongest input association (Domain A)
            cond(3) = get("narrow_representation", 0, cond(3));
            cond(4) = get("layer_structure", 0, cond(4));
            cond(5) = get("numerical_rank", 1, cond(5));
            cond(6) = get("numerical_rank", 3, cond(6));
            cond(7) = get("feature_association", 4, cond(7));
            cond(8) = get("feature_association", 3, cond(8));
        }
        return cond;
    }

    const EigenSignHead& eigen_sign_head() const noexcept { return eigen_head_->inner_head(); }
    EigenSignHead& eigen_sign_head() noexcept { return eigen_head_->inner_head(); }

private:
    void copy_from(const NumericalPredictionHeadRegistry& other) {
        heads_.clear();
        if (other.sparsity_head_) {
            sparsity_head_ = std::make_shared<ActivationSparsityHead>(*other.sparsity_head_);
            register_head(sparsity_head_);
        }
        if (other.rank_head_) {
            rank_head_ = std::make_shared<NumericalRankHead>(*other.rank_head_);
            register_head(rank_head_);
        }
        if (other.narrow_head_) {
            narrow_head_ = std::make_shared<NarrowRepresentationHead>(*other.narrow_head_);
            register_head(narrow_head_);
        }
        if (other.structure_head_) {
            structure_head_ = std::make_shared<LayerStructureHead>(*other.structure_head_);
            register_head(structure_head_);
        }
        if (other.alignment_head_) {
            alignment_head_ = std::make_shared<OutputAlignmentHead>(*other.alignment_head_);
            register_head(alignment_head_);
        }
        if (other.association_head_) {
            association_head_ = std::make_shared<FeatureAssociationHead>(*other.association_head_);
            register_head(association_head_);
        }
        if (other.eigen_head_) {
            eigen_head_ = std::make_shared<EigenSignPredictionHead>(*other.eigen_head_);
            register_head(eigen_head_);
        }
        if (other.spectral_head_) {
            spectral_head_ = std::make_shared<SpectralProfileHead>(*other.spectral_head_);
            register_head(spectral_head_);
        }
    }

    // Shared distiller configuration.
    DistillerConfig config_;
    // Ordered list of all registered NumericalPredictionHead instances.
    std::vector<std::shared_ptr<NumericalPredictionHead>> heads_;
    // Typed pointers to the 8 built-in domain heads for deep copying and direct inspection.
    std::shared_ptr<ActivationSparsityHead> sparsity_head_;
    std::shared_ptr<NumericalRankHead> rank_head_;
    std::shared_ptr<NarrowRepresentationHead> narrow_head_;
    std::shared_ptr<LayerStructureHead> structure_head_;
    std::shared_ptr<OutputAlignmentHead> alignment_head_;
    std::shared_ptr<FeatureAssociationHead> association_head_;
    std::shared_ptr<EigenSignPredictionHead> eigen_head_;
    std::shared_ptr<SpectralProfileHead> spectral_head_;
};

}  // namespace forward_analysis
