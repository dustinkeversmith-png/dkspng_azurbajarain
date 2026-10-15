#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/distiller_config.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>

namespace forward_analysis {

// Ground-truth numerical targets and missing-label validity masks for all registered NumericalPredictionHeads.
struct NumericalTargets {
    // Target vectors keyed by head id.
    std::map<std::string, Eigen::VectorXd> targets;
    // Per-entry validity masks (1.0 = valid label, 0.0 = missing/not applicable) keyed by head id.
    std::map<std::string, Eigen::VectorXd> masks;

    bool has(const std::string& head_id) const {
        return targets.find(head_id) != targets.end();
    }

    static NumericalTargets from_observations(const NetworkObservations& obs,
                                              const Eigen::VectorXd& eigen_signs = Eigen::VectorXd(),
                                              const DistillerConfig& config = DistillerConfig{}) {
        // Assembled multi-head numerical targets.
        NumericalTargets out;
        // Number of hidden layers in `obs`.
        const std::size_t num_layers = obs.layers.size();
        // Sample count as double.
        const double n = static_cast<double>(std::max<std::size_t>(1, obs.shape.num_samples));

        // 1. ActivationSparsityHead ("activation_sparsity", dim = 8):
        //    [0] L1 dead_frac, [1] L1 active_frac, [2] L1 nonpos_frac,
        //    [3] L2 dead_frac, [4] L2 active_frac, [5] L2 nonpos_frac (masked if 1-layer),
        //    [6] gating_present, [7] closed_samples_frac
        Eigen::VectorXd act_t = Eigen::VectorXd::Zero(8);
        // Validity mask for activation_sparsity.
        Eigen::VectorXd act_m = Eigen::VectorXd::Ones(8);
        if (num_layers >= 1) {
            // First hidden layer observations.
            const LayerObservation& l1 = obs.layers.front();
            // Width of first hidden layer.
            const double w1 = static_cast<double>(std::max<std::size_t>(1, l1.width));
            act_t(0) = static_cast<double>(l1.dead_units.size()) / w1;
            act_t(1) = static_cast<double>(l1.always_active_units.size()) / w1;
            act_t(2) = static_cast<double>(l1.nonpositive_samples) / n;
        }
        if (num_layers >= 2) {
            // Second hidden layer observations.
            const LayerObservation& l2 = obs.layers[1];
            // Width of second hidden layer.
            const double w2 = static_cast<double>(std::max<std::size_t>(1, l2.width));
            act_t(3) = static_cast<double>(l2.dead_units.size()) / w2;
            act_t(4) = static_cast<double>(l2.always_active_units.size()) / w2;
            act_t(5) = static_cast<double>(l2.nonpositive_samples) / n;
        } else {
            // Mask out second-layer targets on 1-layer networks instead of supervising artificial zeros.
            act_m(3) = 0.0;
            act_m(4) = 0.0;
            act_m(5) = 0.0;
        }
        act_t(6) = obs.gating.present() ? 1.0 : 0.0;
        act_t(7) = static_cast<double>(obs.gating.closed_samples.size()) / n;
        out.targets["activation_sparsity"] = act_t;
        out.masks["activation_sparsity"] = act_m;

        // 2. NumericalRankHead ("numerical_rank", dim = 4):
        //    [0] L1 effective_rank / live_units, [1] L1 numerical_rank / 16,
        //    [2] L2 effective_rank / live_units, [3] L2 numerical_rank / 16 (masked if 1-layer)
        Eigen::VectorXd rank_t = Eigen::VectorXd::Zero(4);
        // Validity mask for numerical_rank.
        Eigen::VectorXd rank_m = Eigen::VectorXd::Ones(4);
        if (num_layers >= 1) {
            // First hidden layer observations.
            const LayerObservation& l1 = obs.layers.front();
            rank_t(0) = static_cast<double>(l1.effective_rank) /
                        static_cast<double>(std::max<std::size_t>(1, l1.live_count()));
            rank_t(1) = std::min(1.0, static_cast<double>(l1.numerical_rank) / 16.0);
        }
        if (num_layers >= 2) {
            // Second hidden layer observations.
            const LayerObservation& l2 = obs.layers[1];
            rank_t(2) = static_cast<double>(l2.effective_rank) /
                        static_cast<double>(std::max<std::size_t>(1, l2.live_count()));
            rank_t(3) = std::min(1.0, static_cast<double>(l2.numerical_rank) / 16.0);
        } else {
            rank_m(2) = 0.0;
            rank_m(3) = 0.0;
        }
        out.targets["numerical_rank"] = rank_t;
        out.masks["numerical_rank"] = rank_m;

        // 3. NarrowRepresentationHead ("narrow_representation", dim = 4):
        //    [0] network narrow_representation (0/1), [1] L1 narrow (0/1),
        //    [2] L2 narrow (0/1, masked if 1-layer), [3] L1 lies_on_ray (0/1)
        Eigen::VectorXd narrow_t = Eigen::VectorXd::Zero(4);
        // Validity mask for narrow_representation.
        Eigen::VectorXd narrow_m = Eigen::VectorXd::Ones(4);
        narrow_t(0) = obs.narrow_representation ? 1.0 : 0.0;
        if (num_layers >= 1) {
            narrow_t(1) = obs.layers.front().numerical_rank <= config.narrow_rank_limit ? 1.0 : 0.0;
            narrow_t(3) = obs.layers.front().lies_on_ray ? 1.0 : 0.0;
        }
        if (num_layers >= 2) {
            narrow_t(2) = obs.layers[1].numerical_rank <= config.narrow_rank_limit ? 1.0 : 0.0;
        } else {
            narrow_m(2) = 0.0;
        }
        out.targets["narrow_representation"] = narrow_t;
        out.masks["narrow_representation"] = narrow_m;

        // 4. LayerStructureHead ("layer_structure", dim = 6):
        //    [0] is_two_layer (0/1), [1] W_1 / 48, [2] W_2 / 48 (masked if 1-layer),
        //    [3] D_IN / 32, [4] D_OUT / 16, [5] N / 16
        Eigen::VectorXd struct_t = Eigen::VectorXd::Zero(6);
        // Validity mask for layer_structure.
        Eigen::VectorXd struct_m = Eigen::VectorXd::Ones(6);
        struct_t(0) = num_layers >= 2 ? 1.0 : 0.0;
        struct_t(1) = num_layers >= 1 ? std::min(1.0, static_cast<double>(obs.layers[0].width) / 48.0) : 0.0;
        if (num_layers >= 2) {
            struct_t(2) = std::min(1.0, static_cast<double>(obs.layers[1].width) / 48.0);
        } else {
            struct_m(2) = 0.0;
        }
        struct_t(3) = std::min(1.0, static_cast<double>(obs.shape.input_dim) / 32.0);
        struct_t(4) = std::min(1.0, static_cast<double>(obs.shape.output_dim) / 16.0);
        struct_t(5) = std::min(1.0, n / 16.0);
        out.targets["layer_structure"] = struct_t;
        out.masks["layer_structure"] = struct_m;

        // 5. OutputAlignmentHead ("output_alignment", dim = 9):
        //    [0] sign_preserved_frac, [1] order_preserved_frac, [2] mean_amplitude_ratio / 2,
        //    [3] 0.5 * (mean_correlation + 1), [4] flat_frac,
        //    [5] smaller_range_frac, [6] larger_range_frac, [7] comparable_range_frac, [8] sign_agreement_rate
        Eigen::VectorXd align_t = Eigen::VectorXd::Zero(9);
        // Validity mask for output_alignment.
        Eigen::VectorXd align_m = Eigen::VectorXd::Ones(9);
        if (!obs.outputs.empty()) {
            // Reciprocal of output count.
            const double inv_o = 1.0 / static_cast<double>(obs.outputs.size());
            for (const OutputObservation& o : obs.outputs) {
                align_t(0) += (o.sign_preserved ? 1.0 : 0.0) * inv_o;
                align_t(1) += (o.order_preserved ? 1.0 : 0.0) * inv_o;
                align_t(2) += 0.5 * std::min(2.0, std::max(0.0, o.amplitude_ratio)) * inv_o;
                align_t(3) += 0.5 * (o.correlation + 1.0) * inv_o;
                align_t(4) += (o.flat ? 1.0 : 0.0) * inv_o;
            }
            align_t(5) = static_cast<double>(obs.outputs_with_smaller_range) * inv_o;
            align_t(6) = static_cast<double>(obs.outputs_with_larger_range) * inv_o;
            align_t(7) = static_cast<double>(obs.outputs_with_comparable_range) * inv_o;
            align_t(8) = obs.sign_entries > 0
                             ? static_cast<double>(obs.sign_agreements) / static_cast<double>(obs.sign_entries)
                             : 0.0;
        }
        out.targets["output_alignment"] = align_t;
        out.masks["output_alignment"] = align_m;

        // 6. FeatureAssociationHead ("feature_association", dim = 5):
        //    [0] has_gate_coordinate, [1] leading_driver_count / 4, [2] strongest_driver_corr,
        //    [3] strongest_association, [4] nonzero_association_flag
        Eigen::VectorXd assoc_t = Eigen::VectorXd::Zero(5);
        // Validity mask for feature_association.
        Eigen::VectorXd assoc_m = Eigen::VectorXd::Ones(5);
        assoc_t(0) = obs.gating.has_gate_coordinate ? 1.0 : 0.0;
        assoc_t(1) = std::min(1.0, static_cast<double>(obs.driver_coordinates(0).size()) / 4.0);
        assoc_t(2) = obs.drivers.empty() ? 0.0 : obs.drivers.front().strength;
        assoc_t(3) = obs.strongest_association;
        assoc_t(4) = obs.strongest_association > config.dead_activation_threshold ? 1.0 : 0.0;
        out.targets["feature_association"] = assoc_t;
        out.masks["feature_association"] = assoc_m;

        // 7. EigenSignPredictionHead ("eigen_signs", dim = 4):
        //    Signs (+1 / -1) mapped to binary targets (1.0 / 0.0).
        const Eigen::VectorXd& raw_signs = eigen_signs.size() > 0 ? eigen_signs : obs.eigen_signs;
        if (raw_signs.size() > 0) {
            out.targets["eigen_signs"] = (raw_signs.array() >= 0.0).cast<double>().matrix();
            out.masks["eigen_signs"] = Eigen::VectorXd::Ones(raw_signs.size());
        }

        // 8. SpectralProfileHead ("spectral_profile", dim = 4):
        //    0/1 sign indicators + spectral balance target derived from eigen_signs.
        if (raw_signs.size() == 4) {
            // Spectral profile target in [0, 1].
            Eigen::VectorXd spec_t(4);
            spec_t(0) = raw_signs(0) >= 0.0 ? 1.0 : 0.0;
            spec_t(1) = raw_signs(1) >= 0.0 ? 1.0 : 0.0;
            spec_t(2) = raw_signs(2) >= 0.0 ? 1.0 : 0.0;
            spec_t(3) = 0.25 * ((raw_signs.array() >= 0.0).cast<double>().sum());
            out.targets["spectral_profile"] = spec_t;
            out.masks["spectral_profile"] = Eigen::VectorXd::Ones(4);
        }

        return out;
    }
};

}  // namespace forward_analysis
