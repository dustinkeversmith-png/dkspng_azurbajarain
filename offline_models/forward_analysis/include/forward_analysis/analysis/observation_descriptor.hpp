#pragma once

#include "forward_analysis/analysis/network_observations.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>

namespace forward_analysis {

// Fixed-width, shape-independent summary of NetworkObservations; the Aux1 head regresses this vector.
struct ObservationDescriptor {
    static constexpr std::size_t dim() noexcept { return 14; }

    // Components:
    //  0 dead-unit fraction of the first layer      1 dead-unit fraction of the last layer
    //  2 first-layer effective rank / live units    3 first layer lies on a ray (0/1)
    //  4 first layer is gated (0/1)                 5 fraction of samples gated closed
    //  6 gate coordinate found (0/1)                7 leading-axis driver count / 4
    //  8 strongest driver |correlation|             9 fraction of outputs keeping every sign
    // 10 fraction of outputs keeping the order     11 mean output amplitude ratio (clamped to [0, 2])
    // 12 mean output/target correlation            13 fraction of flat outputs
    static Eigen::VectorXd encode(const NetworkObservations& obs) {
        Eigen::VectorXd out = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dim()));
        if (!obs.layers.empty()) {
            const LayerObservation& first = obs.layers.front();
            const LayerObservation& last = obs.layers.back();
            out(0) = fraction(first.dead_units.size(), first.width);
            out(1) = fraction(last.dead_units.size(), last.width);
            out(2) = fraction(first.effective_rank, std::max<std::size_t>(1, first.live_count()));
            out(3) = first.lies_on_ray ? 1.0 : 0.0;
        }
        out(4) = obs.gating.present() ? 1.0 : 0.0;
        out(5) = fraction(obs.gating.closed_samples.size(), obs.shape.num_samples);
        out(6) = obs.gating.has_gate_coordinate ? 1.0 : 0.0;
        out(7) = std::min(1.0, static_cast<double>(obs.driver_coordinates(0).size()) / 4.0);
        out(8) = obs.drivers.empty() ? 0.0 : obs.drivers.front().strength;
        if (!obs.outputs.empty()) {
            const double inv = 1.0 / static_cast<double>(obs.outputs.size());
            for (const OutputObservation& o : obs.outputs) {
                out(9) += (o.sign_preserved ? 1.0 : 0.0) * inv;
                out(10) += (o.order_preserved ? 1.0 : 0.0) * inv;
                out(11) += std::min(2.0, std::max(0.0, o.amplitude_ratio)) * inv;
                out(12) += o.correlation * inv;
                out(13) += (o.flat ? 1.0 : 0.0) * inv;
            }
        }
        return out;
    }

private:
    static double fraction(std::size_t part, std::size_t whole) noexcept {
        return whole > 0 ? static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    }
};

}  // namespace forward_analysis
