#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/math/spectral_ops.hpp"
#include "forward_analysis/math/statistics_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace forward_analysis {

struct ObservationExtractor {
    static NetworkObservations extract(const ForwardTelemetryTensor& telemetry,
                                       const DistillerConfig& config = DistillerConfig{}) {
        NetworkObservations obs;
        obs.shape = telemetry.shape();
        for (const Eigen::MatrixXd& activations : telemetry.layer_activations) {
            obs.layers.push_back(observe_layer(activations, config));
        }
        if (telemetry.layer_activations.empty()) {
            return obs;
        }
        const Eigen::MatrixXd& first_layer = telemetry.layer_activations.front();
        obs.eigen_signs = eigen_signs(first_layer, config);
        obs.gating = observe_gating(telemetry, config);
        obs.drivers = observe_drivers(telemetry.inputs, first_layer, config);
        obs.target_related_coordinates = observe_target_related(telemetry, obs, config);
        for (Eigen::Index k = 0; k < telemetry.targets.cols(); ++k) {
            obs.outputs.push_back(observe_output(telemetry, obs.gating, k, config));
        }
        obs.outputs_move_in_lockstep = outputs_locked(telemetry.predictions);
        return obs;
    }

    static bool outputs_locked(const Eigen::MatrixXd& predictions) {
        if (predictions.cols() < 2) {
            return false;
        }
        for (Eigen::Index a = 0; a < predictions.cols(); ++a) {
            for (Eigen::Index b = a + 1; b < predictions.cols(); ++b) {
                if (std::abs(StatisticsOps::pearson(predictions.col(a), predictions.col(b))) < 0.999) {
                    return false;
                }
            }
        }
        return true;
    }

    static LayerObservation observe_layer(const Eigen::MatrixXd& activations, const DistillerConfig& config) {
        LayerObservation layer;
        layer.width = static_cast<std::size_t>(activations.cols());
        const Eigen::ArrayXXd magnitude = activations.array().abs();
        for (Eigen::Index u = 0; u < activations.cols(); ++u) {
            if (magnitude.col(u).maxCoeff() <= config.dead_activation_threshold) {
                layer.dead_units.push_back(static_cast<std::size_t>(u));
            } else if (magnitude.col(u).minCoeff() > config.dead_activation_threshold) {
                layer.always_active_units.push_back(static_cast<std::size_t>(u));
            }
        }

        const Eigen::VectorXd s =
            SpectralOps::analyze_activation_manifold(activations, layer.width).singular_values;
        if (s.size() > 0 && s(0) > 1e-12) {
            layer.effective_rank =
                static_cast<std::size_t>((s.array() >= config.spectral_rank_tolerance * s(0)).count());
        }

        const Eigen::VectorXd raw = SpectralOps::uncentered_singular_values(activations);
        const bool non_negative = activations.minCoeff() >= -config.dead_activation_threshold;
        layer.lies_on_ray = non_negative && raw.size() > 0 && raw(0) > 1e-12 &&
                            (raw.size() < 2 || raw(1) <= config.ray_tolerance * raw(0));
        return layer;
    }

    static Eigen::VectorXd eigen_signs(const Eigen::MatrixXd& first_layer, const DistillerConfig& config) {
        const Eigen::Index dim = static_cast<Eigen::Index>(config.eigen_sign_dim);
        const SpectralProfile profile =
            SpectralOps::analyze_activation_manifold(first_layer, std::max<std::size_t>(2, config.eigen_sign_dim));
        Eigen::VectorXd signs = Eigen::VectorXd::Ones(dim);
        const Eigen::Index copied = std::min(dim, profile.dominant_eigen_signs.size());
        signs.head(copied) = profile.dominant_eigen_signs.head(copied);
        return signs;
    }

    static GatingObservation observe_gating(const ForwardTelemetryTensor& telemetry, const DistillerConfig& config) {
        GatingObservation gating;
        const Eigen::MatrixXd& first_layer = telemetry.layer_activations.front();
        const Eigen::Index n = first_layer.rows();
        std::vector<bool> closed(static_cast<std::size_t>(n), false);
        for (Eigen::Index i = 0; i < n; ++i) {
            closed[static_cast<std::size_t>(i)] =
                first_layer.row(i).cwiseAbs().maxCoeff() <= config.dead_activation_threshold;
            (closed[static_cast<std::size_t>(i)] ? gating.closed_samples : gating.open_samples)
                .push_back(static_cast<std::size_t>(i));
        }
        if (gating.closed_samples.empty() || gating.open_samples.empty()) {
            gating.closed_samples.clear();
            gating.open_samples.clear();
            return gating;
        }

        double best_margin = -1.0;
        for (Eigen::Index c = 0; c < telemetry.inputs.cols(); ++c) {
            bool matches = true;
            double margin = std::numeric_limits<double>::infinity();
            for (Eigen::Index i = 0; i < n && matches; ++i) {
                const double x = telemetry.inputs(i, c);
                matches = (x < 0.0) == closed[static_cast<std::size_t>(i)];
                margin = std::min(margin, std::abs(x));
            }
            if (matches && margin > best_margin) {
                best_margin = margin;
                gating.has_gate_coordinate = true;
                gating.gate_coordinate = static_cast<std::size_t>(c);
            }
        }

        gating.deeper_layers_constant = telemetry.layer_activations.size() > 1;
        for (std::size_t l = 1; l < telemetry.layer_activations.size(); ++l) {
            const Eigen::MatrixXd& layer = telemetry.layer_activations[l];
            const Eigen::RowVectorXd reference = layer.row(static_cast<Eigen::Index>(gating.closed_samples.front()));
            for (std::size_t sample : gating.closed_samples) {
                if ((layer.row(static_cast<Eigen::Index>(sample)) - reference).cwiseAbs().maxCoeff() > 1e-9) {
                    gating.deeper_layers_constant = false;
                }
            }
        }
        return gating;
    }

    static std::vector<InputDriver> observe_drivers(const Eigen::MatrixXd& inputs,
                                                    const Eigen::MatrixXd& first_layer,
                                                    const DistillerConfig& config) {
        const Eigen::MatrixXd scores = SpectralOps::principal_scores(first_layer, 2);
        std::vector<InputDriver> primary;
        std::vector<InputDriver> secondary;
        for (Eigen::Index c = 0; c < inputs.cols(); ++c) {
            const double r0 = StatisticsOps::pearson(inputs.col(c), scores.col(0));
            const double r1 = StatisticsOps::pearson(inputs.col(c), scores.col(1));
            InputDriver driver;
            driver.coordinate = static_cast<std::size_t>(c);
            driver.strength = std::min(1.0, std::sqrt(r0 * r0 + r1 * r1));
            if (driver.strength < config.driver_correlation_threshold) {
                continue;
            }
            const bool leading = std::abs(r0) >= std::abs(r1);
            driver.axis = leading ? 0 : 1;
            driver.direction = (leading ? r0 : r1) >= 0.0 ? 1 : -1;
            (leading ? primary : secondary).push_back(driver);
        }
        const auto stronger = [](const InputDriver& l, const InputDriver& r) { return l.strength > r.strength; };
        std::stable_sort(primary.begin(), primary.end(), stronger);
        std::stable_sort(secondary.begin(), secondary.end(), stronger);
        primary.insert(primary.end(), secondary.begin(), secondary.end());
        return primary;
    }

    static std::vector<std::size_t> observe_target_related(const ForwardTelemetryTensor& telemetry,
                                                           const NetworkObservations& obs,
                                                           const DistillerConfig& config) {
        std::vector<std::pair<double, std::size_t>> ranked;
        for (Eigen::Index c = 0; c < telemetry.inputs.cols(); ++c) {
            const std::size_t coordinate = static_cast<std::size_t>(c);
            bool known = obs.gating.has_gate_coordinate && obs.gating.gate_coordinate == coordinate;
            for (const InputDriver& d : obs.drivers) {
                known = known || d.coordinate == coordinate;
            }
            if (known) {
                continue;
            }
            double best = 0.0;
            for (Eigen::Index k = 0; k < telemetry.targets.cols(); ++k) {
                best = std::max(best, std::abs(StatisticsOps::pearson(telemetry.inputs.col(c), telemetry.targets.col(k))));
            }
            if (best >= config.driver_correlation_threshold) {
                ranked.push_back({best, coordinate});
            }
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& l, const auto& r) { return l.first > r.first; });
        std::vector<std::size_t> out;
        for (const auto& entry : ranked) {
            out.push_back(entry.second);
        }
        return out;
    }

    static OutputObservation observe_output(const ForwardTelemetryTensor& telemetry,
                                            const GatingObservation& gating,
                                            Eigen::Index k,
                                            const DistillerConfig& config) {
        const Eigen::VectorXd t = telemetry.targets.col(k);
        const Eigen::VectorXd p = telemetry.predictions.col(k);
        OutputObservation out;
        out.index = static_cast<std::size_t>(k);

        out.sign_preserved = true;
        for (Eigen::Index i = 0; i < t.size(); ++i) {
            if (std::abs(t(i)) > 1e-12 && !(t(i) * p(i) > 0.0)) {
                out.sign_preserved = false;
            }
        }
        out.order_preserved = StatisticsOps::same_order(t, p);
        const double target_mass = t.cwiseAbs().sum();
        out.amplitude_ratio = target_mass > 1e-12 ? p.cwiseAbs().sum() / target_mass : 1.0;
        out.correlation = StatisticsOps::pearson(t, p);
        out.flat = StatisticsOps::standard_deviation(p) <=
                   config.flat_output_ratio * StatisticsOps::standard_deviation(t);

        out.open_side_correlation = out.correlation;
        if (gating.closed_samples.size() >= 2) {
            const double reference = p(static_cast<Eigen::Index>(gating.closed_samples.front()));
            out.constant_on_closed_samples = true;
            for (std::size_t sample : gating.closed_samples) {
                if (std::abs(p(static_cast<Eigen::Index>(sample)) - reference) > 1e-9) {
                    out.constant_on_closed_samples = false;
                }
            }
        }
        if (gating.present()) {
            Eigen::VectorXd open_t(static_cast<Eigen::Index>(gating.open_samples.size()));
            Eigen::VectorXd open_p(open_t.size());
            for (std::size_t i = 0; i < gating.open_samples.size(); ++i) {
                open_t(static_cast<Eigen::Index>(i)) = t(static_cast<Eigen::Index>(gating.open_samples[i]));
                open_p(static_cast<Eigen::Index>(i)) = p(static_cast<Eigen::Index>(gating.open_samples[i]));
            }
            out.open_side_correlation = StatisticsOps::pearson(open_t, open_p);
            const Eigen::VectorXd error = (p - t).cwiseAbs();
            double closed_error = 0.0;
            for (std::size_t sample : gating.closed_samples) {
                closed_error += error(static_cast<Eigen::Index>(sample));
            }
            out.closed_error_share = error.sum() > 1e-12 ? closed_error / error.sum() : 0.0;
        }
        return out;
    }
};

}  // namespace forward_analysis
