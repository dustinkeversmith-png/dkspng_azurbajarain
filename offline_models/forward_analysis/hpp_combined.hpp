// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\forward_analysis.hpp ===
#pragma once

#include "forward_analysis/analysis/gating_observation.hpp"
#include "forward_analysis/analysis/input_driver.hpp"
#include "forward_analysis/analysis/layer_observation.hpp"
#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/analysis/observation_extractor.hpp"
#include "forward_analysis/analysis/observation_json_writer.hpp"
#include "forward_analysis/analysis/output_observation.hpp"
#include "forward_analysis/bridge/bridge_checkpoint.hpp"
#include "forward_analysis/bridge/distillation_result.hpp"
#include "forward_analysis/bridge/dual_objective_distillation_bridge.hpp"
#if FORWARD_ANALYSIS_WITH_OPENCL
#include "forward_analysis/compute/decoder_kernel_source.hpp"
#include "forward_analysis/compute/device_buffer.hpp"
#include "forward_analysis/compute/device_gemm.hpp"
#include "forward_analysis/compute/device_parameter.hpp"
#include "forward_analysis/compute/device_parameter_arena.hpp"
#include "forward_analysis/compute/opencl_runtime.hpp"
#include "forward_analysis/decoder/decoder_device_mirror.hpp"
#endif
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/config/narration_style.hpp"
#include "forward_analysis/config/slot_binding.hpp"
#include "forward_analysis/dataset/dataset_shape_summary.hpp"
#include "forward_analysis/dataset/ground_truth_targets.hpp"
#include "forward_analysis/dataset/json_tensor_reader.hpp"
#include "forward_analysis/dataset/training_dataset_importer.hpp"
#include "forward_analysis/dataset/training_sample.hpp"
#include "forward_analysis/decoder/autoregressive_causal_decoder.hpp"
#include "forward_analysis/decoder/batch_statement_delexicalizer.hpp"
#include "forward_analysis/decoder/decoder_device_slot.hpp"
#include "forward_analysis/decoder/decoder_parameter_gradients.hpp"
#include "forward_analysis/decoder/decoder_sequence_cache.hpp"
#include "forward_analysis/decoder/decoder_tensor_refs.hpp"
#include "forward_analysis/decoder/field_delexicalizer.hpp"
#include "forward_analysis/decoder/observation_delexicalizer.hpp"
#include "forward_analysis/decoder/observation_tokenizer.hpp"
#include "forward_analysis/decoder/shape_slot_lexicon.hpp"
#include "forward_analysis/description/batch_statement_checker.hpp"
#include "forward_analysis/description/batch_statistics_narrator.hpp"
#include "forward_analysis/description/measured_facts_narration.hpp"
#include "forward_analysis/description/number_words.hpp"
#include "forward_analysis/description/observation_consistency.hpp"
#include "forward_analysis/description/observation_narrator.hpp"
#include "forward_analysis/description/observation_similarity.hpp"
#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/encoder/channel_feature_extractor.hpp"
#include "forward_analysis/encoder/channel_role.hpp"
#include "forward_analysis/encoder/channel_sequence_formatter.hpp"
#include "forward_analysis/encoder/linear_input_adapter.hpp"
#include "forward_analysis/encoder/perceiver_forward_cache.hpp"
#include "forward_analysis/encoder/perceiver_io_encoder.hpp"
#include "forward_analysis/encoder/sinusoidal_positional_encoding.hpp"
#include "forward_analysis/heads/auxiliary_alignment_heads.hpp"
#include "forward_analysis/heads/auxiliary_evaluation.hpp"
#include "forward_analysis/heads/eigen_sign_head.hpp"
#include "forward_analysis/heads/observation_regression_head.hpp"
#include "forward_analysis/math/adam_moment.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"
#include "forward_analysis/math/spectral_ops.hpp"
#include "forward_analysis/math/spectral_profile.hpp"
#include "forward_analysis/math/statistics_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"
#include "forward_analysis/telemetry/telemetry_builder.hpp"
#include "forward_analysis/telemetry/telemetry_shape.hpp"
#include "forward_analysis/text/csv_field_formatter.hpp"
#include "forward_analysis/text/english_list_formatter.hpp"
#include "forward_analysis/text/json_text_formatter.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"
#include "forward_analysis/training/epoch_record.hpp"
#include "forward_analysis/training/finetune_outcome.hpp"
#include "forward_analysis/training/loss_breakdown.hpp"
#include "forward_analysis/training/split_metric_totals.hpp"
#include "forward_analysis/training/training_run_artifact_writer.hpp"
#include "forward_analysis/training/training_run_settings.hpp"

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\gating_observation.hpp ===
#pragma once

#include <cstddef>
#include <vector>

namespace forward_analysis {

struct GatingObservation {
    // Samples on which every first-layer unit outputs zero (empty unless some, but not all, samples are closed).
    std::vector<std::size_t> closed_samples;
    // Samples on which at least one first-layer unit is non-zero.
    std::vector<std::size_t> open_samples;
    // True when a single input coordinate is negative on exactly the closed samples.
    bool has_gate_coordinate = false;
    // That input coordinate (valid only when has_gate_coordinate).
    std::size_t gate_coordinate = 0;
    // True when every deeper layer shows one identical activation row across all closed samples.
    bool deeper_layers_constant = false;

    bool present() const noexcept {
        return !closed_samples.empty();
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\input_driver.hpp ===
#pragma once

#include <cstddef>

namespace forward_analysis {

struct InputDriver {
    // Input coordinate index (x<coordinate>).
    std::size_t coordinate = 0;
    // The first-layer principal axis it correlates with more strongly: 0 = leading axis, 1 = secondary axis.
    std::size_t axis = 0;
    // +1 when that axis score rises with the coordinate, -1 when it falls.
    int direction = 1;
    // Multiple correlation with the two leading axis scores, sqrt(r1^2 + r2^2), in [0, 1].
    double strength = 0.0;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\layer_observation.hpp ===
#pragma once

#include <cstddef>
#include <vector>

namespace forward_analysis {

struct LayerObservation {
    // Number of units in the layer.
    std::size_t width = 0;
    // Units that output zero on every sample in the batch.
    std::vector<std::size_t> dead_units;
    // Units that are non-zero on every sample in the batch.
    std::vector<std::size_t> always_active_units;
    // Number of covariance singular values above spectral_rank_tolerance of the largest (directions the batch spreads along).
    std::size_t effective_rank = 0;
    // Number of singular values of the centered activation matrix above numerical_rank_tolerance (absolute).
    std::size_t numerical_rank = 0;
    // Samples on which no unit of this layer is positive (every activation <= 0).
    std::size_t nonpositive_samples = 0;
    // True when every activation row is a non-negative multiple of one fixed vector (rank-1 through the origin).
    bool lies_on_ray = false;

    std::size_t live_count() const noexcept {
        return width - dead_units.size();
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\network_observations.hpp ===
#pragma once

#include "forward_analysis/analysis/gating_observation.hpp"
#include "forward_analysis/analysis/input_driver.hpp"
#include "forward_analysis/analysis/layer_observation.hpp"
#include "forward_analysis/analysis/output_observation.hpp"
#include "forward_analysis/telemetry/telemetry_shape.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace forward_analysis {

struct NetworkObservations {
    // Batch size, input width, hidden widths and output width of the observed telemetry.
    TelemetryShape shape;
    // One entry per hidden layer, in forward order.
    std::vector<LayerObservation> layers;
    // Half-space gating of the first layer, if any samples are fully closed.
    GatingObservation gating;
    // Input coordinates strongly correlated with the first layer's leading or secondary principal axis, strongest first.
    std::vector<InputDriver> drivers;
    // Input coordinates that are neither the gate nor a driver but correlate with some target column at or above
    // the driver threshold, strongest first (inputs the targets depend on, whether or not the network uses them).
    std::vector<std::size_t> target_related_coordinates;
    // One entry per network output.
    std::vector<OutputObservation> outputs;
    // True when there are several outputs and every pair of prediction columns is perfectly (+/-) correlated.
    bool outputs_move_in_lockstep = false;
    // Signs (+1 / -1, zero counted as +1) of the eigenvalues, in descending order, of the sample covariance of the first
    // eigen_sign_dim input coordinates with its diagonal set to zero (inputs centered, divisor max(1, N - 1)).
    Eigen::VectorXd eigen_signs;
    // The two input coordinates with the largest max_k |Pearson(input, final hidden unit k)|, strongest first.
    std::vector<std::size_t> strongest_inputs;
    // The two input coordinates with the smallest such association, weakest first.
    std::vector<std::size_t> weakest_inputs;
    // Largest input / final-hidden-unit |Pearson| over all pairs (0 when the final layer does not vary).
    double strongest_association = 0.0;
    // Outputs whose range_ratio is below range_ratio_low.
    std::size_t outputs_with_smaller_range = 0;
    // Outputs whose range_ratio is above range_ratio_high.
    std::size_t outputs_with_larger_range = 0;
    // Outputs whose range_ratio lies in [range_ratio_low, range_ratio_high].
    std::size_t outputs_with_comparable_range = 0;
    // Sum of sign_agreements over all outputs.
    std::size_t sign_agreements = 0;
    // Number of prediction entries (samples x outputs).
    std::size_t sign_entries = 0;
    // True when every hidden layer's numerical rank is at most narrow_rank_limit.
    bool narrow_representation = false;

    std::vector<std::size_t> driver_coordinates(std::size_t axis) const {
        std::vector<std::size_t> out;
        for (const InputDriver& d : drivers) {
            if (d.axis == axis) {
                out.push_back(d.coordinate);
            }
        }
        return out;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\observation_descriptor.hpp ===
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

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\observation_extractor.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/math/spectral_ops.hpp"
#include "forward_analysis/math/statistics_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

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
        obs.eigen_signs = eigen_signs(telemetry.inputs, config);
        obs.gating = observe_gating(telemetry, config);
        obs.drivers = observe_drivers(telemetry.inputs, first_layer, config);
        obs.target_related_coordinates = observe_target_related(telemetry, obs, config);
        for (Eigen::Index k = 0; k < telemetry.targets.cols(); ++k) {
            obs.outputs.push_back(observe_output(telemetry, obs.gating, k, config));
        }
        obs.outputs_move_in_lockstep = outputs_locked(telemetry.predictions);
        observe_input_association(telemetry, obs);
        for (const OutputObservation& o : obs.outputs) {
            obs.outputs_with_smaller_range += o.range_ratio < config.range_ratio_low ? 1 : 0;
            obs.outputs_with_larger_range += o.range_ratio > config.range_ratio_high ? 1 : 0;
            obs.outputs_with_comparable_range +=
                o.range_ratio >= config.range_ratio_low && o.range_ratio <= config.range_ratio_high ? 1 : 0;
            obs.sign_agreements += o.sign_agreements;
        }
        obs.sign_entries = static_cast<std::size_t>(telemetry.targets.size());
        obs.narrow_representation = true;
        for (const LayerObservation& layer : obs.layers) {
            obs.narrow_representation = obs.narrow_representation && layer.numerical_rank <= config.narrow_rank_limit;
        }
        return obs;
    }

    // Ranks the input coordinates by max_k |Pearson(input, final hidden unit k)| (stable, ties by index).
    static void observe_input_association(const ForwardTelemetryTensor& telemetry, NetworkObservations& obs) {
        const Eigen::MatrixXd& last = telemetry.layer_activations.back();
        std::vector<std::pair<double, std::size_t>> scores;
        for (Eigen::Index c = 0; c < telemetry.inputs.cols(); ++c) {
            double best = 0.0;
            for (Eigen::Index k = 0; k < last.cols(); ++k) {
                best = std::max(best, std::abs(StatisticsOps::pearson(telemetry.inputs.col(c), last.col(k))));
            }
            scores.push_back({best, static_cast<std::size_t>(c)});
        }
        std::vector<std::pair<double, std::size_t>> descending = scores;
        std::stable_sort(descending.begin(), descending.end(), [](const auto& l, const auto& r) { return l.first > r.first; });
        std::vector<std::pair<double, std::size_t>> ascending = scores;
        std::stable_sort(ascending.begin(), ascending.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
        for (std::size_t k = 0; k < std::min<std::size_t>(2, scores.size()); ++k) {
            obs.strongest_inputs.push_back(descending[k].second);
            obs.weakest_inputs.push_back(ascending[k].second);
        }
        obs.strongest_association = descending.empty() ? 0.0 : descending.front().first;
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
        const Eigen::MatrixXd centered = activations.rowwise() - activations.colwise().mean();
        layer.numerical_rank = static_cast<std::size_t>(
            (Eigen::BDCSVD<Eigen::MatrixXd>(centered).singularValues().array() > config.numerical_rank_tolerance).count());
        for (Eigen::Index i = 0; i < activations.rows(); ++i) {
            layer.nonpositive_samples += activations.row(i).maxCoeff() <= 0.0 ? 1 : 0;
        }

        const Eigen::VectorXd raw = SpectralOps::uncentered_singular_values(activations);
        const bool non_negative = activations.minCoeff() >= -config.dead_activation_threshold;
        layer.lies_on_ray = non_negative && raw.size() > 0 && raw(0) > 1e-12 &&
                            (raw.size() < 2 || raw(1) <= config.ray_tolerance * raw(0));
        return layer;
    }

    // Signs of the descending eigenvalues of the zero-diagonal covariance of the first eigen_sign_dim inputs
    // (zero counts as +1); positions beyond the input width stay +1.
    static Eigen::VectorXd eigen_signs(const Eigen::MatrixXd& inputs, const DistillerConfig& config) {
        const Eigen::Index dim = static_cast<Eigen::Index>(config.eigen_sign_dim);
        Eigen::VectorXd signs = Eigen::VectorXd::Ones(dim);
        const Eigen::Index used = std::min(dim, inputs.cols());
        if (used == 0 || inputs.rows() == 0) {
            return signs;
        }
        const Eigen::MatrixXd centered = inputs.leftCols(used).rowwise() - inputs.leftCols(used).colwise().mean();
        Eigen::MatrixXd covariance =
            centered.transpose() * centered / std::max(1.0, static_cast<double>(inputs.rows() - 1));
        covariance.diagonal().setZero();
        const Eigen::VectorXd descending = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(covariance).eigenvalues().reverse();
        signs.head(used) = descending.unaryExpr([](double v) { return v >= 0.0 ? 1.0 : -1.0; });
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
            out.sign_agreements += ((t(i) > 0.0) - (t(i) < 0.0)) == ((p(i) > 0.0) - (p(i) < 0.0)) ? 1 : 0;
        }
        const double target_range = t.size() > 0 ? t.maxCoeff() - t.minCoeff() : 0.0;
        out.range_ratio = target_range > 1e-12 ? (p.maxCoeff() - p.minCoeff()) / target_range : 1.0;
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

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\observation_json_writer.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace forward_analysis {

// Serializes the measured facts so they can be inspected or handed to a downstream LLM.
struct ObservationJsonWriter {
    static nlohmann::json to_json(const NetworkObservations& obs) {
        nlohmann::json out;
        out["shape"] = {{"num_samples", obs.shape.num_samples},
                        {"input_dim", obs.shape.input_dim},
                        {"hidden_widths", obs.shape.hidden_widths},
                        {"output_dim", obs.shape.output_dim}};
        out["layers"] = nlohmann::json::array();
        for (const LayerObservation& layer : obs.layers) {
            out["layers"].push_back({{"width", layer.width},
                                     {"dead_units", layer.dead_units},
                                     {"always_active_units", layer.always_active_units},
                                     {"effective_rank", layer.effective_rank},
                                     {"numerical_rank", layer.numerical_rank},
                                     {"nonpositive_samples", layer.nonpositive_samples},
                                     {"lies_on_ray", layer.lies_on_ray}});
        }
        nlohmann::json gating = {{"closed_samples", obs.gating.closed_samples},
                                 {"open_samples", obs.gating.open_samples},
                                 {"deeper_layers_constant", obs.gating.deeper_layers_constant}};
        gating["gate_coordinate"] =
            obs.gating.has_gate_coordinate ? nlohmann::json(obs.gating.gate_coordinate) : nlohmann::json(nullptr);
        out["gating"] = gating;
        out["drivers"] = nlohmann::json::array();
        for (const InputDriver& d : obs.drivers) {
            out["drivers"].push_back({{"coordinate", d.coordinate},
                                      {"axis", d.axis == 0 ? "leading" : "secondary"},
                                      {"direction", d.direction > 0 ? "rises" : "falls"},
                                      {"strength", d.strength}});
        }
        out["target_related_coordinates"] = obs.target_related_coordinates;
        out["outputs"] = nlohmann::json::array();
        for (const OutputObservation& o : obs.outputs) {
            out["outputs"].push_back({{"index", o.index},
                                      {"sign_preserved", o.sign_preserved},
                                      {"order_preserved", o.order_preserved},
                                      {"amplitude_ratio", o.amplitude_ratio},
                                      {"correlation", o.correlation},
                                      {"flat", o.flat},
                                      {"constant_on_closed_samples", o.constant_on_closed_samples},
                                      {"open_side_correlation", o.open_side_correlation},
                                      {"closed_error_share", o.closed_error_share},
                                      {"range_ratio", o.range_ratio},
                                      {"sign_agreements", o.sign_agreements}});
        }
        out["outputs_move_in_lockstep"] = obs.outputs_move_in_lockstep;
        out["eigen_signs"] = std::vector<double>(obs.eigen_signs.data(), obs.eigen_signs.data() + obs.eigen_signs.size());
        out["strongest_inputs"] = obs.strongest_inputs;
        out["weakest_inputs"] = obs.weakest_inputs;
        out["strongest_association"] = obs.strongest_association;
        out["outputs_with_smaller_range"] = obs.outputs_with_smaller_range;
        out["outputs_with_larger_range"] = obs.outputs_with_larger_range;
        out["outputs_with_comparable_range"] = obs.outputs_with_comparable_range;
        out["sign_agreements"] = obs.sign_agreements;
        out["sign_entries"] = obs.sign_entries;
        out["narrow_representation"] = obs.narrow_representation;
        return out;
    }

    static std::string write(const NetworkObservations& obs, int indent = 2) {
        return to_json(obs).dump(indent);
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\analysis\output_observation.hpp ===
#pragma once

#include <cstddef>

namespace forward_analysis {

struct OutputObservation {
    // Output index k.
    std::size_t index = 0;
    // True when the prediction has the same sign as the target on every sample.
    bool sign_preserved = false;
    // True when the predictions order the samples exactly as the targets do.
    bool order_preserved = false;
    // sum |prediction| / sum |target| (below 1 means damped, above 1 means amplified).
    double amplitude_ratio = 1.0;
    // Pearson correlation between prediction and target across samples.
    double correlation = 0.0;
    // True when the prediction's spread is at most flat_output_ratio of the target's spread.
    bool flat = false;
    // True when the prediction is one identical value on every gated (closed) sample.
    bool constant_on_closed_samples = false;
    // Prediction/target correlation over the open (non-gated) samples only; equals correlation when nothing is gated.
    double open_side_correlation = 0.0;
    // Share of the absolute error that falls on closed samples (0 when nothing is gated).
    double closed_error_share = 0.0;
    // (max - min) of the prediction over (max - min) of the target; 1 when the target has no range.
    double range_ratio = 1.0;
    // Samples on which sign(prediction) equals sign(target), with 0 its own sign.
    std::size_t sign_agreements = 0;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\bridge\bridge_checkpoint.hpp ===
#pragma once

#include "forward_analysis/decoder/autoregressive_causal_decoder.hpp"
#include "forward_analysis/encoder/channel_sequence_formatter.hpp"
#include "forward_analysis/encoder/perceiver_io_encoder.hpp"
#include "forward_analysis/heads/auxiliary_alignment_heads.hpp"

namespace forward_analysis {

// Full copy of every trainable component of the bridge (weights and optimizer state) at one epoch.
struct BridgeCheckpoint {
    // Channel-feature adapter.
    ChannelSequenceFormatter formatter;
    // Perceiver IO encoder producing Z.
    PerceiverIOEncoder perceiver;
    // Aux1 / Aux2 heads.
    AuxiliaryAlignmentHeads aux_heads;
    // Cross-attending causal decoder with its vocabulary.
    AutoregressiveCausalDecoder decoder;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\bridge\distillation_result.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/description/observation_text.hpp"

#include <Eigen/Core>

#include <array>
#include <cstddef>

namespace forward_analysis {

struct DistillationResult {
    // Number of channel tokens the encoder read for this telemetry.
    std::size_t channel_count = 0;
    // Encoder latent bottleneck Z (num_latents x d_model).
    Eigen::MatrixXd latent_embeddings;
    // Aux1 head's predicted observation descriptor.
    Eigen::VectorXd predicted_observation_descriptor;
    // Aux2 head's predicted eigenvalue signs.
    Eigen::VectorXd predicted_eigen_signs;
    // Facts measured directly from the telemetry.
    NetworkObservations measured;
    // The four fields exactly as the decoder generated them, slot tokens unfilled, e.g. "<COUNT:DEAD_L1|W> of the ...".
    ObservationText decoder_slot_observations;
    // Per field: true when every slot token was fillable from this network's measurements.
    std::array<bool, ObservationText::kFieldCount> decoder_slots_filled{};
    // The four decoder fields after filling slots from the measured telemetry (the slot text where a slot was unfillable).
    ObservationText decoder_observations;
    // The measured-facts narration (config.narration_style) written without the decoder.
    ObservationText narrated_observations;
    // Final output: each decoder field that passed consistency, otherwise the measured-fact narration.
    ObservationText observations;
    // Per field: true when the decoder's text passed the consistency check against the measured facts.
    std::array<bool, ObservationText::kFieldCount> decoder_field_consistent{};
    // Per field: number of tokens the decoder emitted (BOS and EOS included).
    std::array<std::size_t, ObservationText::kFieldCount> decoder_token_counts{};
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\bridge\dual_objective_distillation_bridge.hpp ===
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
#include "forward_analysis/description/observation_consistency.hpp"
#include "forward_analysis/description/measured_facts_narration.hpp"
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

// Channel tokens -> linear adapter -> Perceiver IO latents Z, which feed both the auxiliary heads
// (Aux1 observation descriptor, Aux2 eigen signs) and the field-conditioned causal decoder. The decoder
// cross-attends over Z, is conditioned on the Aux1-predicted descriptor, and writes slot-token text whose
// shape and index slots are filled from the network's measured telemetry.
// Step 1 trains the representation on the auxiliary losses with the decoder frozen; step 2 trains
// everything on CE + 0.3 Aux1 + 0.2 Aux2 with early stopping; step 3 distills new telemetry into observation text.
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
        return FieldDelexicalizer::delexicalize(sample.ground_truth_targets.observations.field(field), sample.measured, binding);
    }

    void prepare_vocabulary(const std::vector<TrainingSample>& dataset) {
        std::vector<std::vector<std::string>> fields;
        for (const TrainingSample& sample : dataset) {
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
        std::vector<LossBreakdown> history;
        for (std::size_t epoch = 0; epoch < epochs && !dataset.empty(); ++epoch) {
            history.push_back(run_epoch(dataset, learning_rate, false));
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
        FinetuneOutcome outcome;
        outcome.max_epochs = max_epochs;
        if (dataset.empty()) {
            return outcome;
        }
        const std::vector<TrainingSample>& validation = held_out.empty() ? dataset : held_out;
        double best_ce = std::numeric_limits<double>::infinity();
        double best_total = std::numeric_limits<double>::infinity();
        std::size_t epochs_without_ce_decrease = 0;
        BridgeCheckpoint best = checkpoint();
        for (std::size_t epoch = 1; epoch <= max_epochs; ++epoch) {
            EpochRecord record;
            record.epoch = epoch;
            record.train = run_epoch(dataset, learning_rate, true);
            record.eval = evaluate_dataset(validation);
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

    // Losses of one sample without updating anything (decoder in eval mode, conditioned on the Aux1 prediction).
    LossBreakdown evaluate_sample(const TrainingSample& sample) {
        const Eigen::MatrixXd latents = encode(sample.telemetry);
        const AuxiliaryEvaluation aux = aux_heads_.evaluate(
            latents, ObservationDescriptor::encode(sample.measured), sample.ground_truth_targets.eigen_signs);
        double ce = 0.0;
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            ce += decoder_.compute_loss(latents, aux.predicted_observation_descriptor, f,
                                        target_lexemes(sample, f, config_.slot_binding));
        }
        ce /= static_cast<double>(ObservationText::kFieldCount);
        return LossBreakdown::combine(ce, aux.aux1_observation_loss, aux.aux2_eigen_sign_loss, config_);
    }

    // Mean losses over a set of samples without updating anything (e.g. held-out records after each epoch).
    LossBreakdown evaluate_dataset(const std::vector<TrainingSample>& dataset) {
        double ce = 0.0;
        double aux1 = 0.0;
        double aux2 = 0.0;
        for (const TrainingSample& sample : dataset) {
            const LossBreakdown loss = evaluate_sample(sample);
            ce += loss.token_cross_entropy;
            aux1 += loss.aux1_observation_loss;
            aux2 += loss.aux2_eigen_sign_loss;
        }
        const double inv_n = dataset.empty() ? 0.0 : 1.0 / static_cast<double>(dataset.size());
        return LossBreakdown::combine(ce * inv_n, aux1 * inv_n, aux2 * inv_n, config_);
    }

    Eigen::MatrixXd encode(const ForwardTelemetryTensor& telemetry) {
        return perceiver_.encode(formatter_.format_sequence(telemetry));
    }

    DistillationResult distill_telemetry(const ForwardTelemetryTensor& telemetry) {
        DistillationResult result;
        const Eigen::MatrixXd features = formatter_.channel_features(telemetry);
        result.channel_count = static_cast<std::size_t>(features.rows());
        result.latent_embeddings = perceiver_.encode(formatter_.format_sequence(features));
        result.measured = ObservationExtractor::extract(telemetry, config_);

        const AuxiliaryEvaluation aux = aux_heads_.evaluate(result.latent_embeddings);
        result.predicted_observation_descriptor = aux.predicted_observation_descriptor;
        result.predicted_eigen_signs = aux.predicted_eigen_signs;

        const ShapeSlotLexicon slots = ShapeSlotLexicon::from(result.measured);
        result.narrated_observations = MeasuredFactsNarration::describe(result.measured, config_);
        const ObservationText& narrated = result.narrated_observations;
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            const std::vector<int> ids = decoder_.generate_field_tokens(
                result.latent_embeddings, result.predicted_observation_descriptor, f, config_.max_field_tokens);
            const std::vector<std::string> lexemes = decoder_.tokenizer().decode_lexemes(ids);
            result.decoder_token_counts[f] = ids.size();
            result.decoder_slot_observations.field(f) = LexemeSplitter::join(lexemes);
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

    LossBreakdown train_step(const TrainingSample& sample, double learning_rate, bool train_decoder) {
        const Eigen::MatrixXd features = formatter_.channel_features(sample.telemetry);
        const PerceiverForwardCache cache = perceiver_.forward_with_cache(formatter_.format_sequence(features));
        const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(config_.num_latents),
                                                           static_cast<Eigen::Index>(config_.d_model));

        double ce = 0.0;
        Eigen::MatrixXd d_latents = zero;
        if (train_decoder) {
            const Eigen::VectorXd predicted_descriptor =
                aux_heads_.evaluate(cache.latents_z).predicted_observation_descriptor;
            Eigen::MatrixXd d_decoder = zero;
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                ce += decoder_.compute_loss_and_step(cache.latents_z, predicted_descriptor, f,
                                                     target_lexemes(sample, f, config_.slot_binding), d_decoder);
            }
            ce /= static_cast<double>(ObservationText::kFieldCount);
            d_latents += d_decoder / static_cast<double>(ObservationText::kFieldCount);
        }

        const AuxiliaryEvaluation aux = aux_heads_.compute_losses_and_step(
            cache.latents_z, ObservationDescriptor::encode(sample.measured), sample.ground_truth_targets.eigen_signs,
            learning_rate, d_latents);

        const Eigen::MatrixXd d_sequence = perceiver_.backward_and_update(cache, d_latents, learning_rate);
        formatter_.update_adapter(features, d_sequence, learning_rate);
        return LossBreakdown::combine(ce, aux.aux1_observation_loss, aux.aux2_eigen_sign_loss, config_);
    }

    LossBreakdown run_epoch(const std::vector<TrainingSample>& dataset, double learning_rate, bool train_decoder) {
        decoder_.set_frozen(!train_decoder);
        double ce = 0.0;
        double aux1 = 0.0;
        double aux2 = 0.0;
        for (const TrainingSample& sample : dataset) {
            const LossBreakdown step = train_step(sample, learning_rate, train_decoder);
            ce += step.token_cross_entropy;
            aux1 += step.aux1_observation_loss;
            aux2 += step.aux2_eigen_sign_loss;
        }
        const double inv_n = 1.0 / static_cast<double>(dataset.size());
        return LossBreakdown::combine(ce * inv_n, aux1 * inv_n, aux2 * inv_n, config_);
    }

    // Shared model dimensions, decoder budget and learning rate, loss weights and measurement thresholds.
    DistillerConfig config_;
    // Turns telemetry of any shape into the channel-token sequence (C x d_model).
    ChannelSequenceFormatter formatter_;
    // Compresses the channel sequence into the latent bottleneck Z.
    PerceiverIOEncoder perceiver_;
    // Branch A: observation-descriptor and eigen-sign heads on Z.
    AuxiliaryAlignmentHeads aux_heads_;
    // Branch B: generates the four observation fields from Z and the Aux1 prediction.
    AutoregressiveCausalDecoder decoder_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\compute\decoder_kernel_source.hpp ===
#pragma once

#include <cstddef>

namespace forward_analysis {

// OpenCL C source of the element-wise and row-wise decoder kernels, compiled at runtime by OpenClRuntime.
// Matrices are column-major doubles (element (r, c) of an R-row matrix at c * R + r), the layout Eigen uses,
// so host matrices upload without reordering. Each kernel reproduces the arithmetic order of the matching
// Eigen expression in AutoregressiveCausalDecoder / MatrixOps / AdamMoment.
struct DecoderKernelSource {
    // Work-items per work-group in the reduction kernels (layer norm, softmax, gemm_reduce, column gradients); a
    // power of two, passed as -DROW_GROUP.
    static constexpr std::size_t kRowGroup = 128;

    static const char* text() noexcept {
        return R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

// Row kernels run one work-group of ROW_GROUP work-items per row (ROW_GROUP is a power of two passed at build
// time); each work-item strides over the row's columns and the partial results meet in a local-memory tree.
double group_sum(__local double* scratch, const double value) {
    const int l = get_local_id(0);
    scratch[l] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int span = ROW_GROUP / 2; span > 0; span >>= 1) {
        if (l < span) {
            scratch[l] += scratch[l + span];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const double total = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return total;
}

double group_max(__local double* scratch, const double value) {
    const int l = get_local_id(0);
    scratch[l] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int span = ROW_GROUP / 2; span > 0; span >>= 1) {
        if (l < span) {
            scratch[l] = fmax(scratch[l], scratch[l + span]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const double top = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return top;
}

// C (M x N) = alpha op(A) op(B) + beta C, column-major with leading dimensions lda / ldb / ldc; op transposes when
// the flag is set; with beta = 0, C is not read. One work-item per element of C, accumulating over K in order.
__kernel void gemm_direct(const int M, const int N, const int K, const int trans_a, const int trans_b, const double alpha,
                          __global const double* A, const int lda, __global const double* B, const int ldb,
                          const double beta, __global double* C, const int ldc) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= M || j >= N) {
        return;
    }
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
        const double a = trans_a != 0 ? A[i * lda + k] : A[k * lda + i];
        const double b = trans_b != 0 ? B[k * ldb + j] : B[j * ldb + k];
        acc += a * b;
    }
    const int index = j * ldc + i;
    C[index] = beta == 0.0 ? alpha * acc : alpha * acc + beta * C[index];
}

// The same product for outputs with few entries: one ROW_GROUP work-group per element of C, its work-items striding
// over K and meeting in a local-memory tree.
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void gemm_reduce(const int M, const int N, const int K, const int trans_a, const int trans_b, const double alpha,
                 __global const double* A, const int lda, __global const double* B, const int ldb, const double beta,
                 __global double* C, const int ldc) {
    __local double scratch[ROW_GROUP];
    const int element = get_group_id(0);
    const int i = element % M;
    const int j = element / M;
    const int l = get_local_id(0);
    double part = 0.0;
    for (int k = l; k < K; k += ROW_GROUP) {
        const double a = trans_a != 0 ? A[i * lda + k] : A[k * lda + i];
        const double b = trans_b != 0 ? B[k * ldb + j] : B[j * ldb + k];
        part += a * b;
    }
    const double acc = group_sum(scratch, part);
    if (l == 0) {
        const int index = j * ldc + i;
        C[index] = beta == 0.0 ? alpha * acc : alpha * acc + beta * C[index];
    }
}

)CLC"
               R"CLC(// u (S x d): u[t] = E[prev_t] * m1[t] + w2 * (E2[prev2_t] * m2[t]) + F[field] + P[t]
__kernel void build_step_inputs(const int S, const int d, const int V, const int nF, const int nP, const int field,
                                const double w2, __global const int* prev, __global const int* prev2,
                                __global const double* E, __global const double* E2, __global const double* F,
                                __global const double* P, __global const double* m1, __global const double* m2,
                                __global double* u) {
    const int t = get_global_id(0);
    const int j = get_global_id(1);
    if (t >= S || j >= d) {
        return;
    }
    const int i = j * S + t;
    u[i] = ((E[j * V + prev[t]] * m1[i] + w2 * (E2[j * V + prev2[t]] * m2[i])) + F[j * nF + field]) + P[j * nP + t];
}

// Per row r: inv_std[r] = 1 / sqrt(mean((X_r - mean_r)^2) + eps), Y_r = inv_std[r] * (X_r - mean_r)
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void layer_norm_rows(const int R, const int C, const double eps, __global const double* X, __global double* Y,
                     __global double* inv_std) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    double part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        part += X[c * R + r];
    }
    const double mean = group_sum(scratch, part) / (double)C;
    part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const double z = X[c * R + r] - mean;
        part += z * z;
    }
    const double var = group_sum(scratch, part) / (double)C;
    const double s = 1.0 / sqrt(var + eps);
    if (l == 0) {
        inv_std[r] = s;
    }
    for (int c = l; c < C; c += ROW_GROUP) {
        Y[c * R + r] = s * (X[c * R + r] - mean);
    }
}

// Row-wise softmax in place: exp(max(x - rowmax, -60)) / row sum
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void softmax_rows(const int R, const int C, __global double* X) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    double part = -INFINITY;
    for (int c = l; c < C; c += ROW_GROUP) {
        part = fmax(part, X[c * R + r]);
    }
    const double top = group_max(scratch, part);
    part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const double e = exp(fmax(X[c * R + r] - top, -60.0));
        X[c * R + r] = e;
        part += e;
    }
    const double sum = group_sum(scratch, part);
    for (int c = l; c < C; c += ROW_GROUP) {
        X[c * R + r] /= sum;
    }
}

)CLC"
               R"CLC(// softmax_rows for few columns (attention over the M latents): one work-item per row.
__kernel void softmax_rows_narrow(const int R, const int C, __global double* X) {
    const int r = get_global_id(0);
    if (r >= R) {
        return;
    }
    double top = X[r];
    for (int c = 1; c < C; ++c) {
        top = fmax(top, X[c * R + r]);
    }
    double sum = 0.0;
    for (int c = 0; c < C; ++c) {
        const double e = exp(fmax(X[c * R + r] - top, -60.0));
        X[c * R + r] = e;
        sum += e;
    }
    for (int c = 0; c < C; ++c) {
        X[c * R + r] /= sum;
    }
}

// Inverted-dropout masks for one training pass, bit-identical to DeterministicRng::uniform() draws taken in the
// host order: the token mask row by row (S x d), then the second-token mask (S x d), then the hidden mask (S x h).
// Draw k (0-based) of the pass is splitmix64 of state + (k + 1) * golden; a draw below p zeroes the entry,
// otherwise it becomes keep = 1 / (1 - p). Masks are stored column-major like every other matrix. The 64-bit
// state arrives as its low and high 32-bit halves.
__kernel void dropout_masks(const int S, const int d, const int h, const uint state_low, const uint state_high,
                            const double p, const double keep, __global double* m1, __global double* m2,
                            __global double* mh) {
    const int k = get_global_id(0);
    const int token_draws = S * d;
    if (k >= 2 * token_draws + S * h) {
        return;
    }
    const ulong state = ((ulong)state_high << 32) | (ulong)state_low;
    ulong z = state + (ulong)(k + 1) * 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z = z ^ (z >> 31);
    const double unit = (double)(z >> 11) * (1.0 / 9007199254740992.0);
    const double value = unit < p ? 0.0 : keep;
    if (k < token_draws) {
        m1[(k % d) * S + k / d] = value;
    } else if (k < 2 * token_draws) {
        const int q = k - token_draws;
        m2[(q % d) * S + q / d] = value;
    } else {
        const int q = k - 2 * token_draws;
        mh[(q % h) * S + q / h] = value;
    }
}

// X (R x C, holding A V) becomes (U + A V) + (W_aux d_hat)^T in every row; W_aux is C x dim.
__kernel void add_inputs_and_conditioning(const int R, const int C, const int dim, __global double* X,
                                          __global const double* U, __global const double* W,
                                          __global const double* descriptor) {
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    if (r >= R || c >= C) {
        return;
    }
    double conditioning = 0.0;
    for (int k = 0; k < dim; ++k) {
        conditioning += W[k * C + c] * descriptor[k];
    }
    const int i = c * R + r;
    X[i] = (U[i] + X[i]) + conditioning;
}

// H (R x C, holding x W1^T) becomes tanh(H + b1) in place; dropped = H * mask
__kernel void bias_tanh_mask(const int R, const int C, __global double* H, __global const double* b,
                             __global const double* mask, __global double* dropped) {
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    if (r < R && c < C) {
        const int i = c * R + r;
        const double h = tanh(H[i] + b[c]);
        H[i] = h;
        dropped[i] = h * mask[i];
    }
}

// Per row r of X (R x C, holding h W_out^T): P = softmax(X + b) with exp(max(x - rowmax, -60)),
// loss[loss_offset + r] = -log(max(1e-12, P[r, target_r])); with write_gradient, X then holds
// dLoss/dlogits = (P - onehot(target)) / R instead of P.
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void softmax_loss_rows(const int R, const int C, __global double* X, __global const double* b,
                       __global const int* targets, __global double* loss, const int loss_offset,
                       const int write_gradient) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    const int target = targets[r];
    double part = -INFINITY;
    for (int c = l; c < C; c += ROW_GROUP) {
        part = fmax(part, X[c * R + r] + b[c]);
    }
    const double top = group_max(scratch, part);
    part = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const double e = exp(fmax((X[c * R + r] + b[c]) - top, -60.0));
        X[c * R + r] = e;
        part += e;
    }
    const double sum = group_sum(scratch, part);
    for (int c = l; c < C; c += ROW_GROUP) {
        const double p = X[c * R + r] / sum;
        if (c == target) {
            loss[loss_offset + r] = -log(fmax(1e-12, p));
        }
        X[c * R + r] = write_gradient != 0 ? (c == target ? p - 1.0 : p) / (double)R : p;
    }
}

// out (C x dim) = (column sums of dX (R x C)) d_hat^T, the W_aux gradient.
__kernel void descriptor_gradient(const int R, const int C, const int dim, __global const double* dX,
                                  __global const double* descriptor, __global double* out) {
    const int c = get_global_id(0);
    const int k = get_global_id(1);
    if (c >= C || k >= dim) {
        return;
    }
    double s = 0.0;
    for (int r = 0; r < R; ++r) {
        s += dX[c * R + r];
    }
    out[k * C + c] = s * descriptor[k];
}

// out[c * stride + offset] = sum_r X[r, c]
__kernel void column_sums(const int R, const int C, __global const double* X, const int stride, const int offset,
                          __global double* out) {
    const int c = get_global_id(0);
    if (c >= C) {
        return;
    }
    double s = 0.0;
    for (int r = 0; r < R; ++r) {
        s += X[c * R + r];
    }
    out[c * stride + offset] = s;
}

)CLC"
               R"CLC(// One ROW_GROUP work-group per column c of the R x C hidden layer: dpre = (dh * mask) * (1 - h^2), and
// bias_grad[c] = sum_r dpre[r, c] (the b1 gradient).
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void tanh_backward_columns(const int R, const int C, __global const double* dh, __global const double* mask,
                           __global const double* h, __global double* dpre, __global double* bias_grad) {
    __local double scratch[ROW_GROUP];
    const int c = get_group_id(0);
    const int l = get_local_id(0);
    double part = 0.0;
    for (int r = l; r < R; r += ROW_GROUP) {
        const int i = c * R + r;
        const double v = (dh[i] * mask[i]) * (1.0 - h[i] * h[i]);
        dpre[i] = v;
        part += v;
    }
    const double s = group_sum(scratch, part);
    if (l == 0) {
        bias_grad[c] = s;
    }
}

// Per row: dX = [base +] inv_std * ((dY - mean(dY)) - mean(dY * Y) * Y)
__kernel __attribute__((reqd_work_group_size(ROW_GROUP, 1, 1)))
void layer_norm_backward(const int R, const int C, __global const double* Y, __global const double* inv_std,
                         __global const double* dY, const int has_base, __global const double* base,
                         __global double* dX) {
    __local double scratch[ROW_GROUP];
    const int r = get_group_id(0);
    const int l = get_local_id(0);
    double part_grad = 0.0;
    double part_dot = 0.0;
    for (int c = l; c < C; c += ROW_GROUP) {
        const int i = c * R + r;
        part_grad += dY[i];
        part_dot += dY[i] * Y[i];
    }
    const double mean_grad = group_sum(scratch, part_grad) / (double)C;
    const double mean_grad_dot = group_sum(scratch, part_dot) / (double)C;
    const double s = inv_std[r];
    for (int c = l; c < C; c += ROW_GROUP) {
        const int i = c * R + r;
        const double v = s * ((dY[i] - mean_grad) - mean_grad_dot * Y[i]);
        dX[i] = has_base != 0 ? base[i] + v : v;
    }
}

// Per row: dS = A * (dA - sum_c dA * A)
__kernel void softmax_backward_rows(const int R, const int C, __global const double* A, __global const double* dA,
                                    __global double* dS) {
    const int r = get_global_id(0);
    if (r >= R) {
        return;
    }
    double dot = 0.0;
    for (int c = 0; c < C; ++c) {
        dot += dA[c * R + r] * A[c * R + r];
    }
    for (int c = 0; c < C; ++c) {
        const int i = c * R + r;
        dS[i] = A[i] * (dA[i] - dot);
    }
}

// Work-item (t, j): when step t is the first occurrence of token prev_t, gE[prev_t, j] = sum over the steps t' >= t
// with prev_t' = prev_t (in step order) of dU[t', j] * m1[t', j]; likewise gE2[prev2_t, j] with
// w2 * (dU * m2). Work-item (0, j) also writes gF[field, j] = sum_t dU[t, j] (F has nF rows). Rows of tokens
// absent from the pass keep the zeros they were cleared to.
__kernel void embedding_gradients(const int S, const int d, const int V, const double w2, const int nF, const int field,
                                  __global const int* prev, __global const int* prev2, __global const double* dU,
                                  __global const double* m1, __global const double* m2, __global double* gE,
                                  __global double* gE2, __global double* gF) {
    const int t = get_global_id(0);
    const int j = get_global_id(1);
    if (t >= S || j >= d) {
        return;
    }
    const int token = prev[t];
    const int token2 = prev2[t];
    bool first = true;
    bool first2 = true;
    for (int q = 0; q < t; ++q) {
        first = first && prev[q] != token;
        first2 = first2 && prev2[q] != token2;
    }
    if (first) {
        double s = 0.0;
        for (int q = t; q < S; ++q) {
            if (prev[q] == token) {
                s += dU[j * S + q] * m1[j * S + q];
            }
        }
        gE[j * V + token] = s;
    }
    if (first2) {
        double s = 0.0;
        for (int q = t; q < S; ++q) {
            if (prev2[q] == token2) {
                s += w2 * (dU[j * S + q] * m2[j * S + q]);
            }
        }
        gE2[j * V + token2] = s;
    }
    if (t == 0) {
        double field_sum = 0.0;
        for (int q = 0; q < S; ++q) {
            field_sum += dU[j * S + q];
        }
        gF[j * nF + field] = field_sum;
    }
}

// Adam over every packed parameter: m = b1 m + (1 - b1) g, v = b2 v + (1 - b2) g^2,
// p -= lr (m / c1) / (sqrt(v / c2) + eps), with the bias corrections applied as multiplications by inv_c1 = 1 / c1
// and inv_c2 = 1 / c2 (one division per entry instead of three; double-precision division is slow on GPUs);
// entries in [decay_begin, decay_end) are then multiplied by decay_factor (decoupled weight decay on W_out).
__kernel void adam_update(const int n, const double lr, const double inv_c1, const double inv_c2, const double b1,
                          const double b2, const double eps, const int decay_begin, const int decay_end,
                          const double decay_factor, __global double* p, __global const double* g,
                          __global double* m, __global double* v) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const double gi = g[i];
    const double mi = b1 * m[i] + (1.0 - b1) * gi;
    const double vi = b2 * v[i] + (1.0 - b2) * (gi * gi);
    m[i] = mi;
    v[i] = vi;
    double pi = p[i] - (lr * (mi * inv_c1)) / (sqrt(vi * inv_c2) + eps);
    if (i >= decay_begin && i < decay_end) {
        pi *= decay_factor;
    }
    p[i] = pi;
}
)CLC";
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\compute\device_buffer.hpp ===
#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/opencl_runtime.hpp"

#include <CL/cl.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

namespace forward_analysis {

// Device memory for count elements of T (cl_double or cl_int). Grows on reserve() and never shrinks, so step
// buffers sized for the longest field are reused by every shorter one. region() makes a view of part of another
// buffer (an OpenCL sub-buffer) that kernels and CLBlast use like a buffer of its own.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    // View of count elements of parent starting at element offset; offset * sizeof(T) must be a multiple of the
    // device's base address alignment. The view must be released before parent grows or is destroyed.
    static DeviceBuffer region(const DeviceBuffer& parent, std::size_t offset, std::size_t count) {
        const cl_buffer_region span{sizeof(T) * offset, sizeof(T) * std::max<std::size_t>(1, count)};
        cl_int status = CL_SUCCESS;
        DeviceBuffer view;
        view.mem_ = clCreateSubBuffer(parent.mem_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &span, &status);
        OpenClRuntime::check(status, "clCreateSubBuffer");
        view.capacity_ = count;
        return view;
    }

    // Releases the device memory (or view).
    void reset() noexcept {
        if (mem_ != nullptr) {
            clReleaseMemObject(mem_);
            mem_ = nullptr;
        }
        capacity_ = 0;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        swap(other);
        return *this;
    }
    ~DeviceBuffer() { reset(); }

    // Ensures room for count elements; existing contents are discarded when the buffer has to grow.
    void reserve(const OpenClRuntime& runtime, std::size_t count) {
        if (count <= capacity_ && mem_ != nullptr) {
            return;
        }
        if (mem_ != nullptr) {
            clReleaseMemObject(mem_);
            mem_ = nullptr;
        }
        cl_int status = CL_SUCCESS;
        mem_ = clCreateBuffer(runtime.context(), CL_MEM_READ_WRITE, sizeof(T) * std::max<std::size_t>(1, count), nullptr,
                              &status);
        OpenClRuntime::check(status, "clCreateBuffer");
        capacity_ = count;
    }

    // Copies count host elements to the buffer starting at element offset. A non-blocking write requires data to
    // stay valid until the next blocking read on the same queue.
    void write(const OpenClRuntime& runtime, const T* data, std::size_t count, bool blocking, std::size_t offset = 0) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueWriteBuffer(runtime.queue(), mem_, blocking ? CL_TRUE : CL_FALSE, sizeof(T) * offset,
                                                  sizeof(T) * count, data, 0, nullptr, nullptr),
                             "clEnqueueWriteBuffer");
    }

    // Blocking copy of count elements starting at element offset to the host; waits for every command queued before it.
    void read(const OpenClRuntime& runtime, T* data, std::size_t count, std::size_t offset = 0) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueReadBuffer(runtime.queue(), mem_, CL_TRUE, sizeof(T) * offset, sizeof(T) * count, data,
                                                 0, nullptr, nullptr),
                             "clEnqueueReadBuffer");
    }

    // Sets count elements starting at element offset to value.
    void fill(const OpenClRuntime& runtime, T value, std::size_t count, std::size_t offset = 0) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueFillBuffer(runtime.queue(), mem_, &value, sizeof(T), sizeof(T) * offset,
                                                 sizeof(T) * count, 0, nullptr, nullptr),
                             "clEnqueueFillBuffer");
    }

    // Device-side copy of the first count elements of source into this buffer.
    void copy_from(const OpenClRuntime& runtime, const DeviceBuffer& source, std::size_t count) const {
        if (count == 0) {
            return;
        }
        OpenClRuntime::check(clEnqueueCopyBuffer(runtime.queue(), source.mem_, mem_, 0, 0, sizeof(T) * count, 0, nullptr,
                                                 nullptr),
                             "clEnqueueCopyBuffer");
    }

    cl_mem mem() const noexcept { return mem_; }
    std::size_t capacity() const noexcept { return capacity_; }

private:
    void swap(DeviceBuffer& other) noexcept {
        std::swap(mem_, other.mem_);
        std::swap(capacity_, other.capacity_);
    }

    // OpenCL buffer object, null until the first reserve().
    cl_mem mem_ = nullptr;
    // Number of T elements the buffer can hold.
    std::size_t capacity_ = 0;
};

}  // namespace forward_analysis

#endif

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\compute\device_gemm.hpp ===
#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/decoder_kernel_source.hpp"
#include "forward_analysis/compute/opencl_runtime.hpp"

#include <clblast.h>

#include <CL/cl.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace forward_analysis {

// Double-precision GEMM on the runtime's queue, column-major like Eigen:
// C (m x n) = alpha op(A) op(B) + beta C, where op transposes when requested and ld* are the stored row counts.
// Three implementations, chosen by shape:
//   m n <= kReduceMaximumOutputs                          gemm_reduce (one work-group per output element)
//   k <= kDirectMaximumDepth and m n k <= kDirectMaximumWork  gemm_direct (one work-item per output element)
//   otherwise                                              CLBlast
// Measured on an RTX 4070: CLBlast's kernels leave most of the device idle when C has few entries or the inner
// dimension is short (about 20 us for 41 x 8 x 256 against 9 us for gemm_reduce, 18 us for 41 x 256 x 8 against
// 11 us for gemm_direct), and every CLBlast call also costs 15-20 us of host time.
class DeviceGemm {
public:
    static constexpr std::size_t kReduceMaximumOutputs = 1024;
    static constexpr std::size_t kDirectMaximumDepth = 64;
    static constexpr std::size_t kDirectMaximumWork = std::size_t{1} << 20;

    static void multiply(const OpenClRuntime& runtime,
                         bool transpose_a,
                         bool transpose_b,
                         std::size_t m,
                         std::size_t n,
                         std::size_t k,
                         double alpha,
                         cl_mem a,
                         std::size_t lda,
                         cl_mem b,
                         std::size_t ldb,
                         double beta,
                         cl_mem c,
                         std::size_t ldc) {
        const int trans_a = transpose_a ? 1 : 0;
        const int trans_b = transpose_b ? 1 : 0;
        if (m * n <= kReduceMaximumOutputs) {
            runtime.launch_groups("gemm_reduce", m * n, 1, DecoderKernelSource::kRowGroup, 1, m, n, k, trans_a, trans_b,
                                  alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        }
        if (k <= kDirectMaximumDepth && m * n * k <= kDirectMaximumWork) {
            runtime.launch("gemm_direct", m, n, m, n, k, trans_a, trans_b, alpha, a, lda, b, ldb, beta, c, ldc);
            return;
        }
        cl_command_queue queue = runtime.queue();
        const clblast::StatusCode status = clblast::Gemm<double>(
            clblast::Layout::kColMajor, transpose_a ? clblast::Transpose::kYes : clblast::Transpose::kNo,
            transpose_b ? clblast::Transpose::kYes : clblast::Transpose::kNo, m, n, k, alpha, a, 0, lda, b, 0, ldb, beta, c,
            0, ldc, &queue, nullptr);
        if (status != clblast::StatusCode::kSuccess) {
            throw std::runtime_error("CLBlast Gemm failed with status " + std::to_string(static_cast<int>(status)));
        }
    }
};

}  // namespace forward_analysis

#endif

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\compute\device_parameter.hpp ===
#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/device_buffer.hpp"

#include <CL/cl.h>

#include <cstddef>
#include <utility>

namespace forward_analysis {

// One trainable tensor inside a DeviceParameterArena: where it starts in the packed buffers, its host shape, and
// views of its values and gradient (column-major, same layout as the host tensor).
class DeviceParameter {
public:
    // Records the tensor's place in the arena and drops any views of a previous layout.
    void place(std::size_t offset, std::size_t rows, std::size_t cols) {
        release_views();
        offset_ = offset;
        rows_ = rows;
        cols_ = cols;
    }

    // Takes the views of this tensor's value and gradient ranges.
    void attach(DeviceBuffer<double> value_view, DeviceBuffer<double> gradient_view) {
        value_ = std::move(value_view);
        gradient_ = std::move(gradient_view);
    }

    void release_views() noexcept {
        value_.reset();
        gradient_.reset();
    }

    cl_mem value() const noexcept { return value_.mem(); }
    cl_mem gradient() const noexcept { return gradient_.mem(); }
    std::size_t offset() const noexcept { return offset_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return rows_ * cols_; }

private:
    // First element of the tensor in the arena's packed buffers.
    std::size_t offset_ = 0;
    // Rows and columns of the host tensor (a vector has one column).
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    // View of the tensor's current values.
    DeviceBuffer<double> value_;
    // View of dLoss/dvalue of the latest training pass.
    DeviceBuffer<double> gradient_;
};

}  // namespace forward_analysis

#endif

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\compute\device_parameter_arena.hpp ===
#pragma once

#if FORWARD_ANALYSIS_WITH_OPENCL

#include "forward_analysis/compute/device_buffer.hpp"
#include "forward_analysis/compute/device_parameter.hpp"
#include "forward_analysis/compute/opencl_runtime.hpp"
#include "forward_analysis/math/adam_moment.hpp"

#include <Eigen/Core>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace forward_analysis {

// Every trainable decoder tensor packed into four device buffers (values, gradients, Adam first and second
// moments). Each tensor starts at a multiple of the device's base address alignment so it can be viewed as a
// buffer of its own; the gaps stay zero. One kernel then updates all parameters, and gradients of a leading run
// of tensors are cleared with one fill.
class DeviceParameterArena {
public:
    // Lays the tensors out in the given order with the given (rows, cols) shapes and sizes the buffers; existing
    // contents are discarded and everything starts at zero.
    void layout(const OpenClRuntime& runtime,
                const std::vector<DeviceParameter*>& tensors,
                const std::vector<std::pair<std::size_t, std::size_t>>& shapes) {
        for (DeviceParameter* tensor : tensors_) {
            tensor->release_views();
        }
        tensors_ = tensors;
        const std::size_t align = runtime.alignment_bytes() / sizeof(double);
        std::size_t next = 0;
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            tensors[i]->place(next, shapes[i].first, shapes[i].second);
            next += tensors[i]->size();
            next = (next + align - 1) / align * align;
        }
        size_ = next;
        for (DeviceBuffer<double>* buffer : {&values_, &gradients_, &first_, &second_}) {
            buffer->reserve(runtime, size_);
            buffer->fill(runtime, 0.0, size_);
        }
        for (DeviceParameter* tensor : tensors_) {
            tensor->attach(DeviceBuffer<double>::region(values_, tensor->offset(), tensor->size()),
                           DeviceBuffer<double>::region(gradients_, tensor->offset(), tensor->size()));
        }
    }

    // Copies a host tensor and its Adam moments into the tensor's ranges.
    template <typename Dense>
    void write(const OpenClRuntime& runtime, const DeviceParameter& tensor, const Dense& value, AdamMoment& moment) const {
        moment.ensure_shape(static_cast<Eigen::Index>(tensor.rows()), static_cast<Eigen::Index>(tensor.cols()));
        values_.write(runtime, value.data(), tensor.size(), true, tensor.offset());
        first_.write(runtime, moment.first().data(), tensor.size(), true, tensor.offset());
        second_.write(runtime, moment.second().data(), tensor.size(), true, tensor.offset());
    }

    // Copies the tensor's values and Adam moments back into the host tensor and its AdamMoment.
    template <typename Dense>
    void read(const OpenClRuntime& runtime, const DeviceParameter& tensor, Dense& value, AdamMoment& moment) const {
        const Eigen::Index rows = static_cast<Eigen::Index>(tensor.rows());
        const Eigen::Index cols = static_cast<Eigen::Index>(tensor.cols());
        value.resize(rows, cols);
        values_.read(runtime, value.data(), tensor.size(), tensor.offset());
        Eigen::MatrixXd first(rows, cols);
        Eigen::MatrixXd second(rows, cols);
        first_.read(runtime, first.data(), tensor.size(), tensor.offset());
        second_.read(runtime, second.data(), tensor.size(), tensor.offset());
        moment.assign(first, second);
    }

    // Zeroes the gradients of every tensor from the start of the arena through last (inclusive).
    void zero_gradients_through(const OpenClRuntime& runtime, const DeviceParameter& last) const {
        gradients_.fill(runtime, 0.0, last.offset() + last.size());
    }

    // One Adam step on every tensor with bias corrections for the 1-based step count (as AdamMoment::apply, with
    // the corrections applied through their reciprocals), then decayed's values multiplied by decay_factor.
    void adam(const OpenClRuntime& runtime,
              double learning_rate,
              std::uint64_t step,
              const DeviceParameter& decayed,
              double decay_factor) const {
        const double t = static_cast<double>(step);
        const double correction1 = 1.0 - std::pow(AdamMoment::beta1(), t);
        const double correction2 = 1.0 - std::pow(AdamMoment::beta2(), t);
        runtime.launch("adam_update", size_, 1, size_, learning_rate, 1.0 / correction1, 1.0 / correction2, AdamMoment::beta1(),
                       AdamMoment::beta2(), AdamMoment::epsilon(), decayed.offset(), decayed.offset() + decayed.size(),
                       decay_factor, values_.mem(), gradients_.mem(), first_.mem(), second_.mem());
    }

private:
    // Packed buffers, size_ elements each: values, gradients of the latest pass, Adam first and second moments.
    DeviceBuffer<double> values_;
    DeviceBuffer<double> gradients_;
    DeviceBuffer<double> first_;
    DeviceBuffer<double> second_;
    // Elements in the packed layout, alignment gaps included.
    std::size_t size_ = 0;
    // Tensors placed by the latest layout() (owned by the caller), whose views are released before a new layout.
    std::vector<DeviceParameter*> tensors_;
};

}  // namespace forward_analysis

#endif

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\compute\opencl_runtime.hpp ===
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

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\config\distiller_config.hpp ===
#pragma once

#include "forward_analysis/config/narration_style.hpp"
#include "forward_analysis/config/slot_binding.hpp"

#include <cstddef>
#include <stdexcept>

namespace forward_analysis {

struct DistillerConfig {
    // Width of every channel token, latent and decoder input vector.
    std::size_t d_model = 256;
    // Number of learned Perceiver latent queries (M); the encoder output Z is num_latents x d_model.
    std::size_t num_latents = 8;
    // How many eigenvalue signs the eigen-sign head predicts: signs of the descending eigenvalues of the zero-diagonal
    // sample covariance of the first eigen_sign_dim input coordinates.
    std::size_t eigen_sign_dim = 4;
    // Width of the decoder's tanh hidden layer between its input vector and the vocabulary logits.
    std::size_t decoder_hidden_dim = 256;
    // Longest token sequence (including BOS/EOS) the decoder trains on or generates for one observation field.
    std::size_t max_field_tokens = 160;
    // Adam step size for the decoder parameters.
    double decoder_learning_rate = 0.001;
    // Probability of zeroing each decoder token-embedding entry and tanh hidden activation during training (inverted dropout).
    double decoder_dropout = 0.15;
    // Decoupled weight decay on the decoder's vocabulary projection: W_out -= lr * decay * W_out after every Adam step.
    double decoder_output_weight_decay = 0.1;
    // Run decoder training passes and evaluation losses on an OpenCL GPU (CLBlast GEMMs + OpenCL C kernels, double
    // precision) when the library is built with FORWARD_ANALYSIS_WITH_OPENCL and such a GPU is present; otherwise
    // they run on the CPU with Eigen.
    bool use_gpu = true;
    // Fine-tuning stops once the held-out token cross-entropy has not decreased for this many consecutive epochs.
    std::size_t early_stopping_patience = 5;
    // Weight of the observation-descriptor regression loss (Aux1) in the total loss.
    double aux1_observation_weight = 0.3;
    // Weight of the eigen-sign auxiliary loss (Aux2) in the total loss.
    double aux2_eigen_sign_weight = 0.2;
    // |activation| at or below this value counts as "not firing" on a sample.
    double dead_activation_threshold = 1e-6;
    // Singular values below this fraction of the largest one do not count toward effective rank.
    double spectral_rank_tolerance = 0.05;
    // A layer lies on a ray from the origin when its second uncentered singular value is below this fraction of the first.
    double ray_tolerance = 1e-3;
    // Multiple correlation with the two leading hidden axis scores at or above this value makes an input coordinate a driver.
    double driver_correlation_threshold = 0.75;
    // A prediction whose spread is at most this fraction of its target's spread counts as flat.
    double flat_output_ratio = 0.2;
    // Singular values of a centered activation matrix above this absolute value count toward its numerical rank.
    double numerical_rank_tolerance = 1e-3;
    // A layer whose numerical rank is at most this value is a narrow representation.
    std::size_t narrow_rank_limit = 2;
    // Prediction range / target range below this value is a smaller range.
    double range_ratio_low = 0.8;
    // Prediction range / target range above this value is a larger range.
    double range_ratio_high = 1.2;
    // Narrator for the measured-facts text that replaces unverifiable decoder fields.
    NarrationStyle narration_style = NarrationStyle::kBatchStatistics;
    // How numbers in the training text become slot tokens filled from the measured telemetry.
    SlotBinding slot_binding = SlotBinding::kShapeFacts;

    void validate() const {
        if (d_model == 0) {
            throw std::invalid_argument("d_model must be positive");
        }
        if (num_latents == 0) {
            throw std::invalid_argument("num_latents must be positive");
        }
        if (decoder_hidden_dim == 0) {
            throw std::invalid_argument("decoder_hidden_dim must be positive");
        }
        if (max_field_tokens < 2 || max_field_tokens > 512) {
            throw std::invalid_argument("max_field_tokens must be in [2, 512]");
        }
        if (!(decoder_learning_rate > 0.0)) {
            throw std::invalid_argument("decoder_learning_rate must be positive");
        }
        if (!(decoder_dropout >= 0.0 && decoder_dropout < 1.0)) {
            throw std::invalid_argument("decoder_dropout must be in [0, 1)");
        }
        if (!(decoder_output_weight_decay >= 0.0)) {
            throw std::invalid_argument("decoder_output_weight_decay must be non-negative");
        }
        if (early_stopping_patience == 0) {
            throw std::invalid_argument("early_stopping_patience must be positive");
        }
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\config\narration_style.hpp ===
#pragma once

namespace forward_analysis {

// Which deterministic narrator writes the measured-facts text (the fallback for unverifiable decoder fields).
enum class NarrationStyle {
    // ObservationNarrator: dead units, rays, gating, drivers and output fidelity in descriptive prose.
    kMeasuredFacts,
    // BatchStatisticsNarrator: per-layer zero coordinates and non-positive observations, input associations,
    // numerical ranks, range ratios and sign agreement, in the corrected dataset's phrasing.
    kBatchStatistics,
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\config\slot_binding.hpp ===
#pragma once

namespace forward_analysis {

// How numbers in the training text become slot tokens the decoder writes and the measured telemetry fills.
enum class SlotBinding {
    // ObservationDelexicalizer: shape and index references matched by value against layer widths, dead / live
    // counts, gating splits, driver coordinates and index lists.
    kShapeFacts,
    // BatchStatementDelexicalizer: every number in a batch-statistic statement becomes the slot of the quantity
    // that statement position reports (N, W_l, ZERO_Ll, NONPOS_Ll, STRONG_k, WEAK_k, RANK_Ll, D_OUT, range
    // counts, SIGN_AGREE, SIGN_ENTRIES), independent of its value.
    kBatchStatistics,
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\dataset\dataset_shape_summary.hpp ===
#pragma once

#include "forward_analysis/dataset/training_sample.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

// Range of every shape dimension across a loaded batch of samples.
struct DatasetShapeSummary {
    // Number of samples summarized.
    std::size_t sample_count = 0;
    // Smallest and largest batch size (rows) seen.
    std::size_t min_rows = std::numeric_limits<std::size_t>::max();
    std::size_t max_rows = 0;
    // Smallest and largest input width seen.
    std::size_t min_input_dim = std::numeric_limits<std::size_t>::max();
    std::size_t max_input_dim = 0;
    // Smallest and largest hidden layer count seen.
    std::size_t min_layers = std::numeric_limits<std::size_t>::max();
    std::size_t max_layers = 0;
    // Smallest and largest single hidden layer width seen.
    std::size_t min_hidden_width = std::numeric_limits<std::size_t>::max();
    std::size_t max_hidden_width = 0;
    // Smallest and largest output width seen.
    std::size_t min_output_dim = std::numeric_limits<std::size_t>::max();
    std::size_t max_output_dim = 0;
    // Smallest and largest channel-token count (encoder sequence length) seen.
    std::size_t min_channels = std::numeric_limits<std::size_t>::max();
    std::size_t max_channels = 0;

    static DatasetShapeSummary from(const std::vector<TrainingSample>& samples) {
        DatasetShapeSummary s;
        for (const TrainingSample& sample : samples) {
            const TelemetryShape shape = sample.telemetry.shape();
            s.sample_count += 1;
            widen(s.min_rows, s.max_rows, shape.num_samples);
            widen(s.min_input_dim, s.max_input_dim, shape.input_dim);
            widen(s.min_layers, s.max_layers, shape.hidden_widths.size());
            for (std::size_t w : shape.hidden_widths) {
                widen(s.min_hidden_width, s.max_hidden_width, w);
            }
            widen(s.min_output_dim, s.max_output_dim, shape.output_dim);
            widen(s.min_channels, s.max_channels, shape.channel_count());
        }
        return s;
    }

    nlohmann::json to_json() const {
        return {{"sample_count", sample_count},
                {"rows", {min_rows, max_rows}},
                {"input_dim", {min_input_dim, max_input_dim}},
                {"hidden_layers", {min_layers, max_layers}},
                {"hidden_width", {min_hidden_width, max_hidden_width}},
                {"output_dim", {min_output_dim, max_output_dim}},
                {"channel_tokens", {min_channels, max_channels}}};
    }

    std::string to_string() const {
        std::ostringstream out;
        out << sample_count << " samples; rows " << min_rows << "-" << max_rows << ", inputs " << min_input_dim
            << "-" << max_input_dim << ", hidden layers " << min_layers << "-" << max_layers << " of width "
            << min_hidden_width << "-" << max_hidden_width << ", outputs " << min_output_dim << "-" << max_output_dim
            << ", channel tokens " << min_channels << "-" << max_channels;
        return out.str();
    }

private:
    static void widen(std::size_t& low, std::size_t& high, std::size_t value) {
        low = std::min(low, value);
        high = std::max(high, value);
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\dataset\ground_truth_targets.hpp ===
#pragma once

#include "forward_analysis/description/observation_text.hpp"

#include <Eigen/Core>

namespace forward_analysis {

struct GroundTruthTargets {
    // Expected signs (+1/-1) of the dominant first-layer covariance eigenvector components; Aux2 target.
    Eigen::VectorXd eigen_signs;
    // The four qualitative observation fields the decoder learns to produce.
    ObservationText observations;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\dataset\json_tensor_reader.hpp ===
#pragma once

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace forward_analysis {

struct JsonTensorReader {
    static Eigen::MatrixXd matrix(const nlohmann::json& node, const std::string& name) {
        if (!node.is_array() || node.empty() || !node.front().is_array() || node.front().empty()) {
            throw std::invalid_argument(name + " must be a non-empty array of non-empty rows");
        }
        const Eigen::Index rows = static_cast<Eigen::Index>(node.size());
        const Eigen::Index cols = static_cast<Eigen::Index>(node.front().size());
        Eigen::MatrixXd out(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            const nlohmann::json& row = node[static_cast<std::size_t>(r)];
            if (!row.is_array() || static_cast<Eigen::Index>(row.size()) != cols) {
                throw std::invalid_argument(name + " row " + std::to_string(r) + " must have " +
                                            std::to_string(cols) + " values");
            }
            for (Eigen::Index c = 0; c < cols; ++c) {
                out(r, c) = number(row[static_cast<std::size_t>(c)], name);
            }
        }
        return out;
    }

    static Eigen::VectorXd vector(const nlohmann::json& node, const std::string& name) {
        if (!node.is_array() || node.empty()) {
            throw std::invalid_argument(name + " must be a non-empty array");
        }
        Eigen::VectorXd out(static_cast<Eigen::Index>(node.size()));
        for (Eigen::Index i = 0; i < out.size(); ++i) {
            out(i) = number(node[static_cast<std::size_t>(i)], name);
        }
        return out;
    }

    static std::vector<Eigen::MatrixXd> matrix_list(const nlohmann::json& node, const std::string& name) {
        if (!node.is_array() || node.empty()) {
            throw std::invalid_argument(name + " must be a non-empty array of matrices");
        }
        std::vector<Eigen::MatrixXd> out;
        out.reserve(node.size());
        for (std::size_t l = 0; l < node.size(); ++l) {
            out.push_back(matrix(node[l], name + "[" + std::to_string(l) + "]"));
        }
        return out;
    }

private:
    static double number(const nlohmann::json& value, const std::string& name) {
        if (!value.is_number()) {
            throw std::invalid_argument(name + " contains a non-numeric value");
        }
        return value.get<double>();
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\dataset\training_dataset_importer.hpp ===
#pragma once

#include "forward_analysis/analysis/observation_extractor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/dataset/json_tensor_reader.hpp"
#include "forward_analysis/dataset/training_sample.hpp"
#include "forward_analysis/telemetry/telemetry_builder.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace forward_analysis {

// Loads JSONL records of any shape: each line carries its own input width, hidden layer count and
// widths, output width and batch size.
class TrainingDatasetImporter {
public:
    explicit TrainingDatasetImporter(const DistillerConfig& config = DistillerConfig{})
        : config_(config) {
        config_.validate();
    }

    std::vector<TrainingSample> load_jsonl(const std::string& path) const {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Cannot open training dataset: " + path);
        }
        return read_jsonl(in, path);
    }

    std::vector<TrainingSample> read_jsonl(std::istream& in, const std::string& source = "<stream>") const {
        std::vector<TrainingSample> samples;
        std::string line;
        std::size_t line_number = 0;
        while (std::getline(in, line)) {
            ++line_number;
            if (line.find_first_not_of(" \t\r") == std::string::npos) {
                continue;
            }
            try {
                samples.push_back(parse_sample(nlohmann::json::parse(line)));
            } catch (const std::exception& error) {
                throw std::invalid_argument(source + ":" + std::to_string(line_number) + ": " + error.what());
            }
        }
        if (samples.empty()) {
            throw std::invalid_argument(source + " contains no samples");
        }
        return samples;
    }

    TrainingSample parse_sample(const nlohmann::json& record) const {
        TrainingSample sample;
        sample.sample_id = record.at("sample_id").get<std::string>();

        const nlohmann::json& telemetry = record.at("telemetry");
        sample.telemetry = TelemetryBuilder::build(
            JsonTensorReader::matrix(telemetry.at("inputs"), "telemetry.inputs"),
            JsonTensorReader::matrix(telemetry.at("targets"), "telemetry.targets"),
            JsonTensorReader::matrix(telemetry.at("predictions"), "telemetry.predictions"),
            JsonTensorReader::matrix_list(telemetry.at("layer_activations"), "telemetry.layer_activations"),
            config_);
        sample.measured = ObservationExtractor::extract(sample.telemetry, config_);

        const nlohmann::json& targets = record.at("ground_truth_targets");
        GroundTruthTargets& gt = sample.ground_truth_targets;
        gt.observations = ObservationText::from_json(targets.at("target_text"));
        gt.eigen_signs = targets.contains("eigen_signs")
                             ? JsonTensorReader::vector(targets.at("eigen_signs"), "ground_truth_targets.eigen_signs")
                             : sample.measured.eigen_signs;
        return sample;
    }

private:
    // Thresholds used when deriving masks and measured observations from imported telemetry.
    DistillerConfig config_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\dataset\training_sample.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/dataset/ground_truth_targets.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <string>

namespace forward_analysis {

struct TrainingSample {
    // Identifier copied from the JSONL record.
    std::string sample_id;
    // Validated telemetry with derived residuals and dead masks.
    ForwardTelemetryTensor telemetry;
    // Eigen signs and observation text from the dataset file.
    GroundTruthTargets ground_truth_targets;
    // Facts measured directly from the telemetry at load time (Aux1 target source and narrator input).
    NetworkObservations measured;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\autoregressive_causal_decoder.hpp ===
#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/decoder/decoder_device_slot.hpp"
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

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\batch_statement_delexicalizer.hpp ===
#pragma once

#include "forward_analysis/description/batch_statement_checker.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Replaces every number of a batch-statistic statement with the digit slot of the quantity at that position:
//   "Across 14 observations, the hidden widths are 27 and 14. Layer 1 has 0 coordinates that remain zero ..."
//   -> "Across <COUNT:N|d> observations , the hidden widths are <COUNT:W_1|d> and <COUNT:W_2|d> . Layer 1 has
//       <COUNT:ZERO_L1|d> coordinates that remain zero ..."
// Layer numbers ("Layer 1") stay literal; a slot for a layer the network lacks is unfillable at generation time.
// The slots are filled by ObservationDelexicalizer::relexicalize from ShapeSlotLexicon counts.
struct BatchStatementDelexicalizer {
    static std::vector<std::string> delexicalize(const std::string& text) {
        const std::vector<std::string> words = LexemeSplitter::split(text);
        std::vector<std::string> lower;
        for (const std::string& w : words) {
            lower.push_back(LexemeSplitter::lowercase(w));
        }
        std::vector<std::string> out;
        std::size_t layer_context = 0;
        for (std::size_t i = 0; i < words.size();) {
            std::vector<std::size_t> values;
            std::vector<std::size_t> at;
            std::vector<std::string> names;
            std::size_t end = i;
            if (BatchStatementChecker::match(lower, i, {"across", "#", "observations"}, values, at)) {
                names = {"N"};
            } else if (BatchStatementChecker::match(lower, i, {"hidden", "widths", "are", "#*"}, values, at)) {
                for (std::size_t k = 0; k < at.size(); ++k) {
                    names.push_back("W_" + std::to_string(k + 1));
                }
            } else if (BatchStatementChecker::match(lower, i, {"layer", "#", "has", "#", "coordinates"}, values, at)) {
                layer_context = values[0];
                names = {"", "ZERO_L" + std::to_string(values[0])};
            } else if (layer_context > 0 &&
                       BatchStatementChecker::match(lower, i, {"#", "observations", "with", "no", "positive"}, values, at)) {
                names = {"NONPOS_L" + std::to_string(layer_context)};
            } else if (BatchStatementChecker::match(lower, i, {"input", "coordinates", "#", "and", "#", "have", "the", "strongest"},
                                                    values, at)) {
                names = {"STRONG_1", "STRONG_2"};
            } else if (BatchStatementChecker::match(lower, i, {"coordinates", "#", "and", "#", "have", "the", "weakest"}, values, at)) {
                names = {"WEAK_1", "WEAK_2"};
            } else if (BatchStatementChecker::match(lower, i, {"numerical", "ranks", "#*"}, values, at)) {
                for (std::size_t k = 0; k < at.size(); ++k) {
                    names.push_back("RANK_L" + std::to_string(k + 1));
                }
            } else if (BatchStatementChecker::match(lower, i, {"over", "#", "output", "coordinates"}, values, at)) {
                names = {"D_OUT"};
            } else if (BatchStatementChecker::match(lower, i, {"#", "have", "a", "smaller"}, values, at)) {
                names = {"SMALLER_RANGE"};
            } else if (BatchStatementChecker::match(lower, i, {"#", "have", "a", "larger"}, values, at)) {
                names = {"LARGER_RANGE"};
            } else if (BatchStatementChecker::match(lower, i, {"#", "have", "broadly", "comparable"}, values, at)) {
                names = {"COMPARABLE_RANGE"};
            } else if (BatchStatementChecker::match(lower, i, {"agree", "in", "#", "of", "#", "entries"}, values, at)) {
                names = {"SIGN_AGREE", "SIGN_ENTRIES"};
            }
            if (names.empty()) {
                out.push_back(words[i]);
                ++i;
                continue;
            }
            end = at.back() + 1;
            std::size_t slot = 0;
            for (std::size_t k = i; k < end; ++k) {
                if (slot < at.size() && at[slot] == k) {
                    out.push_back(names[slot].empty() ? words[k] : "<COUNT:" + names[slot] + "|d>");
                    ++slot;
                } else {
                    out.push_back(words[k]);
                }
            }
            i = end;
        }
        return out;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\decoder_device_mirror.hpp ===
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

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\decoder_device_slot.hpp ===
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

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\decoder_parameter_gradients.hpp ===
#pragma once

#include <Eigen/Core>

namespace forward_analysis {

// Gradients of one field's mean token cross-entropy with respect to every decoder parameter and the latents.
struct DecoderParameterGradients {
    // Previous-token embeddings (V x d).
    Eigen::MatrixXd token_embeddings;
    // Second-previous-token embeddings (V x d).
    Eigen::MatrixXd second_token_embeddings;
    // Field embeddings (4 x d).
    Eigen::MatrixXd field_embeddings;
    // Attention query projection (d x d).
    Eigen::MatrixXd query_weights;
    // Attention key projection (d x d).
    Eigen::MatrixXd key_weights;
    // Attention value projection (d x d).
    Eigen::MatrixXd value_weights;
    // Predicted-descriptor projection (d x descriptor_dim).
    Eigen::MatrixXd descriptor_weights;
    // Hidden layer weights (H x d).
    Eigen::MatrixXd hidden_weights;
    // Hidden layer biases (H).
    Eigen::VectorXd hidden_biases;
    // Vocabulary projection weights (V x H).
    Eigen::MatrixXd output_weights;
    // Vocabulary projection biases (V).
    Eigen::VectorXd output_biases;
    // Latent bottleneck Z (M x d), one distinct row per latent through the keys and values.
    Eigen::MatrixXd latents;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\decoder_sequence_cache.hpp ===
#pragma once

#include <Eigen/Core>

#include <vector>

namespace forward_analysis {

// Intermediate values of one teacher-forced decoder pass over a field sequence, kept for backprop.
// T = steps, M = latents, d = d_model, H = hidden width, V = vocabulary size.
struct DecoderSequenceCache {
    // Token fed in at each step (T); BOS at step 0.
    std::vector<int> previous_tokens;
    // Token two positions back at each step (T); PAD where none exists.
    std::vector<int> second_previous_tokens;
    // Inverted dropout mask applied to the previous-token embedding rows (T x d); all ones in eval mode.
    Eigen::MatrixXd embedding_mask;
    // Inverted dropout mask applied to the second-previous-token embedding rows (T x d).
    Eigen::MatrixXd second_embedding_mask;
    // Inverted dropout mask applied to the tanh hidden layer (T x H).
    Eigen::MatrixXd hidden_mask;
    // Layer-normalized token/field/position sum used as attention queries' input (T x d).
    Eigen::MatrixXd query_inputs;
    // Per-row 1/std of that layer norm (T).
    Eigen::VectorXd query_inverse_std;
    // Attention queries Q = LN(u) W_Q (T x d).
    Eigen::MatrixXd queries;
    // Attention keys K = Z W_K (M x d).
    Eigen::MatrixXd keys;
    // Attention values V = Z W_V (M x d).
    Eigen::MatrixXd values;
    // Attention weights softmax(Q K^T / sqrt(d)) over the latents (T x M); every row sums to 1.
    Eigen::MatrixXd attention;
    // Layer-normalized step inputs LN(u + c + W_aux d_hat) (T x d).
    Eigen::MatrixXd normalized_inputs;
    // Per-row 1/std used by that layer norm (T).
    Eigen::VectorXd inverse_std;
    // tanh hidden activations before dropout (T x H).
    Eigen::MatrixXd hidden;
    // Softmax next-token distribution at each step (T x V).
    Eigen::MatrixXd probabilities;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\decoder_tensor_refs.hpp ===
#pragma once

#include "forward_analysis/math/adam_moment.hpp"

#include <Eigen/Core>

namespace forward_analysis {

// Addresses of every decoder parameter tensor and its Adam state, used to copy them to and from a device.
struct DecoderTensorRefs {
    // Previous-token embedding (vocab x d_model) and its Adam state.
    Eigen::MatrixXd* token_embeddings = nullptr;
    AdamMoment* token_moment = nullptr;
    // Two-back token embedding (vocab x d_model) and its Adam state.
    Eigen::MatrixXd* second_token_embeddings = nullptr;
    AdamMoment* token2_moment = nullptr;
    // Field embedding (4 x d_model) and its Adam state.
    Eigen::MatrixXd* field_embeddings = nullptr;
    AdamMoment* field_moment = nullptr;
    // Fixed sinusoidal step vectors (max_field_tokens x d_model); never trained.
    Eigen::MatrixXd* positions = nullptr;
    // Attention projections W_Q, W_K, W_V (d_model x d_model) and their Adam state.
    Eigen::MatrixXd* query_weights = nullptr;
    AdamMoment* query_moment = nullptr;
    Eigen::MatrixXd* key_weights = nullptr;
    AdamMoment* key_moment = nullptr;
    Eigen::MatrixXd* value_weights = nullptr;
    AdamMoment* value_moment = nullptr;
    // Descriptor projection W_aux (d_model x 14) and its Adam state.
    Eigen::MatrixXd* descriptor_weights = nullptr;
    AdamMoment* descriptor_moment = nullptr;
    // Hidden layer W1 (hidden x d_model), b1 (hidden) and their Adam state.
    Eigen::MatrixXd* hidden_weights = nullptr;
    AdamMoment* hidden_weights_moment = nullptr;
    Eigen::VectorXd* hidden_biases = nullptr;
    AdamMoment* hidden_biases_moment = nullptr;
    // Vocabulary projection W_out (vocab x hidden), b_out (vocab) and their Adam state.
    Eigen::MatrixXd* output_weights = nullptr;
    AdamMoment* output_weights_moment = nullptr;
    Eigen::VectorXd* output_biases = nullptr;
    AdamMoment* output_biases_moment = nullptr;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\field_delexicalizer.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/slot_binding.hpp"
#include "forward_analysis/decoder/batch_statement_delexicalizer.hpp"
#include "forward_analysis/decoder/observation_delexicalizer.hpp"
#include "forward_analysis/decoder/shape_slot_lexicon.hpp"

#include <string>
#include <vector>

namespace forward_analysis {

// Slot-token lexemes of one training field under the selected SlotBinding.
struct FieldDelexicalizer {
    static std::vector<std::string> delexicalize(const std::string& text, const NetworkObservations& measured, SlotBinding binding) {
        return binding == SlotBinding::kBatchStatistics
                   ? BatchStatementDelexicalizer::delexicalize(text)
                   : ObservationDelexicalizer::delexicalize(text, ShapeSlotLexicon::from(measured));
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\observation_delexicalizer.hpp ===
#pragma once

#include "forward_analysis/decoder/shape_slot_lexicon.hpp"
#include "forward_analysis/description/number_words.hpp"
#include "forward_analysis/text/english_list_formatter.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace forward_analysis {

// Replaces shape and index references in observation text with slot tokens bound to measured quantities,
// and fills slot tokens back in from another network's measurements:
//   "Seven of the twenty-four first-layer units (units 1, 5, 9, 12, 16, 20 and 23) never fire"
//   -> "<COUNT:DEAD_L1|W> of the <DIM:W_1|w> <ORD:LAYER_1|wl> units ( units <LIST:DEAD_L1> ) never fire"
// Forms: |w lowercase word, |W capitalized word, |d digits, |wl / |Wl ordinal + "-layer".
struct ObservationDelexicalizer {
    static bool is_slot(const std::string& lexeme) {
        return lexeme.size() > 2 && lexeme.front() == '<' && lexeme.back() == '>' && lexeme.find(':') != std::string::npos;
    }

    static std::vector<std::string> delexicalize(const std::string& text, const ShapeSlotLexicon& lex) {
        const std::vector<std::string> words = LexemeSplitter::split(text);
        std::vector<std::string> lower;
        for (const std::string& w : words) {
            lower.push_back(LexemeSplitter::lowercase(w));
        }
        std::vector<std::string> out;
        for (std::size_t i = 0; i < words.size();) {
            std::size_t consumed = 0;
            const std::string list = list_slot(words, i, lex, consumed);
            if (!list.empty()) {
                out.push_back(list);
                i += consumed;
                continue;
            }
            out.push_back(single_slot(words, lower, i, lex));
            ++i;
        }
        return out;
    }

    // Fills every slot from lex; false when a slot names a quantity this network does not have.
    static bool relexicalize(const std::vector<std::string>& lexemes, const ShapeSlotLexicon& lex, std::string& text) {
        std::vector<std::string> filled;
        for (const std::string& lexeme : lexemes) {
            if (!is_slot(lexeme)) {
                filled.push_back(lexeme);
                continue;
            }
            std::string value;
            if (!fill(lexeme, lex, value)) {
                return false;
            }
            filled.push_back(value);
        }
        text = LexemeSplitter::join(filled);
        return true;
    }

private:
    static const std::array<std::string, 4>& ordinals() {
        static const std::array<std::string, 4> words = {"first", "second", "third", "fourth"};
        return words;
    }

    static bool capitalized(const std::string& word) {
        return !word.empty() && std::isupper(static_cast<unsigned char>(word[0])) != 0;
    }

    static bool digits(const std::string& word) {
        if (word.empty()) {
            return false;
        }
        for (char c : word) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return true;
    }

    static std::string form_of(const std::string& word) {
        return digits(word) ? "d" : (capitalized(word) ? "W" : "w");
    }

    static std::string slot(const std::string& kind, const std::string& name, const std::string& form = "") {
        return "<" + kind + ":" + name + (form.empty() ? "" : "|" + form) + ">";
    }

    // A run of indices written exactly as EnglishListFormatter writes one of the measured lists.
    static std::string list_slot(const std::vector<std::string>& words,
                                 std::size_t i,
                                 const ShapeSlotLexicon& lex,
                                 std::size_t& consumed) {
        if (!digits(words[i])) {
            return "";
        }
        static const std::set<std::string> single_cues = {"unit", "units", "sample", "samples"};
        const bool cued = i > 0 && single_cues.count(LexemeSplitter::lowercase(words[i - 1])) > 0;
        std::string best;
        for (const auto& [name, values] : lex.lists) {
            if (values.empty() || (values.size() == 1 && !cued)) {
                continue;
            }
            const std::vector<std::string> canonical = LexemeSplitter::split(EnglishListFormatter::indices(values));
            if (i + canonical.size() > words.size() || canonical.size() <= consumed) {
                continue;
            }
            bool match = true;
            for (std::size_t k = 0; k < canonical.size() && match; ++k) {
                match = words[i + k] == canonical[k];
            }
            const bool continues = i + canonical.size() < words.size() &&
                                   (words[i + canonical.size()] == "," || words[i + canonical.size()] == "and") &&
                                   i + canonical.size() + 1 < words.size() && digits(words[i + canonical.size() + 1]);
            if (match && !continues) {
                consumed = canonical.size();
                best = slot("LIST", name);
            }
        }
        return best;
    }

    static std::string noun_group(const std::vector<std::string>& lower, std::size_t i) {
        static const std::set<std::string> units = {"units", "unit", "first-layer", "second-layer", "hidden", "neurons"};
        static const std::set<std::string> samples = {"samples", "sample"};
        static const std::set<std::string> outputs = {"outputs"};
        static const std::set<std::string> coordinates = {"coordinates", "inputs"};
        for (std::size_t k = i + 1; k <= i + 2 && k < lower.size(); ++k) {
            if (k == i + 2 && (lower[i + 1] == "of" || LexemeSplitter::is_punctuation(lower[i + 1][0]))) {
                break;
            }
            const std::string& w = lower[k];
            if (units.count(w)) return "units";
            if (samples.count(w)) return "samples";
            if (outputs.count(w)) return "outputs";
            if (coordinates.count(w)) return "coordinates";
        }
        return "";
    }

    static std::vector<std::string> group_names(const std::string& group, std::size_t layer_hint, std::size_t layers) {
        std::vector<std::string> names;
        if (group == "units") {
            for (std::size_t pass = 0; pass < 2; ++pass) {
                for (std::size_t l = 1; l <= layers; ++l) {
                    if ((pass == 0) != (layer_hint == 0 || l == layer_hint)) {
                        continue;
                    }
                    const std::string s = std::to_string(l);
                    for (const char* base : {"W_", "DEAD_L", "LIVE_L", "ACTIVE_L", "PARTIAL_L"}) {
                        names.push_back(base + s);
                    }
                }
            }
        } else if (group == "samples") {
            names = {"N", "CLOSED", "OPEN"};
        } else if (group == "outputs") {
            names = {"D_OUT"};
        } else if (group == "coordinates") {
            names = {"D_IN", "DRIVERS", "OTHER_INPUTS", "D_IN_MINUS_1"};
        }
        return names;
    }

    static std::string match_count(std::size_t value, const std::vector<std::string>& names, const ShapeSlotLexicon& lex) {
        for (const std::string& name : names) {
            const auto it = lex.counts.find(name);
            if (it != lex.counts.end() && it->second == value) {
                return name;
            }
        }
        return "";
    }

    // Parts that can be stated "of the <whole>" for each whole quantity.
    static std::vector<std::string> parts_of(const std::string& whole) {
        if (whole.rfind("W_", 0) == 0) {
            const std::string s = whole.substr(2);
            return {"DEAD_L" + s, "LIVE_L" + s, "ACTIVE_L" + s, "PARTIAL_L" + s};
        }
        if (whole == "N") return {"CLOSED", "OPEN"};
        if (whole == "D_IN") return {"DRIVERS", "OTHER_INPUTS", "D_IN_MINUS_1"};
        return {};
    }

    static std::string whole_name(const std::vector<std::string>& lower, std::size_t j, const ShapeSlotLexicon& lex) {
        std::size_t value = 0;
        if (!NumberWords::parse(lower[j], value)) {
            return "";
        }
        std::string group = noun_group(lower, j);
        std::vector<std::string> names = group.empty() ? std::vector<std::string>{} : group_names(group, 0, lex.layer_count);
        if (names.empty()) {
            names = group_names("units", 0, lex.layer_count);
            names.insert(names.end(), {"N", "D_IN"});
        }
        std::vector<std::string> wholes;
        for (const std::string& n : names) {
            if (n.rfind("W_", 0) == 0 || n == "N" || n == "D_IN" || n == "D_OUT") {
                wholes.push_back(n);
            }
        }
        return match_count(value, wholes, lex);
    }

    static std::string single_slot(const std::vector<std::string>& words,
                                   const std::vector<std::string>& lower,
                                   std::size_t i,
                                   const ShapeSlotLexicon& lex) {
        const std::string& word = words[i];
        const std::string& w = lower[i];

        if (w.size() > 1 && w[0] == 'x' && digits(w.substr(1))) {
            const std::string name = lex.coordinate_name(static_cast<std::size_t>(std::stoul(w.substr(1))));
            return name.empty() ? word : slot("COORD", name);
        }

        for (std::size_t k = 0; k < ordinals().size(); ++k) {
            const std::string& ord = ordinals()[k];
            const std::string next = i + 1 < lower.size() ? lower[i + 1] : "";
            const std::string name = "LAYER_" + std::to_string(k + 1);
            if (w == ord + "-layer") {
                return slot("ORD", name, capitalized(word) ? "Wl" : "wl");
            }
            if (w == ord && (next == "layer" || next == "layer's" || next == "layers")) {
                return slot("ORD", name, capitalized(word) ? "W" : "w");
            }
        }

        std::size_t value = 0;
        if (!NumberWords::parse(w, value) || value < 2) {
            return word;
        }
        if (i + 3 < lower.size() && lower[i + 1] == "of" && (lower[i + 2] == "the" || lower[i + 2] == "its")) {
            const std::string whole = whole_name(lower, i + 3, lex);
            if (!whole.empty()) {
                const std::string part = match_count(value, parts_of(whole), lex);
                if (!part.empty()) {
                    return slot("COUNT", part, form_of(word));
                }
            }
        }
        if (i >= 3 && lower[i - 2] == "of" && (lower[i - 1] == "the" || lower[i - 1] == "its")) {
            const std::string whole = whole_name(lower, i, lex);
            if (!whole.empty()) {
                return slot("DIM", whole, form_of(word));
            }
        }
        const std::string group = noun_group(lower, i);
        if (group.empty()) {
            return word;
        }
        const std::string name = match_count(value, group_names(group, 0, lex.layer_count), lex);
        if (name.empty()) {
            return word;
        }
        const bool dimension = name.rfind("W_", 0) == 0 || name == "N" || name == "D_IN" || name == "D_OUT";
        return slot(dimension ? "DIM" : "COUNT", name, form_of(word));
    }

    static bool fill(const std::string& token, const ShapeSlotLexicon& lex, std::string& value) {
        const std::string body = token.substr(1, token.size() - 2);
        const std::size_t colon = body.find(':');
        const std::size_t bar = body.find('|');
        const std::string kind = body.substr(0, colon);
        const std::string name = body.substr(colon + 1, bar == std::string::npos ? std::string::npos : bar - colon - 1);
        const std::string form = bar == std::string::npos ? "" : body.substr(bar + 1);

        if (kind == "COORD") {
            const auto it = lex.coordinates.find(name);
            if (it == lex.coordinates.end()) {
                return false;
            }
            value = "x" + std::to_string(it->second);
            return true;
        }
        if (kind == "LIST") {
            const auto it = lex.lists.find(name);
            if (it == lex.lists.end() || it->second.empty()) {
                return false;
            }
            value = EnglishListFormatter::indices(it->second);
            return true;
        }
        if (kind == "ORD") {
            const std::size_t k = static_cast<std::size_t>(std::stoul(name.substr(name.find('_') + 1)));
            if (k == 0 || k > lex.layer_count || k > ordinals().size()) {
                return false;
            }
            value = ordinals()[k - 1];
            if (form == "W" || form == "Wl") {
                value[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(value[0])));
            }
            if (form == "wl" || form == "Wl") {
                value += "-layer";
            }
            return true;
        }
        const auto it = lex.counts.find(name);
        if (it == lex.counts.end()) {
            return false;
        }
        value = form == "d" ? std::to_string(it->second)
                            : (form == "W" ? NumberWords::capitalized(it->second) : NumberWords::word(it->second));
        return true;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\observation_tokenizer.hpp ===
#pragma once

#include "forward_analysis/text/lexeme_splitter.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace forward_analysis {

// Word-level vocabulary for observation text: each word and each punctuation mark is one token.
class ObservationTokenizer {
public:
    static constexpr int kPadToken = 0;
    static constexpr int kBosToken = 1;
    static constexpr int kEosToken = 2;
    static constexpr int kUnkToken = 3;

    ObservationTokenizer() {
        for (const char* special : {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}) {
            register_token(special);
        }
    }

    int register_token(const std::string& token) {
        const auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        const int id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_.emplace(token, id);
        return id;
    }

    // BOS, word ids, EOS; at most max_tokens ids, always ending with EOS.
    std::vector<int> encode(const std::string& text, std::size_t max_tokens, bool register_new) {
        return encode_lexemes(LexemeSplitter::split(text), max_tokens, register_new);
    }

    // Like encode, but unseen words map to UNK and the vocabulary is left unchanged.
    std::vector<int> encode_known(const std::string& text, std::size_t max_tokens) const {
        return encode_known_lexemes(LexemeSplitter::split(text), max_tokens);
    }

    // encode over an already split lexeme sequence (slot tokens such as "<DIM:W_1|w>" stay whole).
    std::vector<int> encode_lexemes(const std::vector<std::string>& lexemes, std::size_t max_tokens, bool register_new) {
        std::vector<int> ids{kBosToken};
        for (const std::string& lex : lexemes) {
            if (ids.size() + 1 >= std::max<std::size_t>(2, max_tokens)) {
                break;
            }
            const auto it = token_to_id_.find(lex);
            ids.push_back(it != token_to_id_.end() ? it->second : (register_new ? register_token(lex) : kUnkToken));
        }
        ids.push_back(kEosToken);
        return ids;
    }

    std::vector<int> encode_known_lexemes(const std::vector<std::string>& lexemes, std::size_t max_tokens) const {
        std::vector<int> ids{kBosToken};
        for (const std::string& lex : lexemes) {
            if (ids.size() + 1 >= std::max<std::size_t>(2, max_tokens)) {
                break;
            }
            const auto it = token_to_id_.find(lex);
            ids.push_back(it != token_to_id_.end() ? it->second : kUnkToken);
        }
        ids.push_back(kEosToken);
        return ids;
    }

    // Token strings up to the first EOS, skipping PAD/BOS and out-of-range ids.
    std::vector<std::string> decode_lexemes(const std::vector<int>& ids) const {
        std::vector<std::string> lexemes;
        for (int id : ids) {
            if (id == kEosToken) {
                break;
            }
            if (id == kPadToken || id == kBosToken || id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
                continue;
            }
            lexemes.push_back(id_to_token_[static_cast<std::size_t>(id)]);
        }
        return lexemes;
    }

    std::string decode(const std::vector<int>& ids) const {
        return LexemeSplitter::join(decode_lexemes(ids));
    }

    std::size_t vocab_size() const noexcept { return id_to_token_.size(); }

    const std::string& token(int id) const { return id_to_token_.at(static_cast<std::size_t>(id)); }

private:
    // Token string for each id; index = token id.
    std::vector<std::string> id_to_token_;
    // Reverse lookup from token string to id.
    std::unordered_map<std::string, int> token_to_id_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\decoder\shape_slot_lexicon.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace forward_analysis {

// Named, measured quantities of one network that slot tokens refer to. Slot names are shared across
// networks (W_1 is always the first hidden width); the values come from that network's telemetry.
struct ShapeSlotLexicon {
    // Scalar sizes and counts, e.g. {"W_1": 24, "DEAD_L1": 5, "N": 8, "D_IN": 16, "CLOSED": 3}, plus the batch
    // statistics ZERO_Ll, NONPOS_Ll, RANK_Ll, STRONG_k, WEAK_k (input indices), SMALLER_RANGE, LARGER_RANGE,
    // COMPARABLE_RANGE, SIGN_AGREE and SIGN_ENTRIES.
    std::map<std::string, std::size_t> counts;
    // Index lists, e.g. {"DEAD_L1": [3, 8, 14, 19, 22], "CLOSED": [1, 4, 5]}.
    std::map<std::string, std::vector<std::size_t>> lists;
    // Named input coordinates: GATE, RISE_k / FALL_k (leading-axis drivers by direction), SEC_k (secondary-axis
    // drivers), TGT_k (target-related coordinates), each k counted from 1 in strength order.
    std::map<std::string, std::size_t> coordinates;
    // Number of hidden layers (ORD:LAYER_k slots exist for k <= layer_count).
    std::size_t layer_count = 0;

    static ShapeSlotLexicon from(const NetworkObservations& obs) {
        ShapeSlotLexicon lex;
        lex.layer_count = obs.layers.size();
        lex.counts["N"] = obs.shape.num_samples;
        lex.counts["D_IN"] = obs.shape.input_dim;
        lex.counts["D_OUT"] = obs.shape.output_dim;
        lex.counts["D_IN_MINUS_1"] = obs.shape.input_dim - 1;
        lex.counts["DRIVERS"] = obs.drivers.size();
        lex.counts["OTHER_INPUTS"] = obs.shape.input_dim - std::min(obs.drivers.size(), obs.shape.input_dim);
        for (std::size_t l = 0; l < obs.layers.size(); ++l) {
            const LayerObservation& layer = obs.layers[l];
            const std::string suffix = "_L" + std::to_string(l + 1);
            std::vector<std::size_t> partial;
            for (std::size_t u = 0; u < layer.width; ++u) {
                const bool dead = std::find(layer.dead_units.begin(), layer.dead_units.end(), u) != layer.dead_units.end();
                const bool always = std::find(layer.always_active_units.begin(), layer.always_active_units.end(), u) !=
                                    layer.always_active_units.end();
                if (!dead && !always) {
                    partial.push_back(u);
                }
            }
            lex.counts["W_" + std::to_string(l + 1)] = layer.width;
            lex.counts["DEAD" + suffix] = layer.dead_units.size();
            lex.counts["LIVE" + suffix] = layer.live_count();
            lex.counts["ACTIVE" + suffix] = layer.always_active_units.size();
            lex.counts["PARTIAL" + suffix] = partial.size();
            lex.lists["DEAD" + suffix] = layer.dead_units;
            lex.lists["ACTIVE" + suffix] = layer.always_active_units;
            lex.lists["PARTIAL" + suffix] = partial;
            lex.counts["ZERO" + suffix] = layer.dead_units.size();
            lex.counts["NONPOS" + suffix] = layer.nonpositive_samples;
            lex.counts["RANK" + suffix] = layer.numerical_rank;
        }
        for (std::size_t k = 0; k < obs.strongest_inputs.size(); ++k) {
            lex.counts["STRONG_" + std::to_string(k + 1)] = obs.strongest_inputs[k];
        }
        for (std::size_t k = 0; k < obs.weakest_inputs.size(); ++k) {
            lex.counts["WEAK_" + std::to_string(k + 1)] = obs.weakest_inputs[k];
        }
        lex.counts["SMALLER_RANGE"] = obs.outputs_with_smaller_range;
        lex.counts["LARGER_RANGE"] = obs.outputs_with_larger_range;
        lex.counts["COMPARABLE_RANGE"] = obs.outputs_with_comparable_range;
        lex.counts["SIGN_AGREE"] = obs.sign_agreements;
        lex.counts["SIGN_ENTRIES"] = obs.sign_entries;
        if (obs.gating.present()) {
            lex.counts["CLOSED"] = obs.gating.closed_samples.size();
            lex.counts["OPEN"] = obs.gating.open_samples.size();
            lex.lists["CLOSED"] = obs.gating.closed_samples;
            lex.lists["OPEN"] = obs.gating.open_samples;
        }
        if (obs.gating.has_gate_coordinate) {
            lex.coordinates["GATE"] = obs.gating.gate_coordinate;
        }
        std::size_t rises = 0;
        std::size_t falls = 0;
        std::size_t secondary = 0;
        for (const InputDriver& d : obs.drivers) {
            if (d.axis == 1) {
                lex.coordinates["SEC_" + std::to_string(++secondary)] = d.coordinate;
            } else if (d.direction > 0) {
                lex.coordinates["RISE_" + std::to_string(++rises)] = d.coordinate;
            } else {
                lex.coordinates["FALL_" + std::to_string(++falls)] = d.coordinate;
            }
        }
        for (std::size_t k = 0; k < obs.target_related_coordinates.size(); ++k) {
            lex.coordinates["TGT_" + std::to_string(k + 1)] = obs.target_related_coordinates[k];
        }
        return lex;
    }

    // Name of the coordinate slot holding input index c, or "" when c has no measured role.
    std::string coordinate_name(std::size_t c) const {
        if (coordinates.count("GATE") > 0 && coordinates.at("GATE") == c) {
            return "GATE";
        }
        for (const auto& [name, value] : coordinates) {
            if (value == c) {
                return name;
            }
        }
        return "";
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\batch_statement_checker.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Verifies the batch-statistic statements of the observation text against measured facts, wherever they occur:
//   "across N observations"                                   N = batch size
//   "hidden widths are a and b"                               hidden widths in order
//   "layer L has Z coordinates that remain zero"              Z = always-zero units of layer L
//   "K observations with no positive coordinates"             K = samples with no positive unit in the layer named last
//   "input coordinates a and b have the strongest"            the two strongest-associated inputs, strongest first
//   "coordinates a and b have the weakest"                    the two weakest-associated inputs, weakest first
//   "numerical ranks r1, r2"                                  numerical rank of every layer in order
//   "narrow observed representation" / "multiple directions"  every layer rank <= narrow_rank_limit / some layer above
//   "over K output coordinates"                               K = output width
//   "a have a smaller prediction range", "b have a larger range", "c have broadly comparable ranges"
//   "agree in A of E entries"                                 A = sign agreements, E = prediction entries
// Every number lexeme it verifies is marked in `verified` so other checks do not reinterpret it.
struct BatchStatementChecker {
    // lex: lowercase lexemes of one field. Returns the first contradicted statement, or "" when all hold.
    static std::string verify(const std::vector<std::string>& lex, const NetworkObservations& obs, std::vector<bool>& verified) {
        verified.assign(lex.size(), false);
        std::size_t layer_context = 0;
        for (std::size_t i = 0; i < lex.size(); ++i) {
            std::vector<std::size_t> v;
            std::vector<std::size_t> at;
            if (match(lex, i, {"across", "#", "observations"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.shape.num_samples) {
                    return "\"across " + lex[at[0]] + " observations\" but the batch has " + std::to_string(obs.shape.num_samples);
                }
            } else if (match(lex, i, {"hidden", "widths", "are", "#*"}, v, at)) {
                mark(verified, at);
                if (v != obs.shape.hidden_widths) {
                    return "stated hidden widths differ from the telemetry";
                }
            } else if (match(lex, i, {"layer", "#", "has", "#", "coordinates", "that", "remain", "zero"}, v, at)) {
                mark(verified, at);
                if (v[0] == 0 || v[0] > obs.layers.size()) {
                    return "\"layer " + lex[at[0]] + "\" but the network has " + std::to_string(obs.layers.size()) + " hidden layers";
                }
                layer_context = v[0];
                if (v[1] != obs.layers[v[0] - 1].dead_units.size()) {
                    return "layer " + lex[at[0]] + " has " + std::to_string(obs.layers[v[0] - 1].dead_units.size()) +
                           " always-zero coordinates, not " + lex[at[1]];
                }
            } else if (match(lex, i, {"#", "observations", "with", "no", "positive", "coordinates"}, v, at)) {
                mark(verified, at);
                if (layer_context == 0) {
                    return "\"" + lex[at[0]] + " observations with no positive coordinates\" names no layer";
                }
                if (v[0] != obs.layers[layer_context - 1].nonpositive_samples) {
                    return "layer " + std::to_string(layer_context) + " has " +
                           std::to_string(obs.layers[layer_context - 1].nonpositive_samples) +
                           " observations with no positive coordinate, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"input", "coordinates", "#", "and", "#", "have", "the", "strongest"}, v, at)) {
                mark(verified, at);
                if (obs.strongest_association <= 0.0) {
                    return "\"strongest\" associations but no input correlates with the final hidden layer";
                }
                if (v != obs.strongest_inputs) {
                    return "strongest-associated inputs are " + pair_text(obs.strongest_inputs) + ", not " + pair_text(v);
                }
            } else if (match(lex, i, {"coordinates", "#", "and", "#", "have", "the", "weakest"}, v, at)) {
                mark(verified, at);
                if (obs.strongest_association <= 0.0) {
                    return "\"weakest\" associations but no input correlates with the final hidden layer";
                }
                if (v != obs.weakest_inputs) {
                    return "weakest-associated inputs are " + pair_text(obs.weakest_inputs) + ", not " + pair_text(v);
                }
            } else if (match(lex, i, {"numerical", "ranks", "#*"}, v, at)) {
                mark(verified, at);
                std::vector<std::size_t> ranks;
                for (const LayerObservation& layer : obs.layers) {
                    ranks.push_back(layer.numerical_rank);
                }
                if (v != ranks) {
                    return "stated numerical ranks differ from the measured ranks";
                }
            } else if (match(lex, i, {"narrow", "observed", "representation"}, v, at) && !obs.narrow_representation) {
                return "\"narrow observed representation\" but a layer's numerical rank exceeds the narrow limit";
            } else if (match(lex, i, {"multiple", "directions"}, v, at) && obs.narrow_representation) {
                return "\"multiple directions\" but every layer's numerical rank is within the narrow limit";
            } else if (match(lex, i, {"over", "#", "output", "coordinates"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.shape.output_dim) {
                    return "\"over " + lex[at[0]] + " output coordinates\" but the network has " + std::to_string(obs.shape.output_dim);
                }
            } else if (match(lex, i, {"#", "have", "a", "smaller", "prediction", "range"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.outputs_with_smaller_range) {
                    return std::to_string(obs.outputs_with_smaller_range) + " outputs have a smaller range, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"#", "have", "a", "larger", "range"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.outputs_with_larger_range) {
                    return std::to_string(obs.outputs_with_larger_range) + " outputs have a larger range, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"#", "have", "broadly", "comparable", "ranges"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.outputs_with_comparable_range) {
                    return std::to_string(obs.outputs_with_comparable_range) + " outputs have a comparable range, not " + lex[at[0]];
                }
            } else if (match(lex, i, {"agree", "in", "#", "of", "#", "entries"}, v, at)) {
                mark(verified, at);
                if (v[0] != obs.sign_agreements || v[1] != obs.sign_entries) {
                    return "signs agree in " + std::to_string(obs.sign_agreements) + " of " + std::to_string(obs.sign_entries) +
                           " entries, not " + lex[at[0]] + " of " + lex[at[1]];
                }
            }
        }
        return "";
    }

    // True for a lexeme made only of decimal digits (at most nine).
    static bool number(const std::string& token) {
        if (token.empty()) {
            return false;
        }
        for (char c : token) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return token.size() < 10;
    }

    // Matches pattern at lex[start]: "#" is one number, "#*" a list of numbers joined by "," or "and"; values and
    // positions of every matched number are returned in order.
    static bool match(const std::vector<std::string>& lex,
                      std::size_t start,
                      const std::vector<std::string>& pattern,
                      std::vector<std::size_t>& values,
                      std::vector<std::size_t>& positions) {
        values.clear();
        positions.clear();
        std::size_t i = start;
        for (const std::string& p : pattern) {
            if (p == "#" || p == "#*") {
                if (i >= lex.size() || !number(lex[i])) {
                    return false;
                }
                values.push_back(static_cast<std::size_t>(std::stoul(lex[i])));
                positions.push_back(i++);
                while (p == "#*" && i + 1 < lex.size() && (lex[i] == "," || lex[i] == "and") && number(lex[i + 1])) {
                    values.push_back(static_cast<std::size_t>(std::stoul(lex[i + 1])));
                    positions.push_back(i + 1);
                    i += 2;
                }
            } else if (i < lex.size() && lex[i] == p) {
                ++i;
            } else {
                return false;
            }
        }
        return true;
    }

private:
    static void mark(std::vector<bool>& verified, const std::vector<std::size_t>& positions) {
        for (std::size_t p : positions) {
            verified[p] = true;
        }
    }

    static std::string pair_text(const std::vector<std::size_t>& v) {
        std::string out;
        for (std::size_t k = 0; k < v.size(); ++k) {
            out += (k == 0 ? "" : " and ") + std::to_string(v[k]);
        }
        return out;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\batch_statistics_narrator.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/description/observation_text.hpp"

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

// Writes the four observation fields as batch statistics measured from the telemetry, in the phrasing of the
// corrected dataset (manifest version corrected-2026-10-14): always-zero coordinates and observations with no
// positive coordinate per layer, strongest / weakest input-to-final-hidden sample correlations, numerical ranks,
// and prediction-range and sign agreement against the targets.
struct BatchStatisticsNarrator {
    static ObservationText describe(const NetworkObservations& obs, const DistillerConfig& config = DistillerConfig{}) {
        ObservationText text;
        text.pathway_activity = pathway_activity(obs);
        text.feature_sensitivity = feature_sensitivity(obs);
        text.representational_structure = representational_structure(obs);
        text.functional_mapping = functional_mapping(obs, config);
        return text;
    }

private:
    // "27 and 14", "36", "8, 12 and 20".
    static std::string list(const std::vector<std::size_t>& values, const std::string& separator) {
        std::string out;
        for (std::size_t k = 0; k < values.size(); ++k) {
            if (k > 0) {
                out += (separator == "and" && k + 1 == values.size()) ? " and " : ", ";
            }
            out += std::to_string(values[k]);
        }
        return out;
    }

    static std::string pathway_activity(const NetworkObservations& obs) {
        std::string out = "Across " + std::to_string(obs.shape.num_samples) + " observations, the hidden widths are " +
                          list(obs.shape.hidden_widths, "and") + ".";
        for (std::size_t l = 0; l < obs.layers.size(); ++l) {
            out += " Layer " + std::to_string(l + 1) + " has " + std::to_string(obs.layers[l].dead_units.size()) +
                   " coordinates that remain zero throughout the batch and " +
                   std::to_string(obs.layers[l].nonpositive_samples) + " observations with no positive coordinates.";
        }
        return out;
    }

    static std::string feature_sensitivity(const NetworkObservations& obs) {
        if (obs.strongest_association <= 0.0 || obs.strongest_inputs.size() < 2 || obs.weakest_inputs.size() < 2) {
            return "Observed association only: no input coordinate has a nonzero sample correlation with any final hidden "
                   "coordinate. These associations do not establish causal feature influence.";
        }
        return "Observed association only: input coordinates " + list(obs.strongest_inputs, "and") +
               " have the strongest absolute sample correlations with at least one final hidden coordinate; coordinates " +
               list(obs.weakest_inputs, "and") +
               " have the weakest. These associations do not establish causal feature influence.";
    }

    static std::string representational_structure(const NetworkObservations& obs) {
        std::vector<std::size_t> ranks;
        for (const LayerObservation& layer : obs.layers) {
            ranks.push_back(layer.numerical_rank);
        }
        return "The centered activation matrices have observed numerical ranks " + list(ranks, ",") +
               " across successive hidden layers. " +
               (obs.narrow_representation
                    ? "This is a narrow observed representation relative to its hidden widths."
                    : "The observed activation variation occupies multiple directions in at least one layer.");
    }

    static std::string functional_mapping(const NetworkObservations& obs, const DistillerConfig& config) {
        std::ostringstream window;
        window << config.range_ratio_low << " to " << config.range_ratio_high;
        return "Comparing predictions against targets over " + std::to_string(obs.shape.output_dim) + " output coordinates, " +
               std::to_string(obs.outputs_with_smaller_range) + " have a smaller prediction range, " +
               std::to_string(obs.outputs_with_larger_range) + " have a larger range, and " +
               std::to_string(obs.outputs_with_comparable_range) +
               " have broadly comparable ranges using a range-ratio window from " + window.str() +
               ". Prediction and target signs agree in " + std::to_string(obs.sign_agreements) + " of " +
               std::to_string(obs.sign_entries) + " entries. These statements describe only the observed batch.";
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\measured_facts_narration.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/description/batch_statistics_narrator.hpp"
#include "forward_analysis/description/observation_narrator.hpp"
#include "forward_analysis/description/observation_text.hpp"

namespace forward_analysis {

// The narrator selected by config.narration_style.
struct MeasuredFactsNarration {
    static ObservationText describe(const NetworkObservations& obs, const DistillerConfig& config) {
        return config.narration_style == NarrationStyle::kBatchStatistics ? BatchStatisticsNarrator::describe(obs, config)
                                                                         : ObservationNarrator::describe(obs);
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\number_words.hpp ===
#pragma once

#include <array>
#include <cctype>
#include <cstddef>
#include <string>

namespace forward_analysis {

// English cardinal words for 0..99, both directions.
struct NumberWords {
    static std::string word(std::size_t n) {
        if (n < 20) {
            return units()[n];
        }
        if (n < 100) {
            const std::string ten = tens()[n / 10];
            return n % 10 == 0 ? ten : ten + "-" + units()[n % 10];
        }
        return std::to_string(n);
    }

    static std::string capitalized(std::size_t n) {
        std::string out = word(n);
        out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
        return out;
    }

    // Parses "twenty-four", "Eleven", "7"; returns false for anything else.
    static bool parse(const std::string& text, std::size_t& value) {
        std::string lower;
        for (char c : text) {
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (!lower.empty() && std::isdigit(static_cast<unsigned char>(lower[0]))) {
            for (char c : lower) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    return false;
                }
            }
            value = static_cast<std::size_t>(std::stoul(lower));
            return true;
        }
        for (std::size_t n = 0; n < 100; ++n) {
            if (word(n) == lower) {
                value = n;
                return true;
            }
        }
        return false;
    }

private:
    static const std::array<std::string, 20>& units() {
        static const std::array<std::string, 20> words = {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
        return words;
    }

    static const std::array<std::string, 10>& tens() {
        static const std::array<std::string, 10> words = {
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
        return words;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\observation_consistency.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/description/batch_statement_checker.hpp"
#include "forward_analysis/description/number_words.hpp"
#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace forward_analysis {

// Checks the verifiable claims in observation text against measured facts: named coordinates are
// measured drivers, the gate or target-related coordinates, unit indices exist, counts attached to units / samples / outputs and
// "<count> of the <total>" pairs are real sizes, statements about dead units, rays, sheets, deeper
// layers, gating, preserved signs and close / flat outputs hold for the telemetry, batch-statistic statements
// (BatchStatementChecker) match the measured counts, ranks, associations and ranges, and the text does not loop.
struct ObservationConsistency {
    static std::array<bool, ObservationText::kFieldCount> verify(const ObservationText& text,
                                                                 const NetworkObservations& obs) {
        std::array<bool, ObservationText::kFieldCount> out{};
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            out[f] = verify_field(text.field(f), obs);
        }
        return out;
    }

    static bool verify_field(const std::string& text, const NetworkObservations& obs) {
        return violation(text, obs).empty();
    }

    // The first claim in the text that the measured facts contradict; empty when every checked claim holds.
    static std::string violation(const std::string& text, const NetworkObservations& obs) {
        const std::vector<std::string> lex = LexemeSplitter::split(LexemeSplitter::lowercase(text));
        if (lex.empty()) {
            return "empty text";
        }
        std::vector<bool> verified;
        const std::string batch = BatchStatementChecker::verify(lex, obs, verified);
        if (!batch.empty()) {
            return batch;
        }
        for (const std::string& reason : {indices_exist(lex, obs, verified), coordinates_measured(lex, obs),
                                          count_pairs_match(lex, obs, verified), noun_counts_match(lex, obs, verified),
                                          claims_hold(lex, obs), repetition(lex)}) {
            if (!reason.empty()) {
                return reason;
            }
        }
        return "";
    }

private:
    static bool all_digits(const std::string& s, std::size_t from = 0) {
        if (s.size() <= from) {
            return false;
        }
        return std::all_of(s.begin() + static_cast<std::ptrdiff_t>(from), s.end(),
                           [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; });
    }

    static bool is_coordinate(const std::string& token) {
        return token.size() > 1 && token[0] == 'x' && all_digits(token, 1);
    }

    static const std::set<std::string>& count_nouns() {
        static const std::set<std::string> nouns = {"units", "samples", "outputs", "coordinates", "inputs",
                                                    "first-layer", "second-layer", "hidden", "observations"};
        return nouns;
    }

    // lex[i] is the integer or fractional part of a decimal number split at its point, e.g. "1" "." "825".
    static bool is_decimal_part(const std::vector<std::string>& lex, std::size_t i) {
        return (i + 1 < lex.size() && LexemeSplitter::is_decimal_point(lex, i + 1)) ||
               (i > 0 && LexemeSplitter::is_decimal_point(lex, i - 1));
    }

    // lex[i] directly precedes a count noun, or precedes one with a single modifier between ("25 input coordinates").
    static bool before_count_noun(const std::vector<std::string>& lex, std::size_t i) {
        return (i + 1 < lex.size() && count_nouns().count(lex[i + 1]) > 0) ||
               (i + 2 < lex.size() && lex[i + 1] != "of" && count_nouns().count(lex[i + 2]) > 0);
    }

    static std::string indices_exist(const std::vector<std::string>& lex,
                                     const NetworkObservations& obs,
                                     const std::vector<bool>& verified) {
        std::size_t largest = std::max(obs.shape.num_samples, obs.shape.output_dim);
        for (std::size_t w : obs.shape.hidden_widths) {
            largest = std::max(largest, w);
        }
        for (std::size_t i = 0; i < lex.size(); ++i) {
            const std::string& token = lex[i];
            if (is_coordinate(token) && std::stoul(token.substr(1)) >= obs.shape.input_dim) {
                return token + " exceeds input width " + std::to_string(obs.shape.input_dim);
            }
            if (all_digits(token) && !verified[i] && !is_decimal_part(lex, i) && !before_count_noun(lex, i) &&
                std::stoul(token) >= largest) {
                return "index " + token + " exceeds every layer, output and sample count";
            }
        }
        return "";
    }

    static std::string coordinates_measured(const std::vector<std::string>& lex, const NetworkObservations& obs) {
        std::set<std::size_t> known;
        if (obs.gating.has_gate_coordinate) {
            known.insert(obs.gating.gate_coordinate);
        }
        for (const InputDriver& d : obs.drivers) {
            known.insert(d.coordinate);
        }
        known.insert(obs.target_related_coordinates.begin(), obs.target_related_coordinates.end());
        for (const std::string& token : lex) {
            if (is_coordinate(token) && known.count(static_cast<std::size_t>(std::stoul(token.substr(1)))) == 0) {
                return token + " is not the gate coordinate, a measured driver or a target-related coordinate";
            }
        }
        return "";
    }

    static std::string count_pairs_match(const std::vector<std::string>& lex,
                                         const NetworkObservations& obs,
                                         const std::vector<bool>& verified) {
        std::set<std::pair<std::size_t, std::size_t>> allowed;
        for (const LayerObservation& layer : obs.layers) {
            allowed.insert({layer.dead_units.size(), layer.width});
            allowed.insert({layer.live_count(), layer.width});
            allowed.insert({layer.always_active_units.size(), layer.width});
        }
        allowed.insert({obs.gating.closed_samples.size(), obs.shape.num_samples});
        allowed.insert({obs.gating.open_samples.size(), obs.shape.num_samples});
        for (std::size_t i = 0; i + 3 < lex.size(); ++i) {
            std::size_t part = 0;
            std::size_t whole = 0;
            if (verified[i] || verified[i + 3] || lex[i + 1] != "of" || (lex[i + 2] != "the" && lex[i + 2] != "its") ||
                !NumberWords::parse(lex[i], part) || !NumberWords::parse(lex[i + 3], whole)) {
                continue;
            }
            if (i > 0 && (lex[i - 1] == "every" || lex[i - 1] == "each")) {
                continue;
            }
            if (allowed.count({part, whole}) == 0) {
                return "\"" + lex[i] + " of " + lex[i + 2] + " " + lex[i + 3] + "\" matches no layer or sample split";
            }
        }
        return "";
    }

    static std::string noun_counts_match(const std::vector<std::string>& lex,
                                         const NetworkObservations& obs,
                                         const std::vector<bool>& verified) {
        std::set<std::size_t> allowed = {1};
        for (const LayerObservation& layer : obs.layers) {
            allowed.insert({layer.width, layer.dead_units.size(), layer.live_count(), layer.always_active_units.size(),
                            layer.width - layer.always_active_units.size()});
        }
        const std::size_t drivers = obs.drivers.size();
        std::set<std::string> named;
        for (const std::string& token : lex) {
            if (is_coordinate(token)) {
                named.insert(token);
            }
        }
        allowed.insert({obs.shape.num_samples, obs.gating.closed_samples.size(), obs.gating.open_samples.size(),
                        obs.shape.output_dim, obs.shape.input_dim, drivers, obs.shape.input_dim - 1,
                        obs.shape.input_dim - std::min(drivers, obs.shape.input_dim),
                        obs.shape.input_dim - std::min(named.size(), obs.shape.input_dim)});
        for (std::size_t i = 0; i + 1 < lex.size(); ++i) {
            std::size_t value = 0;
            if (verified[i] || is_decimal_part(lex, i) || !NumberWords::parse(lex[i], value) || negated(lex, i)) {
                continue;
            }
            if (before_count_noun(lex, i) && allowed.count(value) == 0) {
                return "count \"" + lex[i] + "\" is not a size present in the telemetry";
            }
        }
        return "";
    }

    static bool negated(const std::vector<std::string>& lex, std::size_t i) {
        for (std::size_t back = 1; back <= 3 && back <= i; ++back) {
            const std::string& w = lex[i - back];
            if (w == "no" || w == "not" || w == "none" || w == "without") {
                return true;
            }
        }
        return false;
    }

    static std::string claims_hold(const std::vector<std::string>& lex, const NetworkObservations& obs) {
        bool any_dead = false;
        bool any_ray = false;
        for (const LayerObservation& layer : obs.layers) {
            any_dead = any_dead || !layer.dead_units.empty();
            any_ray = any_ray || layer.lies_on_ray;
        }
        bool any_sign_kept = false;
        bool any_close = false;
        bool any_flat = false;
        for (const OutputObservation& o : obs.outputs) {
            any_sign_kept = any_sign_kept || o.sign_preserved;
            any_close = any_close || std::max(o.correlation, o.open_side_correlation) >= 0.9;
            any_flat = any_flat || o.flat;
        }
        const bool multi_layer = obs.layers.size() >= 2;
        const bool planar = !obs.layers.empty() && !obs.layers.front().lies_on_ray &&
                            obs.layers.front().effective_rank >= 2;

        for (std::size_t i = 0; i < lex.size(); ++i) {
            if (negated(lex, i)) {
                continue;
            }
            const std::string& w = lex[i];
            const std::string next = i + 1 < lex.size() ? lex[i + 1] : "";
            const bool dead_claim =
                w == "dead" || w == "silent" || w == "dormant" ||
                (w == "never" && (next == "fire" || next == "fires" || next == "activate" || next == "leave"));
            if (dead_claim && !any_dead) {
                return "\"" + w + "\" but no unit is dead";
            }
            if (w == "ray" && !any_ray) {
                return "\"ray\" but no layer lies on a ray";
            }
            if ((w == "gate" || w == "gates" || w == "gated" || w == "half-space") && !obs.gating.present()) {
                return "\"" + w + "\" but no samples close the first layer";
            }
            if ((w == "keep" || w == "keeps") && next == "the" && i + 2 < lex.size() && lex[i + 2] == "sign" &&
                !any_sign_kept) {
                return "\"keeps the sign\" but no output preserves its target's sign";
            }
            const bool deeper_layer_claim = w == "second-layer" || w == "deeper" || w == "depth" ||
                                            (w == "second" && (next == "layer" || next == "layer's")) ||
                                            (w == "both" && next == "layers");
            if (deeper_layer_claim && !multi_layer) {
                return "\"" + w + (next.empty() ? "" : " " + next) + "\" but the network has one hidden layer";
            }
            if ((w == "two-dimensional" || w == "sheet" || w == "plane") && !planar) {
                return "\"" + w + "\" but the first layer has fewer than two effective dimensions";
            }
            if (w == "closely" && !any_close) {
                return "\"closely\" but no output correlates with its target at 0.9 or above";
            }
            if (w == "flat" && !any_flat) {
                return "\"flat\" but every prediction varies with at least 0.2 of its target's spread";
            }
            if (w == "the" && next == "prediction" && obs.shape.output_dim > 1) {
                return "\"the prediction\" but the network has " + std::to_string(obs.shape.output_dim) + " outputs";
            }
        }
        return "";
    }

    static std::string repetition(const std::vector<std::string>& lex) {
        std::map<std::string, std::size_t> counts;
        for (std::size_t i = 0; i + 4 <= lex.size(); ++i) {
            const std::string key = lex[i] + ' ' + lex[i + 1] + ' ' + lex[i + 2] + ' ' + lex[i + 3];
            if (++counts[key] >= 3) {
                return "\"" + key + "\" repeats three or more times";
            }
        }
        return "";
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\observation_narrator.hpp ===
#pragma once

#include "forward_analysis/analysis/network_observations.hpp"
#include "forward_analysis/description/number_words.hpp"
#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/text/english_list_formatter.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Writes the four observation fields from measured facts only; used wherever the decoder's text
// cannot be verified against the telemetry.
struct ObservationNarrator {
    static ObservationText describe(const NetworkObservations& obs) {
        ObservationText text;
        text.pathway_activity = pathway_activity(obs);
        text.feature_sensitivity = feature_sensitivity(obs);
        text.representational_structure = representational_structure(obs);
        text.functional_mapping = functional_mapping(obs);
        return text;
    }

    static std::string pathway_activity(const NetworkObservations& obs) {
        const GatingObservation& g = obs.gating;
        const std::string width = obs.layers.empty() ? "" : NumberWords::word(obs.layers.front().width);
        if (g.present()) {
            const std::string closed = NumberWords::word(g.closed_samples.size());
            const std::string open = NumberWords::word(g.open_samples.size());
            std::string out;
            if (g.has_gate_coordinate) {
                const std::string x = "x" + std::to_string(g.gate_coordinate);
                out = "The sign of " + x + " gates the first layer. On the " + closed + " samples where " + x +
                      " is negative every one of the " + width + " units returns exactly zero, while the " + open +
                      " samples with positive " + x + " open the layer.";
            } else {
                out = NumberWords::capitalized(g.closed_samples.size()) + " of the " +
                      NumberWords::word(obs.shape.num_samples) +
                      " samples leave every first-layer unit at exactly zero, while the remaining " + open +
                      " samples open the layer.";
            }
            if (g.deeper_layers_constant) {
                out += " In the deeper layers the closed samples share one identical activation pattern.";
            }
            return out;
        }

        std::string out;
        for (std::size_t l = 0; l < obs.layers.size(); ++l) {
            const LayerObservation& layer = obs.layers[l];
            const std::string name = layer_adjective(l);
            std::string sentence;
            if (!layer.dead_units.empty()) {
                sentence = NumberWords::capitalized(layer.dead_units.size()) + " of the " +
                           NumberWords::word(layer.width) + " " + name + " units";
                if (layer.dead_units.size() <= 12) {
                    sentence += " (units " + EnglishListFormatter::indices(layer.dead_units) + ")";
                }
                sentence += " output exactly zero on every sample, leaving " + NumberWords::word(layer.live_count()) +
                            " live units.";
            } else if (layer.always_active_units.size() == layer.width) {
                sentence = "Every one of the " + NumberWords::word(layer.width) + " " + name +
                           " units is active on every sample.";
            } else {
                sentence = "All " + NumberWords::word(layer.width) + " " + name +
                           " units fire on at least part of the batch.";
            }
            if (layer.lies_on_ray) {
                sentence += " The active units are fixed multiples of one per-sample intensity, so the layer carries "
                            "a single signal.";
            }
            out += (out.empty() ? "" : " ") + sentence;
        }
        return out;
    }

    static std::string feature_sensitivity(const NetworkObservations& obs) {
        const GatingObservation& g = obs.gating;
        std::vector<std::size_t> rises;
        std::vector<std::size_t> falls;
        std::vector<std::size_t> secondary;
        for (const InputDriver& d : obs.drivers) {
            if (g.has_gate_coordinate && d.coordinate == g.gate_coordinate) {
                continue;
            }
            if (d.axis == 1) {
                secondary.push_back(d.coordinate);
            } else {
                (d.direction > 0 ? rises : falls).push_back(d.coordinate);
            }
        }

        if (g.has_gate_coordinate) {
            std::string out = "x" + std::to_string(g.gate_coordinate) +
                              " acts as a switch on the whole pathway: its sign decides whether the first layer "
                              "responds at all.";
            std::vector<std::size_t> shaping = rises;
            shaping.insert(shaping.end(), falls.begin(), falls.end());
            if (!shaping.empty()) {
                out += " Within the positive half-space the response is shaped mostly by " +
                       EnglishListFormatter::coordinates(shaping) + ".";
            }
            return out;
        }
        if (rises.empty() && falls.empty()) {
            return "No single input coordinate shows a consistent relation to the hidden activity; sensitivity is "
                   "spread across many inputs.";
        }

        std::string out = "The leading hidden axis";
        if (!rises.empty()) {
            out += " rises with " + EnglishListFormatter::coordinates(rises);
        }
        if (!falls.empty()) {
            out += std::string(rises.empty() ? "" : " and") + " falls with " + EnglishListFormatter::coordinates(falls);
        }
        out += ".";
        if (!secondary.empty()) {
            out += " A secondary axis follows " + EnglishListFormatter::coordinates(secondary) + ".";
        }
        const bool ray = !obs.layers.empty() && obs.layers.front().lies_on_ray;
        out += ray ? " Any input change that leaves that projection unchanged leaves every hidden unit unchanged."
                   : " The remaining coordinates show no consistent relation to the hidden activity.";
        return out;
    }

    static std::string representational_structure(const NetworkObservations& obs) {
        if (obs.layers.empty()) {
            return "";
        }
        const LayerObservation& first = obs.layers.front();
        const GatingObservation& g = obs.gating;
        if (g.present()) {
            return std::string("The batch splits into two groups: the closed samples collapse onto the origin of the "
                               "first layer") +
                   (g.deeper_layers_constant ? " and onto one shared pattern deeper in the network" : "") +
                   ", while the open samples spread over " +
                   (first.effective_rank >= 2 ? "a multi-dimensional region." : "a single direction.");
        }
        if (first.lies_on_ray) {
            std::string out = "The first-layer activations lie on a single ray from the origin, and samples differ "
                              "only in their distance along it.";
            for (std::size_t l = 1; l < obs.layers.size(); ++l) {
                if (obs.layers[l].lies_on_ray) {
                    out += " The " + layer_ordinal(l) + " layer maps that ray onto another ray, adding no new "
                           "dimensions.";
                }
            }
            return out;
        }
        std::string out = "The first-layer activations occupy " + dimension_phrase(first.effective_rank);
        out += first.dead_units.empty() ? ", and the samples remain distinct from one another."
                                        : ", and the dead coordinates add no separation between samples.";
        return out;
    }

    static std::string functional_mapping(const NetworkObservations& obs) {
        if (obs.outputs.empty()) {
            return "";
        }
        if (obs.gating.present()) {
            bool pinned = true;
            double open_corr = 0.0;
            double closed_share = 0.0;
            for (const OutputObservation& o : obs.outputs) {
                pinned = pinned && o.constant_on_closed_samples;
                open_corr += o.open_side_correlation / static_cast<double>(obs.outputs.size());
                closed_share += o.closed_error_share / static_cast<double>(obs.outputs.size());
            }
            if (pinned) {
                std::string out = "On the closed side every output is pinned to one constant regardless of the "
                                  "target; on the open side the predictions ";
                out += open_corr >= 0.9 ? "follow the targets closely" : "vary from sample to sample";
                out += closed_share >= 0.7 ? ", so most of the error sits on the closed side." : ".";
                return out;
            }
        }

        std::array<std::vector<std::size_t>, 6> groups;
        std::array<double, 6> amplitude_sum{};
        for (const OutputObservation& o : obs.outputs) {
            const std::size_t c = category(o);
            groups[c].push_back(o.index);
            amplitude_sum[c] += o.amplitude_ratio;
        }

        const bool single = obs.outputs.size() == 1;
        std::string out;
        if (obs.outputs_move_in_lockstep) {
            out = "All outputs are driven by one shared signal, so their predictions move in rigid proportion.";
        }
        for (std::size_t c = 0; c < groups.size(); ++c) {
            if (groups[c].empty()) {
                continue;
            }
            const bool plural = groups[c].size() > 1;
            const std::string subject =
                single ? "The prediction"
                       : (plural ? "Outputs " : "Output ") + EnglishListFormatter::indices(groups[c]);
            const std::string their = plural ? "their targets" : "its target";
            std::string clause;
            switch (c) {
                case 0:
                    clause = std::string(plural ? " keep" : " keeps") + " the sign and ordering of " + their + " but " +
                             (plural ? "are" : "is") + " damped to " +
                             fraction_phrase(amplitude_sum[c] / static_cast<double>(groups[c].size())) +
                             " of the target amplitude";
                    break;
                case 1: clause = std::string(plural ? " follow " : " follows ") + their + " closely"; break;
                case 2:
                    clause = std::string(plural ? " stay nearly flat and carry" : " stays nearly flat and carries") +
                             " no information about " + their;
                    break;
                case 3: clause = std::string(plural ? " follow " : " follows ") + their + " only partially"; break;
                case 4: clause = std::string(plural ? " move against " : " moves against ") + their; break;
                default: clause = std::string(plural ? " bear" : " bears") + " little relation to " + their; break;
            }
            out += (out.empty() ? "" : " ") + subject + clause + ".";
        }
        return out;
    }

private:
    // 0 damped but sign/order kept, 1 tracks closely, 2 flat, 3 partial, 4 inverted, 5 unrelated.
    static std::size_t category(const OutputObservation& o) {
        if (o.sign_preserved && o.order_preserved && o.amplitude_ratio < 0.7) {
            return 0;
        }
        if (o.correlation >= 0.95) {
            return 1;
        }
        if (o.flat) {
            return 2;
        }
        if (o.correlation >= 0.6) {
            return 3;
        }
        if (o.correlation <= -0.3) {
            return 4;
        }
        return 5;
    }

    static std::string fraction_phrase(double ratio) {
        if (ratio < 0.2) {
            return "a small fraction";
        }
        if (ratio < 0.3) {
            return "roughly a quarter";
        }
        if (ratio < 0.42) {
            return "roughly a third";
        }
        if (ratio < 0.6) {
            return "roughly half";
        }
        return "most";
    }

    static std::string dimension_phrase(std::size_t rank) {
        if (rank <= 1) {
            return "an essentially one-dimensional band";
        }
        if (rank == 2) {
            return "a two-dimensional sheet";
        }
        return "a " + NumberWords::word(rank) + "-dimensional region";
    }

    static std::string layer_ordinal(std::size_t layer) {
        static const std::array<std::string, 4> names = {"first", "second", "third", "fourth"};
        return layer < names.size() ? names[layer] : "layer-" + std::to_string(layer + 1);
    }

    static std::string layer_adjective(std::size_t layer) {
        return layer_ordinal(layer) + "-layer";
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\observation_similarity.hpp ===
#pragma once

#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>

namespace forward_analysis {

// Word-overlap F1 between generated and reference observation text (punctuation ignored, case-folded).
struct ObservationSimilarity {
    static double field_f1(const std::string& generated, const std::string& reference) {
        const std::map<std::string, std::size_t> g = word_counts(generated);
        const std::map<std::string, std::size_t> r = word_counts(reference);
        std::size_t g_total = 0;
        std::size_t r_total = 0;
        std::size_t overlap = 0;
        for (const auto& [word, count] : g) {
            g_total += count;
            const auto it = r.find(word);
            if (it != r.end()) {
                overlap += std::min(count, it->second);
            }
        }
        for (const auto& entry : r) {
            r_total += entry.second;
        }
        if (g_total == 0 || r_total == 0 || overlap == 0) {
            return 0.0;
        }
        const double precision = static_cast<double>(overlap) / static_cast<double>(g_total);
        const double recall = static_cast<double>(overlap) / static_cast<double>(r_total);
        return 2.0 * precision * recall / (precision + recall);
    }

    static double mean_f1(const ObservationText& generated, const ObservationText& reference) {
        double sum = 0.0;
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            sum += field_f1(generated.field(f), reference.field(f));
        }
        return sum / static_cast<double>(ObservationText::kFieldCount);
    }

private:
    static std::map<std::string, std::size_t> word_counts(const std::string& text) {
        std::map<std::string, std::size_t> counts;
        for (const std::string& lex : LexemeSplitter::split(LexemeSplitter::lowercase(text))) {
            if (lex.size() == 1 && LexemeSplitter::is_punctuation(lex[0])) {
                continue;
            }
            ++counts[lex];
        }
        return counts;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\description\observation_text.hpp ===
#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace forward_analysis {

// The four qualitative observation fields a sample's ground truth (and the model's output) consists of.
struct ObservationText {
    static constexpr std::size_t kFieldCount = 4;

    // Which units carry signal, which are silent or gated, and how routing changes across samples.
    std::string pathway_activity;
    // Which input coordinates drive the hidden activity and which are ignored.
    std::string feature_sensitivity;
    // Geometry of the hidden representations: rays, sheets, clusters, collapse.
    std::string representational_structure;
    // How predictions track, damp, flatten or distort the targets.
    std::string functional_mapping;

    static const std::array<std::string, kFieldCount>& field_names() {
        static const std::array<std::string, kFieldCount> names = {
            "pathway_activity", "feature_sensitivity", "representational_structure", "functional_mapping"};
        return names;
    }

    std::string& field(std::size_t index) {
        switch (index) {
            case 0: return pathway_activity;
            case 1: return feature_sensitivity;
            case 2: return representational_structure;
            case 3: return functional_mapping;
            default: throw std::out_of_range("ObservationText field index must be < 4");
        }
    }

    const std::string& field(std::size_t index) const {
        return const_cast<ObservationText*>(this)->field(index);
    }

    nlohmann::json to_json() const {
        nlohmann::json out = nlohmann::json::object();
        for (std::size_t f = 0; f < kFieldCount; ++f) {
            out[field_names()[f]] = field(f);
        }
        return out;
    }

    static ObservationText from_json(const nlohmann::json& node) {
        if (!node.is_object()) {
            throw std::invalid_argument("target_text must be an object with the four observation fields");
        }
        ObservationText out;
        for (std::size_t f = 0; f < kFieldCount; ++f) {
            const std::string& name = field_names()[f];
            if (!node.contains(name) || !node.at(name).is_string() || node.at(name).get<std::string>().empty()) {
                throw std::invalid_argument("target_text." + name + " must be a non-empty string");
            }
            out.field(f) = node.at(name).get<std::string>();
        }
        return out;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\encoder\channel_feature_extractor.hpp ===
#pragma once

#include "forward_analysis/analysis/observation_extractor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/encoder/channel_role.hpp"
#include "forward_analysis/math/spectral_ops.hpp"
#include "forward_analysis/math/statistics_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace forward_analysis {

// Turns telemetry of any shape into a variable-length set of fixed-width channel tokens.
// Row order: inputs, hidden units layer by layer, then for each output its target, prediction
// and residual. Every feature is scale-bounded so networks of different sizes share one space.
struct ChannelFeatureExtractor {
    // Width of a channel token.
    static constexpr std::size_t kFeatureDim = 25;

    static Eigen::MatrixXd extract(const ForwardTelemetryTensor& telemetry,
                                   const DistillerConfig& config = DistillerConfig{}) {
        if (telemetry.num_samples() == 0 || telemetry.layer_activations.empty()) {
            throw std::invalid_argument("ChannelFeatureExtractor requires samples and at least one hidden layer");
        }
        const TelemetryShape shape = telemetry.shape();
        const Eigen::MatrixXd& first = telemetry.layer_activations.front();
        const Eigen::MatrixXd scores = SpectralOps::principal_scores(first, 2);
        const Eigen::VectorXd closed = closed_indicator(first, config);
        const double ray = ObservationExtractor::observe_layer(first, config).lies_on_ray ? 1.0 : 0.0;
        const double log_n = std::log(static_cast<double>(shape.num_samples)) / std::log(16.0);
        const double layer_span = static_cast<double>(shape.hidden_widths.size() + 1);

        Eigen::MatrixXd features =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(shape.channel_count()), kFeatureDim);
        Eigen::Index row = 0;
        const auto emit = [&](const Eigen::VectorXd& column, ChannelRole role, double depth, std::size_t index,
                              std::size_t group_width, const Eigen::VectorXd* paired) {
            fill_row(features.row(row++), column, role, depth, index, group_width, paired, scores, closed, ray,
                     log_n);
        };

        const Eigen::Index in_dim = telemetry.inputs.cols();
        for (Eigen::Index c = 0; c < in_dim; ++c) {
            emit(telemetry.inputs.col(c), ChannelRole::Input, 0.0, static_cast<std::size_t>(c),
                 static_cast<std::size_t>(in_dim), nullptr);
        }
        for (std::size_t l = 0; l < telemetry.layer_activations.size(); ++l) {
            const Eigen::MatrixXd& layer = telemetry.layer_activations[l];
            const double depth = static_cast<double>(l + 1) / layer_span;
            for (Eigen::Index u = 0; u < layer.cols(); ++u) {
                emit(layer.col(u), ChannelRole::Hidden, depth, static_cast<std::size_t>(u),
                     static_cast<std::size_t>(layer.cols()), nullptr);
            }
        }
        const Eigen::Index out_dim = telemetry.targets.cols();
        for (Eigen::Index k = 0; k < out_dim; ++k) {
            const Eigen::VectorXd t = telemetry.targets.col(k);
            const Eigen::VectorXd p = telemetry.predictions.col(k);
            const Eigen::VectorXd r = telemetry.residuals.col(k);
            const std::size_t idx = static_cast<std::size_t>(k);
            const std::size_t w = static_cast<std::size_t>(out_dim);
            emit(t, ChannelRole::Target, 1.0, idx, w, &p);
            emit(p, ChannelRole::Prediction, 1.0, idx, w, &t);
            emit(r, ChannelRole::Residual, 1.0, idx, w, &t);
        }
        return features;
    }

private:
    static Eigen::VectorXd closed_indicator(const Eigen::MatrixXd& first, const DistillerConfig& config) {
        Eigen::VectorXd closed(first.rows());
        for (Eigen::Index i = 0; i < first.rows(); ++i) {
            closed(i) = first.row(i).cwiseAbs().maxCoeff() <= config.dead_activation_threshold ? 1.0 : 0.0;
        }
        return closed;
    }

    template <typename Row>
    static void fill_row(Row out,
                         const Eigen::VectorXd& v,
                         ChannelRole role,
                         double depth,
                         std::size_t index,
                         std::size_t group_width,
                         const Eigen::VectorXd* paired,
                         const Eigen::MatrixXd& scores,
                         const Eigen::VectorXd& closed,
                         double ray,
                         double log_n) {
        constexpr double kTwoPi = 6.283185307179586;
        const double n = static_cast<double>(v.size());
        const double std_v = StatisticsOps::standard_deviation(v);

        out(static_cast<Eigen::Index>(role)) = 1.0;
        out(5) = depth;
        const double phase = kTwoPi * static_cast<double>(index) / static_cast<double>(group_width);
        out(6) = std::sin(phase);
        out(7) = std::cos(phase);
        out(8) = static_cast<double>(index) / static_cast<double>(group_width);
        out(9) = std::tanh(v.mean());
        out(10) = std::tanh(std_v);
        out(11) = std::tanh(v.minCoeff());
        out(12) = std::tanh(v.maxCoeff());
        out(13) = (v.array().abs() <= 1e-9).cast<double>().sum() / n;
        out(14) = (v.array() > 1e-9).cast<double>().sum() / n;
        out(15) = (v.array() < -1e-9).cast<double>().sum() / n;
        out(16) = StatisticsOps::pearson(v, scores.col(0));
        out(17) = StatisticsOps::pearson(v, scores.col(1));
        out(18) = StatisticsOps::pearson(v, closed);
        if (paired != nullptr) {
            out(19) = StatisticsOps::pearson(v, *paired);
            if (role == ChannelRole::Residual) {
                out(20) = std::tanh(std_v / std::max(1e-9, StatisticsOps::standard_deviation(*paired)));
            } else {
                out(20) = (v.array() * paired->array() > 0.0).cast<double>().sum() / n;
            }
        }
        out(21) = std::log1p(static_cast<double>(group_width)) / std::log(32.0);
        out(22) = log_n;
        out(23) = ray;
        out(24) = std_v <= 1e-9 ? 1.0 : 0.0;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\encoder\channel_role.hpp ===
#pragma once

#include <cstddef>

namespace forward_analysis {

// What a single telemetry column is: one token is produced per input coordinate, hidden unit,
// and per output for its target, prediction and residual.
enum class ChannelRole : std::size_t {
    Input = 0,
    Hidden = 1,
    Target = 2,
    Prediction = 3,
    Residual = 4,
};

struct ChannelRoleNames {
    // Number of distinct roles (width of the role one-hot block).
    static constexpr std::size_t kCount = 5;

    static const char* name(ChannelRole role) noexcept {
        switch (role) {
            case ChannelRole::Input: return "input";
            case ChannelRole::Hidden: return "hidden";
            case ChannelRole::Target: return "target";
            case ChannelRole::Prediction: return "prediction";
            case ChannelRole::Residual: return "residual";
        }
        return "unknown";
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\encoder\channel_sequence_formatter.hpp ===
#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/encoder/channel_feature_extractor.hpp"
#include "forward_analysis/encoder/linear_input_adapter.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <cstdint>

namespace forward_analysis {

// Channel tokens (C x F) projected by the linear adapter into the encoder's input sequence (C x d_model).
class ChannelSequenceFormatter {
public:
    explicit ChannelSequenceFormatter(const DistillerConfig& config = DistillerConfig{},
                                      std::uint64_t seed = 1337ULL)
        : config_(config),
          adapter_(config.d_model, ChannelFeatureExtractor::kFeatureDim, seed) {
        config_.validate();
    }

    Eigen::MatrixXd channel_features(const ForwardTelemetryTensor& telemetry) const {
        return ChannelFeatureExtractor::extract(telemetry, config_);
    }

    Eigen::MatrixXd format_sequence(const Eigen::MatrixXd& features) {
        return adapter_.project(features);
    }

    Eigen::MatrixXd format_sequence(const ForwardTelemetryTensor& telemetry) {
        return format_sequence(channel_features(telemetry));
    }

    void update_adapter(const Eigen::MatrixXd& features, const Eigen::MatrixXd& d_sequence, double learning_rate) {
        adapter_.update_from_sequence_grad(features, d_sequence, learning_rate);
    }

    const LinearInputAdapter& adapter() const noexcept { return adapter_; }

private:
    // Thresholds used by the feature extractor and the model width of the adapter.
    DistillerConfig config_;
    // Learned projection from channel features (F) to d_model.
    LinearInputAdapter adapter_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\encoder\linear_input_adapter.hpp ===
#pragma once

#include "forward_analysis/math/deterministic_rng.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

class LinearInputAdapter {
public:
    LinearInputAdapter() = default;

    explicit LinearInputAdapter(std::size_t d_model,
                                std::size_t initial_d_total = 32,
                                std::uint64_t seed = 1337ULL)
        : d_model_(d_model),
          seed_(seed),
          weights_(static_cast<Eigen::Index>(d_model), 0),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d_model))) {
        ensure_input_capacity(initial_d_total);
    }

    void ensure_input_capacity(std::size_t required_d_total) {
        const Eigen::Index required = static_cast<Eigen::Index>(required_d_total);
        const Eigen::Index current_cols = weights_.cols();
        if (required <= current_cols) {
            return;
        }
        DeterministicRng rng(seed_ + static_cast<std::uint64_t>(current_cols) * 7919ULL);
        const double scale = 1.0 / std::sqrt(static_cast<double>(d_model_ + required_d_total));
        weights_.conservativeResize(Eigen::NoChange, required);
        for (Eigen::Index r = 0; r < weights_.rows(); ++r) {
            for (Eigen::Index c = current_cols; c < required; ++c) {
                weights_(r, c) = rng.uniform(-scale, scale);
            }
        }
    }

    Eigen::MatrixXd project(const Eigen::MatrixXd& probes) {
        ensure_input_capacity(static_cast<std::size_t>(probes.cols()));
        Eigen::MatrixXd projected = probes * weights_.leftCols(probes.cols()).transpose();
        projected.rowwise() += biases_.transpose();
        return projected;
    }

    void update_from_sequence_grad(const Eigen::MatrixXd& probes,
                                   const Eigen::MatrixXd& d_projected,
                                   double learning_rate) {
        if (probes.rows() == 0 || d_projected.rows() == 0 || learning_rate <= 0.0) {
            return;
        }
        if (d_projected.cols() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("LinearInputAdapter gradient width must equal d_model");
        }
        const Eigen::Index n = std::min(probes.rows(), d_projected.rows());
        ensure_input_capacity(static_cast<std::size_t>(probes.cols()));

        const Eigen::MatrixXd grad =
            d_projected.topRows(n).cwiseMax(-5.0).cwiseMin(5.0) * (learning_rate / static_cast<double>(n));
        biases_ -= grad.colwise().sum().transpose();
        weights_.leftCols(probes.cols()).noalias() -= grad.transpose() * probes.topRows(n);
    }

    std::size_t d_model() const noexcept { return d_model_; }
    const Eigen::MatrixXd& weights() const noexcept { return weights_; }
    const Eigen::VectorXd& biases() const noexcept { return biases_; }

private:
    // Output width of the projection (the encoder's model width).
    std::size_t d_model_ = 256;
    // Base seed for initializing newly added weight columns when the probe width grows.
    std::uint64_t seed_ = 1337ULL;
    // Projection matrix (d_model x probe width); grows column-wise to fit wider telemetry.
    Eigen::MatrixXd weights_;
    // Bias added to every projected probe row (d_model).
    Eigen::VectorXd biases_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\encoder\perceiver_forward_cache.hpp ===
#pragma once

#include <Eigen/Core>

namespace forward_analysis {

// Intermediate tensors from one Perceiver forward pass, kept so the backward pass can reuse them.
// N = number of probes, M = number of latents, D = d_model.
struct PerceiverForwardCache {
    // Adapted probe sequence fed to the encoder (N x D).
    Eigen::MatrixXd input_sequence;
    // Latent queries projected by W_q (M x D).
    Eigen::MatrixXd queries;
    // Probe sequence projected by W_k (N x D).
    Eigen::MatrixXd keys;
    // Probe sequence projected by W_v (N x D).
    Eigen::MatrixXd values;
    // Softmax cross-attention weights; row m is how latent m distributes attention over probes (M x N).
    Eigen::MatrixXd attn_weights;
    // Attention-weighted sum of values per latent (M x D).
    Eigen::MatrixXd context;
    // Latent queries + output projection of context, before layer norm (M x D).
    Eigen::MatrixXd pre_norm_latents;
    // Final layer-normalized latent bottleneck Z (M x D).
    Eigen::MatrixXd latents_z;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\encoder\perceiver_io_encoder.hpp ===
#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/encoder/perceiver_forward_cache.hpp"
#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

class PerceiverIOEncoder {
public:
    explicit PerceiverIOEncoder(const DistillerConfig& config = DistillerConfig{},
                                std::uint64_t seed = 2026ULL)
        : config_(config) {
        config_.validate();
        initialize_parameters(seed);
    }

    void initialize_parameters(std::uint64_t seed) {
        DeterministicRng rng(seed);
        const std::size_t m = config_.num_latents;
        const std::size_t d = config_.d_model;

        latent_queries_ = rng.xavier_matrix(m, d);
        w_q_ = rng.xavier_matrix(d, d);
        w_k_ = rng.xavier_matrix(d, d);
        w_v_ = rng.xavier_matrix(d, d);
        w_o_ = rng.xavier_matrix(d, d);
        b_o_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d));
    }

    PerceiverForwardCache forward_with_cache(const Eigen::MatrixXd& input_sequence) const {
        if (input_sequence.rows() == 0) {
            throw std::invalid_argument("PerceiverIOEncoder input_sequence cannot be empty");
        }
        if (input_sequence.cols() != static_cast<Eigen::Index>(config_.d_model)) {
            throw std::invalid_argument("PerceiverIOEncoder input dimension mismatch with d_model");
        }

        const double scale = 1.0 / std::sqrt(static_cast<double>(config_.d_model));

        PerceiverForwardCache cache;
        cache.input_sequence = input_sequence;
        cache.queries.noalias() = latent_queries_ * w_q_.transpose();
        cache.keys.noalias() = input_sequence * w_k_.transpose();
        cache.values.noalias() = input_sequence * w_v_.transpose();
        cache.attn_weights = MatrixOps::softmax_rows((cache.queries * cache.keys.transpose()) * scale);
        cache.context.noalias() = cache.attn_weights * cache.values;
        cache.pre_norm_latents = latent_queries_ + MatrixOps::affine_rows(cache.context, w_o_, b_o_);
        cache.latents_z = MatrixOps::layer_norm_rows(cache.pre_norm_latents);
        return cache;
    }

    Eigen::MatrixXd encode(const Eigen::MatrixXd& input_sequence) const {
        return forward_with_cache(input_sequence).latents_z;
    }

    Eigen::MatrixXd backward_and_update(const PerceiverForwardCache& cache,
                                        const Eigen::MatrixXd& d_latents_z,
                                        double learning_rate) {
        const Eigen::Index n = cache.input_sequence.rows();
        const Eigen::Index m = static_cast<Eigen::Index>(config_.num_latents);
        const Eigen::Index d = static_cast<Eigen::Index>(config_.d_model);

        if (learning_rate <= 0.0 || d_latents_z.rows() != m || d_latents_z.cols() != d) {
            return Eigen::MatrixXd::Zero(n, d);
        }

        const double lr = MatrixOps::clamp(learning_rate, 0.0, 0.25);
        const Eigen::MatrixXd d_pre = d_latents_z.cwiseMax(-4.0).cwiseMin(4.0);
        const Eigen::MatrixXd d_context = d_pre * w_o_;

        b_o_ -= lr * d_pre.colwise().sum().transpose();
        latent_queries_ -= lr * d_pre;
        w_o_.noalias() -= lr * (d_pre.transpose() * cache.context);

        const Eigen::MatrixXd d_values =
            (cache.attn_weights.transpose() * d_context).cwiseMax(-4.0).cwiseMin(4.0);
        Eigen::MatrixXd d_input = d_values * w_v_;

        const double n_scale = lr / static_cast<double>(std::max<Eigen::Index>(1, n));
        w_v_.noalias() -= n_scale * (d_values.transpose() * cache.input_sequence);
        return d_input;
    }

    const Eigen::MatrixXd& latent_queries() const noexcept {
        return latent_queries_;
    }

    std::size_t num_latents() const noexcept { return config_.num_latents; }
    std::size_t d_model() const noexcept { return config_.d_model; }

private:
    // Model dimensions (num_latents, d_model).
    DistillerConfig config_;
    // Learned latent query vectors, one row per latent (M x D); also the residual base of Z.
    Eigen::MatrixXd latent_queries_;
    // Query projection applied to latent queries (D x D).
    Eigen::MatrixXd w_q_;
    // Key projection applied to the probe sequence (D x D).
    Eigen::MatrixXd w_k_;
    // Value projection applied to the probe sequence (D x D).
    Eigen::MatrixXd w_v_;
    // Output projection applied to the attention context (D x D).
    Eigen::MatrixXd w_o_;
    // Output projection bias (D).
    Eigen::VectorXd b_o_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\encoder\sinusoidal_positional_encoding.hpp ===
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstddef>

namespace forward_analysis {

struct SinusoidalPositionalEncoding {
    static Eigen::MatrixXd encoding(std::size_t num_positions, std::size_t d_model) {
        const Eigen::Index n = static_cast<Eigen::Index>(num_positions);
        const Eigen::Index d = static_cast<Eigen::Index>(d_model);
        Eigen::MatrixXd pe(n, d);
        if (n == 0 || d == 0) {
            return pe;
        }
        const Eigen::ArrayXd positions = Eigen::ArrayXd::LinSpaced(n, 1.0, static_cast<double>(n));
        for (Eigen::Index dim = 0; dim < d; ++dim) {
            const Eigen::Index pair_idx = (dim / 2) * 2;
            const double exponent = static_cast<double>(pair_idx) / static_cast<double>(d);
            const Eigen::ArrayXd angle = positions / std::pow(10000.0, exponent);
            if (dim % 2 == 0) {
                pe.col(dim) = angle.sin().matrix();
            } else {
                pe.col(dim) = angle.cos().matrix();
            }
        }
        return pe;
    }

    static void add_in_place(Eigen::MatrixXd& sequence) {
        if (sequence.size() == 0) {
            return;
        }
        sequence += encoding(static_cast<std::size_t>(sequence.rows()),
                             static_cast<std::size_t>(sequence.cols()));
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\heads\auxiliary_alignment_heads.hpp ===
#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/heads/auxiliary_evaluation.hpp"
#include "forward_analysis/heads/eigen_sign_head.hpp"
#include "forward_analysis/heads/observation_regression_head.hpp"

#include <Eigen/Core>

#include <cstdint>

namespace forward_analysis {

// Aux1 (observation descriptor) and Aux2 (eigen signs) heads, both reading mean-pooled latents.
class AuxiliaryAlignmentHeads {
public:
    explicit AuxiliaryAlignmentHeads(const DistillerConfig& config = DistillerConfig{},
                                     std::uint64_t seed = 3000ULL)
        : config_(config),
          observation_head_(config.d_model, ObservationDescriptor::dim(), seed + 1ULL),
          eigen_sign_head_(config.d_model, config.eigen_sign_dim, seed + 2ULL) {
        config_.validate();
    }

    AuxiliaryEvaluation evaluate(const Eigen::MatrixXd& latents_z,
                                 const Eigen::VectorXd& target_descriptor = Eigen::VectorXd(),
                                 const Eigen::VectorXd& target_eigen_signs = Eigen::VectorXd()) const {
        const Eigen::VectorXd pooled = latents_z.colwise().mean().transpose();
        AuxiliaryEvaluation out;
        out.predicted_observation_descriptor = observation_head_.predict(pooled);
        out.predicted_eigen_signs = eigen_sign_head_.predict_signs(pooled);
        out.aux1_observation_loss = observation_head_.compute_loss(pooled, target_descriptor);
        if (target_eigen_signs.size() > 0) {
            out.aux2_eigen_sign_loss = eigen_sign_head_.compute_loss(pooled, target_eigen_signs);
        }
        return out;
    }

    AuxiliaryEvaluation compute_losses_and_step(const Eigen::MatrixXd& latents_z,
                                                const Eigen::VectorXd& target_descriptor,
                                                const Eigen::VectorXd& target_eigen_signs,
                                                double learning_rate,
                                                Eigen::MatrixXd& d_latents_z_accum) {
        const Eigen::VectorXd pooled = latents_z.colwise().mean().transpose();
        Eigen::VectorXd d_pooled = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(config_.d_model));

        AuxiliaryEvaluation out;
        out.predicted_observation_descriptor = observation_head_.predict(pooled);
        out.predicted_eigen_signs = eigen_sign_head_.predict_signs(pooled);
        out.aux1_observation_loss = observation_head_.compute_loss_and_step(
            pooled, target_descriptor, learning_rate, config_.aux1_observation_weight, d_pooled);
        out.aux2_eigen_sign_loss = eigen_sign_head_.compute_loss_and_step(
            pooled, target_eigen_signs, learning_rate, config_.aux2_eigen_sign_weight, d_pooled);

        if (d_latents_z_accum.rows() != latents_z.rows() || d_latents_z_accum.cols() != latents_z.cols()) {
            d_latents_z_accum = Eigen::MatrixXd::Zero(latents_z.rows(), latents_z.cols());
        }
        d_latents_z_accum.rowwise() += d_pooled.transpose() / static_cast<double>(latents_z.rows());
        return out;
    }

    const ObservationRegressionHead& observation_head() const noexcept { return observation_head_; }
    const EigenSignHead& eigen_sign_head() const noexcept { return eigen_sign_head_; }

private:
    // d_model, eigen_sign_dim and the Aux1/Aux2 loss weights.
    DistillerConfig config_;
    // Aux1: regresses the observation descriptor from the mean-pooled latents.
    ObservationRegressionHead observation_head_;
    // Aux2: predicts dominant-eigenvector signs from the mean-pooled latents.
    EigenSignHead eigen_sign_head_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\heads\auxiliary_evaluation.hpp ===
#pragma once

#include <Eigen/Core>

namespace forward_analysis {

struct AuxiliaryEvaluation {
    // Descriptor predicted from the latent bottleneck (same layout as ObservationDescriptor).
    Eigen::VectorXd predicted_observation_descriptor;
    // Predicted dominant-eigenvector signs (+1 / -1).
    Eigen::VectorXd predicted_eigen_signs;
    // Aux1: half mean squared error between predicted and measured descriptor.
    double aux1_observation_loss = 0.0;
    // Aux2: binary cross-entropy between predicted and target eigenvector signs.
    double aux2_eigen_sign_loss = 0.0;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\heads\eigen_sign_head.hpp ===
#pragma once

#include "forward_analysis/math/deterministic_rng.hpp"
#include "forward_analysis/math/matrix_ops.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

class EigenSignHead {
public:
    explicit EigenSignHead(std::size_t d_model = 256,
                           std::size_t eigen_dim = 4,
                           std::uint64_t seed = 3002ULL)
        : d_model_(d_model),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(eigen_dim))) {
        DeterministicRng rng(seed);
        weights_ = rng.xavier_matrix(eigen_dim, d_model_);
    }

    Eigen::VectorXd predict_sign_probabilities(const Eigen::VectorXd& pooled_z) const {
        if (pooled_z.size() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("EigenSignHead input width must equal d_model");
        }
        return MatrixOps::sigmoid(weights_ * pooled_z + biases_);
    }

    Eigen::VectorXd predict_signs(const Eigen::VectorXd& pooled_z) const {
        return predict_sign_probabilities(pooled_z).unaryExpr(
            [](double p) { return p >= 0.5 ? 1.0 : -1.0; });
    }

    double compute_loss(const Eigen::VectorXd& pooled_z,
                        const Eigen::VectorXd& target_signs) const {
        const Eigen::VectorXd probs = predict_sign_probabilities(pooled_z);
        const Eigen::Index k = std::min(probs.size(), target_signs.size());
        if (k == 0) {
            return 0.0;
        }
        return binary_cross_entropy(clamp_probs(probs.head(k)), sign_labels(target_signs.head(k)));
    }

    double compute_loss_and_step(const Eigen::VectorXd& pooled_z,
                                 const Eigen::VectorXd& target_signs,
                                 double learning_rate,
                                 double loss_weight,
                                 Eigen::VectorXd& d_pooled_accum) {
        const Eigen::VectorXd probs = predict_sign_probabilities(pooled_z);
        const Eigen::Index k = std::min(probs.size(), target_signs.size());
        if (k == 0) {
            return 0.0;
        }
        if (d_pooled_accum.size() != static_cast<Eigen::Index>(d_model_)) {
            d_pooled_accum = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d_model_));
        }

        const Eigen::ArrayXd p = clamp_probs(probs.head(k));
        const Eigen::ArrayXd y = sign_labels(target_signs.head(k));
        const double inv_k = 1.0 / static_cast<double>(k);
        const Eigen::VectorXd d_logit = ((p - y) * (inv_k * loss_weight)).matrix();

        d_pooled_accum.noalias() += weights_.topRows(k).transpose() * d_logit;
        if (learning_rate > 0.0) {
            biases_.head(k) -= learning_rate * d_logit;
            weights_.topRows(k).noalias() -= (learning_rate * d_logit) * pooled_z.transpose();
        }
        return binary_cross_entropy(p, y);
    }

private:
    static Eigen::ArrayXd clamp_probs(const Eigen::VectorXd& probs) {
        return probs.array().cwiseMax(1e-12).cwiseMin(1.0 - 1e-12);
    }

    static Eigen::ArrayXd sign_labels(const Eigen::VectorXd& signs) {
        return (signs.array() >= 0.0).cast<double>();
    }

    static double binary_cross_entropy(const Eigen::ArrayXd& p, const Eigen::ArrayXd& y) {
        return -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean();
    }

    // Width of the pooled latent vector this head reads.
    std::size_t d_model_ = 256;
    // Linear map from pooled latent to one logit per eigenvector component (eigen_dim x d_model).
    Eigen::MatrixXd weights_;
    // Bias per eigenvector component logit (eigen_dim).
    Eigen::VectorXd biases_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\heads\observation_regression_head.hpp ===
#pragma once

#include "forward_analysis/math/deterministic_rng.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace forward_analysis {

// Aux1: linear regression from the pooled latent vector to the ObservationDescriptor.
class ObservationRegressionHead {
public:
    explicit ObservationRegressionHead(std::size_t d_model = 256,
                                       std::size_t descriptor_dim = 14,
                                       std::uint64_t seed = 3001ULL)
        : d_model_(d_model),
          biases_(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(descriptor_dim))) {
        DeterministicRng rng(seed);
        weights_ = rng.xavier_matrix(descriptor_dim, d_model_);
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& pooled_z) const {
        if (pooled_z.size() != static_cast<Eigen::Index>(d_model_)) {
            throw std::invalid_argument("ObservationRegressionHead input width must equal d_model");
        }
        return weights_ * pooled_z + biases_;
    }

    double compute_loss(const Eigen::VectorXd& pooled_z, const Eigen::VectorXd& target) const {
        if (target.size() != biases_.size()) {
            return 0.0;
        }
        return 0.5 * (predict(pooled_z) - target).squaredNorm() / static_cast<double>(target.size());
    }

    double compute_loss_and_step(const Eigen::VectorXd& pooled_z,
                                 const Eigen::VectorXd& target,
                                 double learning_rate,
                                 double loss_weight,
                                 Eigen::VectorXd& d_pooled_accum) {
        if (target.size() != biases_.size()) {
            return 0.0;
        }
        if (d_pooled_accum.size() != static_cast<Eigen::Index>(d_model_)) {
            d_pooled_accum = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d_model_));
        }
        const Eigen::VectorXd error = predict(pooled_z) - target;
        const double inv_k = 1.0 / static_cast<double>(target.size());
        const Eigen::VectorXd d_out = error * (inv_k * loss_weight);

        d_pooled_accum.noalias() += weights_.transpose() * d_out;
        if (learning_rate > 0.0) {
            const Eigen::VectorXd d_param = error * inv_k;
            biases_ -= learning_rate * d_param;
            weights_.noalias() -= (learning_rate * d_param) * pooled_z.transpose();
        }
        return 0.5 * error.squaredNorm() * inv_k;
    }

    std::size_t descriptor_dim() const noexcept { return static_cast<std::size_t>(biases_.size()); }

private:
    // Width of the pooled latent vector this head reads.
    std::size_t d_model_ = 256;
    // Linear map from pooled latent to descriptor (descriptor_dim x d_model).
    Eigen::MatrixXd weights_;
    // Bias per descriptor component (descriptor_dim).
    Eigen::VectorXd biases_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\math\adam_moment.hpp ===
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstdint>

namespace forward_analysis {

// First/second moment estimates for one parameter tensor updated with Adam.
class AdamMoment {
public:
    // Grows the moment buffers to rows x cols, zero-filling new entries (used when a vocabulary grows).
    void ensure_shape(Eigen::Index rows, Eigen::Index cols) {
        if (first_.rows() == rows && first_.cols() == cols) {
            return;
        }
        const Eigen::Index old_rows = first_.rows();
        const Eigen::Index old_cols = first_.cols();
        first_.conservativeResize(rows, cols);
        second_.conservativeResize(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) {
                if (r >= old_rows || c >= old_cols) {
                    first_(r, c) = 0.0;
                    second_(r, c) = 0.0;
                }
            }
        }
    }

    const Eigen::MatrixXd& first() const noexcept { return first_; }
    const Eigen::MatrixXd& second() const noexcept { return second_; }

    // Replaces both moment estimates, e.g. with the copies a device has been updating.
    void assign(const Eigen::MatrixXd& first, const Eigen::MatrixXd& second) {
        first_ = first;
        second_ = second;
    }

    static constexpr double beta1() noexcept { return kBeta1; }
    static constexpr double beta2() noexcept { return kBeta2; }
    static constexpr double epsilon() noexcept { return kEpsilon; }

    // param -= lr * m_hat / (sqrt(v_hat) + eps); step is the 1-based global Adam step count.
    template <typename Param, typename Grad>
    void apply(Param& param, const Grad& grad, double learning_rate, std::uint64_t step) {
        ensure_shape(param.rows(), param.cols());
        first_ = kBeta1 * first_ + (1.0 - kBeta1) * grad;
        second_ = kBeta2 * second_ + (1.0 - kBeta2) * grad.cwiseProduct(grad);
        const double t = static_cast<double>(step);
        const double correction1 = 1.0 - std::pow(kBeta1, t);
        const double correction2 = 1.0 - std::pow(kBeta2, t);
        param -= (learning_rate * (first_.array() / correction1) /
                  ((second_.array() / correction2).sqrt() + kEpsilon))
                     .matrix();
    }

private:
    static constexpr double kBeta1 = 0.9;
    static constexpr double kBeta2 = 0.999;
    static constexpr double kEpsilon = 1e-8;

    // Running mean of gradients.
    Eigen::MatrixXd first_;
    // Running mean of squared gradients.
    Eigen::MatrixXd second_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\math\deterministic_rng.hpp ===
#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace forward_analysis {

struct DeterministicRng {
    // splitmix64 internal state; advancing it yields the next pseudo-random 64-bit value.
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;

    explicit DeterministicRng(std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept
        : state(seed == 0 ? 0x9E3779B97F4A7C15ULL : seed) {}

    // Advances the generator as if next_u64() had been called draws times (each call adds one golden increment).
    void skip(std::uint64_t draws) noexcept { state += draws * 0x9E3779B97F4A7C15ULL; }

    std::uint64_t next_u64() noexcept {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    double uniform(double low = 0.0, double high = 1.0) noexcept {
        constexpr double inv = 1.0 / static_cast<double>(1ULL << 53);
        const double unit = static_cast<double>(next_u64() >> 11) * inv;
        return low + unit * (high - low);
    }

    double normal(double mean = 0.0, double stddev = 1.0) noexcept {
        const double u1 = std::max(1e-12, uniform(0.0, 1.0));
        const double u2 = uniform(0.0, 1.0);
        constexpr double two_pi = 6.28318530717958647692;
        const double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
        return mean + z0 * stddev;
    }

    Eigen::MatrixXd xavier_matrix(std::size_t rows, std::size_t cols) {
        const Eigen::Index r_count = static_cast<Eigen::Index>(rows);
        const Eigen::Index c_count = static_cast<Eigen::Index>(cols);
        Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(r_count, c_count);
        if (rows == 0 || cols == 0) {
            return mat;
        }
        const double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
        for (Eigen::Index r = 0; r < r_count; ++r) {
            for (Eigen::Index c = 0; c < c_count; ++c) {
                mat(r, c) = uniform(-limit, limit);
            }
        }
        return mat;
    }

    Eigen::VectorXd uniform_vector(std::size_t dim, double low = -0.1, double high = 0.1) {
        Eigen::VectorXd vec(static_cast<Eigen::Index>(dim));
        for (Eigen::Index i = 0; i < vec.size(); ++i) {
            vec(i) = uniform(low, high);
        }
        return vec;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\math\matrix_ops.hpp ===
#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace forward_analysis {

struct MatrixOps {
    template <typename Derived>
    static void require_finite(const Eigen::DenseBase<Derived>& values) {
        if (!values.allFinite()) {
            throw std::invalid_argument("Non-finite floating-point value encountered");
        }
    }

    static double clamp(double value, double low, double high) noexcept {
        return std::max(low, std::min(high, value));
    }

    // Row-major sample convention: X is N x D_in, W is D_out x D_in, result is N x D_out.
    static Eigen::MatrixXd affine_rows(const Eigen::MatrixXd& X,
                                       const Eigen::MatrixXd& W,
                                       const Eigen::VectorXd& b) {
        Eigen::MatrixXd out = X * W.transpose();
        out.rowwise() += b.transpose();
        return out;
    }

    static Eigen::VectorXd softmax(const Eigen::VectorXd& logits) {
        if (logits.size() == 0) {
            return logits;
        }
        const Eigen::ArrayXd shifted = (logits.array() - logits.maxCoeff()).cwiseMax(-60.0).exp();
        const double sum = shifted.sum();
        if (!(sum > 0.0)) {
            return Eigen::VectorXd::Constant(logits.size(), 1.0 / static_cast<double>(logits.size()));
        }
        return (shifted / sum).matrix();
    }

    static Eigen::MatrixXd softmax_rows(const Eigen::MatrixXd& logits) {
        if (logits.size() == 0) {
            return logits;
        }
        const Eigen::VectorXd row_max = logits.rowwise().maxCoeff();
        Eigen::ArrayXXd shifted = (logits.colwise() - row_max).array().cwiseMax(-60.0).exp();
        const Eigen::ArrayXd row_sum = shifted.rowwise().sum();
        shifted.colwise() /= row_sum;
        return shifted.matrix();
    }

    static Eigen::VectorXd layer_norm(const Eigen::VectorXd& x, double eps = 1e-6) {
        if (x.size() == 0) {
            return x;
        }
        const Eigen::ArrayXd centered = x.array() - x.mean();
        const double inv_std = 1.0 / std::sqrt(centered.square().mean() + eps);
        return (centered * inv_std).matrix();
    }

    static Eigen::MatrixXd layer_norm_rows(const Eigen::MatrixXd& X, double eps = 1e-6) {
        if (X.size() == 0) {
            return X;
        }
        const Eigen::MatrixXd centered = X.colwise() - X.rowwise().mean();
        return row_inverse_std(X, eps).asDiagonal() * centered;
    }

    // 1 / sqrt(variance + eps) of each row, the scale layer_norm_rows applies.
    static Eigen::VectorXd row_inverse_std(const Eigen::MatrixXd& X, double eps = 1e-6) {
        const Eigen::MatrixXd centered = X.colwise() - X.rowwise().mean();
        return (centered.array().square().rowwise().mean() + eps).rsqrt().matrix();
    }

    // Gradient through layer_norm_rows given its output and the per-row scale it used.
    static Eigen::MatrixXd layer_norm_rows_backward(const Eigen::MatrixXd& normalized,
                                                    const Eigen::VectorXd& inv_std,
                                                    const Eigen::MatrixXd& d_normalized) {
        const Eigen::VectorXd mean_grad = d_normalized.rowwise().mean();
        const Eigen::VectorXd mean_grad_dot =
            (d_normalized.array() * normalized.array()).rowwise().mean().matrix();
        Eigen::MatrixXd d_input = d_normalized.colwise() - mean_grad;
        d_input -= mean_grad_dot.asDiagonal() * normalized;
        return inv_std.asDiagonal() * d_input;
    }

    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
        const Eigen::ArrayXd clamped = x.array().cwiseMax(-40.0).cwiseMin(40.0);
        return (1.0 + (-clamped).exp()).inverse().matrix();
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\math\spectral_ops.hpp ===
#pragma once

#include "forward_analysis/math/spectral_profile.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace forward_analysis {

struct SpectralOps {
    static Eigen::MatrixXd covariance(const Eigen::MatrixXd& samples) {
        if (samples.size() == 0) {
            return Eigen::MatrixXd();
        }
        const Eigen::RowVectorXd mean = samples.colwise().mean();
        const Eigen::MatrixXd centered = samples.rowwise() - mean;
        const double denom =
            samples.rows() > 1 ? static_cast<double>(samples.rows() - 1) : 1.0;
        return (centered.transpose() * centered) / denom;
    }

    static SpectralProfile analyze_activation_manifold(const Eigen::MatrixXd& activations,
                                                       std::size_t max_components = 4) {
        SpectralProfile profile;
        if (activations.size() == 0) {
            return profile;
        }

        const Eigen::MatrixXd cov = covariance(activations);
        Eigen::BDCSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

        const Eigen::Index k = std::min<Eigen::Index>(
            static_cast<Eigen::Index>(max_components), svd.singularValues().size());
        profile.singular_values = svd.singularValues().head(k);
        profile.principal_axes = svd.matrixV().leftCols(k);

        // SVD axes are sign-ambiguous; pin each so its largest-magnitude entry is positive.
        for (Eigen::Index c = 0; c < k; ++c) {
            Eigen::Index max_row = 0;
            profile.principal_axes.col(c).cwiseAbs().maxCoeff(&max_row);
            if (profile.principal_axes(max_row, c) < 0.0) {
                profile.principal_axes.col(c) *= -1.0;
            }
        }

        // Entries of zero-variance (dead) units are zero up to round-off and count as +1.
        if (k > 0) {
            profile.dominant_eigen_signs = profile.principal_axes.col(0).unaryExpr(
                [](double v) { return v >= -1e-9 ? 1.0 : -1.0; });
        }

        const Eigen::VectorXd& s = profile.singular_values;
        if (s.size() >= 2 && s(0) > 1e-9) {
            const double ratio = s(1) / s(0);
            profile.eccentricity = std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
        } else if (s.size() == 1 && s(0) > 1e-9) {
            profile.eccentricity = 1.0;
        }
        return profile;
    }

    // Per-sample scores on the leading principal axes (N x components); missing axes score 0.
    // Each axis is oriented like analyze_activation_manifold: largest-magnitude loading positive.
    static Eigen::MatrixXd principal_scores(const Eigen::MatrixXd& activations, std::size_t components) {
        const Eigen::Index k = static_cast<Eigen::Index>(components);
        Eigen::MatrixXd scores = Eigen::MatrixXd::Zero(activations.rows(), k);
        if (activations.rows() < 2 || activations.cols() == 0) {
            return scores;
        }
        const Eigen::MatrixXd centered = activations.rowwise() - activations.colwise().mean();
        const SpectralProfile profile = analyze_activation_manifold(activations, components);
        const Eigen::VectorXd& s = profile.singular_values;
        for (Eigen::Index c = 0; c < std::min<Eigen::Index>(k, s.size()); ++c) {
            if (s(c) > 1e-12 * std::max(1.0, s(0))) {
                scores.col(c) = centered * profile.principal_axes.col(c);
            }
        }
        return scores;
    }

    // Singular values of the raw (uncentered) activation matrix, largest first.
    static Eigen::VectorXd uncentered_singular_values(const Eigen::MatrixXd& activations) {
        if (activations.size() == 0) {
            return Eigen::VectorXd();
        }
        return Eigen::BDCSVD<Eigen::MatrixXd>(activations).singularValues();
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\math\spectral_profile.hpp ===
#pragma once

#include <Eigen/Core>

namespace forward_analysis {

struct SpectralProfile {
    // Singular values of the activation covariance matrix, largest first.
    Eigen::VectorXd singular_values;
    // Principal axes (right singular vectors) as columns, each sign-pinned so its largest entry is positive.
    Eigen::MatrixXd principal_axes;
    // Sign (+1 / -1) of each component of the dominant principal axis.
    Eigen::VectorXd dominant_eigen_signs;
    // sqrt(1 - (s1/s0)^2): 0 when the top two directions carry equal variance, 1 when one direction dominates.
    double eccentricity = 0.0;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\math\statistics_ops.hpp ===
#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace forward_analysis {

struct StatisticsOps {
    static double standard_deviation(const Eigen::VectorXd& values) {
        if (values.size() < 2) {
            return 0.0;
        }
        return std::sqrt((values.array() - values.mean()).square().mean());
    }

    // Pearson correlation; 0 when either side has no spread.
    static double pearson(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        if (a.size() != b.size() || a.size() < 2) {
            return 0.0;
        }
        const Eigen::ArrayXd ac = a.array() - a.mean();
        const Eigen::ArrayXd bc = b.array() - b.mean();
        const double denom = std::sqrt(ac.square().sum() * bc.square().sum());
        return denom > 1e-12 ? (ac * bc).sum() / denom : 0.0;
    }

    // 0-based ranks with ties sharing their average rank.
    static Eigen::VectorXd ranks(const Eigen::VectorXd& values) {
        const Eigen::Index n = values.size();
        std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), Eigen::Index{0});
        std::stable_sort(order.begin(), order.end(),
                         [&values](Eigen::Index l, Eigen::Index r) { return values(l) < values(r); });
        Eigen::VectorXd out(n);
        Eigen::Index start = 0;
        while (start < n) {
            Eigen::Index end = start;
            while (end + 1 < n && values(order[static_cast<std::size_t>(end + 1)]) ==
                                      values(order[static_cast<std::size_t>(start)])) {
                ++end;
            }
            const double shared = 0.5 * static_cast<double>(start + end);
            for (Eigen::Index k = start; k <= end; ++k) {
                out(order[static_cast<std::size_t>(k)]) = shared;
            }
            start = end + 1;
        }
        return out;
    }

    static double spearman(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return pearson(ranks(a), ranks(b));
    }

    // True when every pair of samples is ordered the same way in both vectors.
    static bool same_order(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (Eigen::Index i = 0; i < a.size(); ++i) {
            for (Eigen::Index j = i + 1; j < a.size(); ++j) {
                const double da = a(i) - a(j);
                const double db = b(i) - b(j);
                if ((da > 0.0 && db <= 0.0) || (da < 0.0 && db >= 0.0) || (da == 0.0 && db != 0.0)) {
                    return false;
                }
            }
        }
        return true;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\telemetry\forward_telemetry_tensor.hpp ===
#pragma once

#include "forward_analysis/telemetry/telemetry_shape.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace forward_analysis {

// Per-sample, per-unit 0/1 flags (N x units), stored compactly as bytes.
using MaskMatrix = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>;

struct ForwardTelemetryTensor {
    // Inputs fed to the observed network (N samples x D_in).
    Eigen::MatrixXd inputs;
    // Desired outputs for each sample (N x D_out).
    Eigen::MatrixXd targets;
    // Outputs the observed network actually produced (N x D_out).
    Eigen::MatrixXd predictions;
    // predictions - targets for each sample (N x D_out).
    Eigen::MatrixXd residuals;
    // Post-activation values of every hidden layer, in forward order; layer l is N x width_l.
    std::vector<Eigen::MatrixXd> layer_activations;
    // Per layer: 1 where |activation| <= dead_activation_threshold (unit not firing on that sample).
    std::vector<MaskMatrix> dead_masks;

    std::size_t num_samples() const noexcept {
        return static_cast<std::size_t>(inputs.rows());
    }

    std::size_t num_layers() const noexcept {
        return layer_activations.size();
    }

    std::size_t input_dim() const noexcept {
        return static_cast<std::size_t>(inputs.cols());
    }

    std::size_t output_dim() const noexcept {
        return static_cast<std::size_t>(targets.cols());
    }

    TelemetryShape shape() const {
        TelemetryShape out;
        out.num_samples = num_samples();
        out.input_dim = input_dim();
        out.output_dim = output_dim();
        for (const Eigen::MatrixXd& layer : layer_activations) {
            out.hidden_widths.push_back(static_cast<std::size_t>(layer.cols()));
        }
        return out;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\telemetry\telemetry_builder.hpp ===
#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/math/matrix_ops.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace forward_analysis {

struct TelemetryBuilder {
    static ForwardTelemetryTensor build(Eigen::MatrixXd inputs,
                                        Eigen::MatrixXd targets,
                                        Eigen::MatrixXd predictions,
                                        std::vector<Eigen::MatrixXd> layer_activations,
                                        const DistillerConfig& config = DistillerConfig{}) {
        const Eigen::Index n = inputs.rows();
        if (n == 0 || inputs.cols() == 0) {
            throw std::invalid_argument("Telemetry inputs must be a non-empty N x D_in matrix");
        }
        require_rows(targets, n, "targets");
        require_rows(predictions, n, "predictions");
        if (targets.cols() == 0 || predictions.cols() != targets.cols()) {
            throw std::invalid_argument("Telemetry predictions and targets must share a non-zero width");
        }
        if (layer_activations.empty()) {
            throw std::invalid_argument("Telemetry requires at least one hidden layer activation matrix");
        }
        for (std::size_t l = 0; l < layer_activations.size(); ++l) {
            require_rows(layer_activations[l], n, "layer_activations[" + std::to_string(l) + "]");
            if (layer_activations[l].cols() == 0) {
                throw std::invalid_argument("Telemetry layer_activations[" + std::to_string(l) + "] has no units");
            }
            MatrixOps::require_finite(layer_activations[l]);
        }
        MatrixOps::require_finite(inputs);
        MatrixOps::require_finite(targets);
        MatrixOps::require_finite(predictions);

        ForwardTelemetryTensor telemetry;
        telemetry.residuals = predictions - targets;
        telemetry.dead_masks.reserve(layer_activations.size());
        for (const Eigen::MatrixXd& activations : layer_activations) {
            telemetry.dead_masks.push_back(
                (activations.array().abs() <= config.dead_activation_threshold).cast<std::uint8_t>().matrix());
        }
        telemetry.inputs = std::move(inputs);
        telemetry.targets = std::move(targets);
        telemetry.predictions = std::move(predictions);
        telemetry.layer_activations = std::move(layer_activations);
        return telemetry;
    }

private:
    static void require_rows(const Eigen::MatrixXd& m, Eigen::Index n, const std::string& name) {
        if (m.rows() != n) {
            throw std::invalid_argument("Telemetry " + name + " has " + std::to_string(m.rows()) +
                                        " rows, expected " + std::to_string(n));
        }
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\telemetry\telemetry_shape.hpp ===
#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

struct TelemetryShape {
    // Number of samples (rows) in the batch.
    std::size_t num_samples = 0;
    // Number of input coordinates fed to the observed network.
    std::size_t input_dim = 0;
    // Width of each hidden layer, in forward order.
    std::vector<std::size_t> hidden_widths;
    // Number of network outputs (and targets).
    std::size_t output_dim = 0;

    std::size_t total_hidden_units() const noexcept {
        std::size_t total = 0;
        for (std::size_t w : hidden_widths) {
            total += w;
        }
        return total;
    }

    // One channel per input coordinate, hidden unit, and per output its target, prediction and residual.
    std::size_t channel_count() const noexcept {
        return input_dim + total_hidden_units() + 3 * output_dim;
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "N=" << num_samples << " in=" << input_dim << " hidden=[";
        for (std::size_t l = 0; l < hidden_widths.size(); ++l) {
            out << (l > 0 ? "," : "") << hidden_widths[l];
        }
        out << "] out=" << output_dim;
        return out.str();
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\text\csv_field_formatter.hpp ===
#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

// RFC 4180 CSV cells: values containing a comma, quote or newline are quoted with inner quotes doubled.
struct CsvFieldFormatter {
    static std::string escape(const std::string& value) {
        if (value.find_first_of(",\"\r\n") == std::string::npos) {
            return value;
        }
        std::string out = "\"";
        for (char c : value) {
            out += c == '"' ? std::string("\"\"") : std::string(1, c);
        }
        return out + "\"";
    }

    static std::string number(double value) {
        std::ostringstream out;
        out << std::setprecision(8) << value;
        return out.str();
    }

    static std::string row(const std::vector<std::string>& cells) {
        std::string out;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            out += (i > 0 ? "," : "") + escape(cells[i]);
        }
        return out;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\text\english_list_formatter.hpp ===
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

struct EnglishListFormatter {
    // {"a"} -> "a", {"a","b"} -> "a and b", {"a","b","c"} -> "a, b and c".
    static std::string join(const std::vector<std::string>& items) {
        std::string out;
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i > 0) {
                out += (i + 1 == items.size()) ? " and " : ", ";
            }
            out += items[i];
        }
        return out;
    }

    static std::string indices(const std::vector<std::size_t>& values, const std::string& prefix = "") {
        std::vector<std::string> items;
        items.reserve(values.size());
        for (std::size_t v : values) {
            items.push_back(prefix + std::to_string(v));
        }
        return join(items);
    }

    static std::string coordinates(const std::vector<std::size_t>& values) {
        return indices(values, "x");
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\text\json_text_formatter.hpp ===
#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

struct JsonTextFormatter {
    static std::string escape_json(const std::string& input) {
        std::ostringstream out;
        for (char c : input) {
            switch (c) {
                case '"': out << "\\\""; break;
                case '\\': out << "\\\\"; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out << c; break;
            }
        }
        return out.str();
    }

    static std::string format_double(double value, int precision = 2) {
        std::ostringstream out;
        out.setf(std::ios::fixed);
        out << std::setprecision(precision) << value;
        return out.str();
    }

    static std::string format_double_array(const Eigen::VectorXd& values, int precision = 2) {
        std::ostringstream out;
        out << "[";
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            out << (i > 0 ? ", " : "") << format_double(values(i), precision);
        }
        out << "]";
        return out.str();
    }

    static std::string format_index_array(const std::vector<std::size_t>& values) {
        return "[" + join_indices(values, ", ") + "]";
    }

    static std::string join_indices(const std::vector<std::size_t>& values, const std::string& separator) {
        std::ostringstream out;
        for (std::size_t i = 0; i < values.size(); ++i) {
            out << (i > 0 ? separator : "") << values[i];
        }
        return out.str();
    }

    static std::string single_string_field(const std::string& key, const std::string& value) {
        return "{\"" + escape_json(key) + "\": \"" + escape_json(value) + "\"}";
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\text\lexeme_splitter.hpp ===
#pragma once

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Splits prose into word and punctuation lexemes and joins them back with natural spacing.
// A "<...>" span (slot token such as "<DIM:W_1|w>") is always one lexeme.
struct LexemeSplitter {
    static bool is_punctuation(char c) noexcept {
        return c == ',' || c == '.' || c == ';' || c == ':' || c == '(' || c == ')' || c == '/';
    }

    static std::vector<std::string> split(const std::string& text) {
        std::vector<std::string> lexemes;
        std::string current;
        bool in_slot = false;
        const auto flush = [&]() {
            if (!current.empty()) {
                lexemes.push_back(current);
                current.clear();
            }
        };
        for (char c : text) {
            if (in_slot) {
                current.push_back(c);
                if (c == '>') {
                    flush();
                    in_slot = false;
                }
            } else if (c == '<') {
                flush();
                current.push_back(c);
                in_slot = true;
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                flush();
            } else if (is_punctuation(c)) {
                flush();
                lexemes.emplace_back(1, c);
            } else {
                current.push_back(c);
            }
        }
        flush();
        return lexemes;
    }

    // A "." between a lexeme ending in a digit and one starting with a digit is a decimal point ("1.825",
    // "x3-0.5·x7") and is joined without spaces.
    static std::string join(const std::vector<std::string>& lexemes) {
        std::string out;
        bool suppress_next_space = true;
        for (std::size_t i = 0; i < lexemes.size(); ++i) {
            const std::string& lex = lexemes[i];
            const bool closes = lex == "," || lex == "." || lex == ";" || lex == ":" || lex == ")" || lex == "/";
            if (!suppress_next_space && !closes) {
                out += ' ';
            }
            out += lex;
            suppress_next_space = (lex == "(" || lex == "/") || is_decimal_point(lexemes, i);
        }
        return out;
    }

    static bool is_decimal_point(const std::vector<std::string>& lexemes, std::size_t i) {
        return lexemes[i] == "." && i > 0 && i + 1 < lexemes.size() && !lexemes[i - 1].empty() &&
               !lexemes[i + 1].empty() && std::isdigit(static_cast<unsigned char>(lexemes[i - 1].back())) &&
               std::isdigit(static_cast<unsigned char>(lexemes[i + 1].front()));
    }

    static std::string lowercase(const std::string& text) {
        std::string out = text;
        for (char& c : out) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return out;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\training\epoch_record.hpp ===
#pragma once

#include "forward_analysis/training/loss_breakdown.hpp"

#include <cstddef>

namespace forward_analysis {

// Losses of one fine-tuning epoch.
struct EpochRecord {
    // 1-based epoch number.
    std::size_t epoch = 0;
    // Mean training losses over that epoch's steps.
    LossBreakdown train;
    // Held-out losses measured after the epoch.
    LossBreakdown eval;
    // True when this epoch's eval total was the lowest so far (its weights were snapshotted).
    bool best_so_far = false;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\training\finetune_outcome.hpp ===
#pragma once

#include "forward_analysis/training/epoch_record.hpp"

#include <cstddef>
#include <vector>

namespace forward_analysis {

// Result of a fine-tuning phase with early stopping and best-checkpoint restore.
struct FinetuneOutcome {
    // One record per epoch actually run.
    std::vector<EpochRecord> history;
    // 1-based epoch whose weights were restored (lowest eval total); 0 when no epoch ran.
    std::size_t best_epoch = 0;
    // Eval losses of the restored epoch.
    LossBreakdown best_eval;
    // True when training stopped before max_epochs because eval token CE had not decreased for patience epochs.
    bool stopped_early = false;
    // Upper bound on epochs requested.
    std::size_t max_epochs = 0;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\training\loss_breakdown.hpp ===
#pragma once

#include "forward_analysis/config/distiller_config.hpp"

namespace forward_analysis {

struct LossBreakdown {
    // Mean per-token cross-entropy of the decoder over the four observation fields (0 while the decoder is frozen).
    double token_cross_entropy = 0.0;
    // Aux1 observation-descriptor regression loss.
    double aux1_observation_loss = 0.0;
    // Aux2 eigen-sign loss.
    double aux2_eigen_sign_loss = 0.0;
    // token_cross_entropy + aux1_weight * aux1 + aux2_weight * aux2.
    double total_loss = 0.0;

    static LossBreakdown combine(double ce_loss,
                                 double aux1_loss,
                                 double aux2_loss,
                                 const DistillerConfig& config) noexcept {
        LossBreakdown result;
        result.token_cross_entropy = ce_loss;
        result.aux1_observation_loss = aux1_loss;
        result.aux2_eigen_sign_loss = aux2_loss;
        result.total_loss = ce_loss + (config.aux1_observation_weight * aux1_loss) +
                            (config.aux2_eigen_sign_weight * aux2_loss);
        return result;
    }
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\training\split_metric_totals.hpp ===
#pragma once

#include <cstddef>
#include <vector>

namespace forward_analysis {

// Running sums of distillation metrics for one split (e.g. "train", "validation", "ood"); divided by samples when reported.
struct SplitMetricTotals {
    // Records accumulated.
    std::size_t samples = 0;
    // Sum of per-record mean word F1 of the raw decoder text.
    double decoder_f1 = 0.0;
    // Sum of per-record mean word F1 of the narrator text.
    double narrated_f1 = 0.0;
    // Sum of per-record mean word F1 of the final output text.
    double output_f1 = 0.0;
    // Decoder fields that passed the consistency check.
    std::size_t consistent_fields = 0;
    // Narrator fields that passed the consistency check.
    std::size_t narrated_consistent_fields = 0;
    // Final output fields that passed the consistency check.
    std::size_t output_consistent_fields = 0;
    // Ground-truth fields that passed the consistency check.
    std::size_t ground_truth_consistent_fields = 0;
    // Per field: sum of decoder word F1.
    std::vector<double> field_decoder_f1;
    // Per field: decoder fields that passed the consistency check.
    std::vector<std::size_t> field_decoder_consistent;
    // Eigen-sign components predicted correctly.
    std::size_t sign_matches = 0;
    // Eigen-sign components compared.
    std::size_t sign_total = 0;
    // Per eigen-sign component: records where that component was predicted correctly.
    std::vector<std::size_t> sign_component_matches;
    // Records whose whole eigen-sign vector was predicted correctly.
    std::size_t sign_vector_matches = 0;
    // Sum of per-record descriptor mean absolute error.
    double descriptor_mae = 0.0;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\training\training_run_artifact_writer.hpp ===
#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/analysis/observation_json_writer.hpp"
#include "forward_analysis/bridge/distillation_result.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/dataset/dataset_shape_summary.hpp"
#include "forward_analysis/dataset/training_sample.hpp"
#include "forward_analysis/decoder/field_delexicalizer.hpp"
#include "forward_analysis/description/observation_consistency.hpp"
#include "forward_analysis/description/observation_similarity.hpp"
#include "forward_analysis/text/csv_field_formatter.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"
#include "forward_analysis/training/loss_breakdown.hpp"
#include "forward_analysis/training/split_metric_totals.hpp"
#include "forward_analysis/training/training_run_settings.hpp"

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace forward_analysis {

// Writes one training run into its own folder:
//   run.json          settings, config, dataset shapes, per-split aggregate metrics, file manifest
//   loss_history.csv  phase, epoch, train and held-out losses per epoch
//   samples.csv       one row per record with shape, F1, consistency, eigen-sign and descriptor metrics
//   fields.csv        one row per record x field with consistency verdict, violation, F1 and token count
//   samples/<id>.json full detail per record: measured facts, descriptor and signs (target vs predicted),
//                     decoder text, narrated text, final text, ground truth, latents
class TrainingRunArtifactWriter {
public:
    // write_sample_files false skips samples/<id>.json (the CSVs and run.json are always written).
    TrainingRunArtifactWriter(const std::filesystem::path& root, const std::string& run_name, bool write_sample_files = true)
        : directory_((root / run_name).lexically_normal()), write_sample_files_(write_sample_files) {
        std::filesystem::create_directories(write_sample_files_ ? directory_ / "samples" : directory_);
    }

    // "run_YYYYMMDD_HHMMSS" in local time, so successive runs never overwrite each other.
    static std::string timestamped_name() {
        const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm local{};
#if defined(_WIN32)
        localtime_s(&local, &now);
#else
        localtime_r(&now, &local);
#endif
        std::ostringstream out;
        out << "run_" << std::put_time(&local, "%Y%m%d_%H%M%S");
        return out.str();
    }

    const std::filesystem::path& directory() const noexcept { return directory_; }

    void record_epoch(const std::string& phase, std::size_t epoch, const LossBreakdown& train, const LossBreakdown& eval) {
        epoch_rows_.push_back(CsvFieldFormatter::row(
            {phase, std::to_string(epoch), CsvFieldFormatter::number(train.token_cross_entropy),
             CsvFieldFormatter::number(train.aux1_observation_loss), CsvFieldFormatter::number(train.aux2_eigen_sign_loss),
             CsvFieldFormatter::number(train.total_loss), CsvFieldFormatter::number(eval.token_cross_entropy),
             CsvFieldFormatter::number(eval.aux1_observation_loss), CsvFieldFormatter::number(eval.aux2_eigen_sign_loss),
             CsvFieldFormatter::number(eval.total_loss)}));
    }

    // binding: the SlotBinding the decoder was trained with (for the target slot text in samples/<id>.json).
    void record_sample(const std::string& split,
                       const TrainingSample& sample,
                       const DistillationResult& result,
                       SlotBinding binding = SlotBinding::kShapeFacts) {
        const ObservationText& truth = sample.ground_truth_targets.observations;
        const ObservationText& narrated = result.narrated_observations;
        const Eigen::VectorXd target_descriptor = ObservationDescriptor::encode(sample.measured);
        const Eigen::VectorXd& target_signs = sample.ground_truth_targets.eigen_signs;
        SplitMetricTotals& totals = totals_[split];
        totals.sign_component_matches.resize(static_cast<std::size_t>(target_signs.size()), 0);
        totals.field_decoder_f1.resize(ObservationText::kFieldCount, 0.0);
        totals.field_decoder_consistent.resize(ObservationText::kFieldCount, 0);

        std::size_t sign_matches = 0;
        for (Eigen::Index i = 0; i < std::min(target_signs.size(), result.predicted_eigen_signs.size()); ++i) {
            const bool match = target_signs(i) == result.predicted_eigen_signs(i);
            sign_matches += match ? 1 : 0;
            totals.sign_component_matches[static_cast<std::size_t>(i)] += match ? 1 : 0;
        }
        totals.sign_vector_matches += sign_matches == static_cast<std::size_t>(target_signs.size()) ? 1 : 0;
        const double descriptor_mae =
            (result.predicted_observation_descriptor - target_descriptor).cwiseAbs().mean();

        std::size_t consistent_fields = 0;
        std::size_t decoder_tokens = 0;
        nlohmann::json fields = nlohmann::json::object();
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            const std::string& name = ObservationText::field_names()[f];
            const bool filled = result.decoder_slots_filled[f];
            const std::string violation =
                filled ? ObservationConsistency::violation(result.decoder_observations.field(f), result.measured)
                       : std::string("a slot names a quantity this network does not have");
            const double decoder_f1 = ObservationSimilarity::field_f1(result.decoder_observations.field(f), truth.field(f));
            const double output_f1 = ObservationSimilarity::field_f1(result.observations.field(f), truth.field(f));
            const double narrated_f1 = ObservationSimilarity::field_f1(narrated.field(f), truth.field(f));
            const bool consistent = result.decoder_field_consistent[f];
            const std::string narrated_violation = ObservationConsistency::violation(narrated.field(f), result.measured);
            const std::string truth_violation = ObservationConsistency::violation(truth.field(f), sample.measured);
            consistent_fields += consistent ? 1 : 0;
            decoder_tokens += result.decoder_token_counts[f];
            totals.narrated_consistent_fields += narrated_violation.empty() ? 1 : 0;
            totals.output_consistent_fields +=
                ObservationConsistency::verify_field(result.observations.field(f), result.measured) ? 1 : 0;
            totals.ground_truth_consistent_fields += truth_violation.empty() ? 1 : 0;
            totals.field_decoder_f1[f] += decoder_f1;
            totals.field_decoder_consistent[f] += consistent ? 1 : 0;

            fields[name] = {{"target_slot_text",
                             write_sample_files_
                                 ? LexemeSplitter::join(FieldDelexicalizer::delexicalize(truth.field(f), sample.measured, binding))
                                 : std::string()},
                            {"decoder_slot_text", result.decoder_slot_observations.field(f)},
                            {"decoder_slots_filled", filled},
                            {"decoder_text", result.decoder_observations.field(f)},
                            {"decoder_consistent", consistent},
                            {"decoder_violation", violation},
                            {"decoder_tokens", result.decoder_token_counts[f]},
                            {"narrated_text", narrated.field(f)},
                            {"narrated_violation", narrated_violation},
                            {"ground_truth_violation", truth_violation},
                            {"output_text", result.observations.field(f)},
                            {"output_source", consistent ? "decoder" : "narrator"},
                            {"ground_truth", truth.field(f)},
                            {"decoder_f1", decoder_f1},
                            {"narrated_f1", narrated_f1},
                            {"output_f1", output_f1}};
            field_rows_.push_back(CsvFieldFormatter::row(
                {split, sample.sample_id, name, filled ? "1" : "0", consistent ? "1" : "0",
                 consistent ? "decoder" : "narrator", violation,
                 std::to_string(result.decoder_token_counts[f]), CsvFieldFormatter::number(decoder_f1),
                 CsvFieldFormatter::number(narrated_f1), CsvFieldFormatter::number(output_f1), truth_violation}));
        }

        const double decoder_f1 = ObservationSimilarity::mean_f1(result.decoder_observations, truth);
        const double narrated_f1 = ObservationSimilarity::mean_f1(narrated, truth);
        const double output_f1 = ObservationSimilarity::mean_f1(result.observations, truth);
        const std::string shape = sample.telemetry.shape().to_string();
        sample_rows_.push_back(CsvFieldFormatter::row(
            {split, sample.sample_id, shape, std::to_string(result.channel_count), CsvFieldFormatter::number(decoder_f1),
             CsvFieldFormatter::number(narrated_f1), CsvFieldFormatter::number(output_f1), std::to_string(consistent_fields),
             std::to_string(ObservationText::kFieldCount - consistent_fields), std::to_string(decoder_tokens),
             std::to_string(sign_matches), CsvFieldFormatter::number(descriptor_mae)}));

        totals.samples += 1;
        totals.decoder_f1 += decoder_f1;
        totals.narrated_f1 += narrated_f1;
        totals.output_f1 += output_f1;
        totals.consistent_fields += consistent_fields;
        totals.sign_matches += sign_matches;
        totals.sign_total += static_cast<std::size_t>(target_signs.size());
        totals.descriptor_mae += descriptor_mae;

        const nlohmann::json detail = {
            {"split", split},
            {"sample_id", sample.sample_id},
            {"shape", shape},
            {"channel_count", result.channel_count},
            {"measured_observations", ObservationJsonWriter::to_json(result.measured)},
            {"descriptor", {{"target", vector_json(target_descriptor)},
                            {"predicted", vector_json(result.predicted_observation_descriptor)},
                            {"mean_absolute_error", descriptor_mae}}},
            {"eigen_signs", {{"target", vector_json(target_signs)},
                             {"predicted", vector_json(result.predicted_eigen_signs)},
                             {"matches", sign_matches}}},
            {"mean_f1", {{"decoder", decoder_f1}, {"narrated", narrated_f1}, {"output", output_f1}}},
            {"fields", fields},
            {"latent_embeddings", matrix_json(result.latent_embeddings)}};
        if (write_sample_files_) {
            write_text(directory_ / "samples" / (sample.sample_id + ".json"), detail.dump(2));
        }
    }

    void finish(const TrainingRunSettings& settings,
                const DistillerConfig& config,
                const std::vector<TrainingSample>& train,
                const std::vector<TrainingSample>& eval) const {
        finish(settings, config, {{"train", &train}, {"eval", &eval}});
    }

    // datasets: split name and records for each dataset whose shape ranges go into run.json.
    void finish(const TrainingRunSettings& settings,
                const DistillerConfig& config,
                const std::vector<std::pair<std::string, const std::vector<TrainingSample>*>>& datasets) const {
        write_csv("loss_history.csv",
                  "phase,epoch,train_token_ce,train_aux1,train_aux2,train_total,eval_token_ce,eval_aux1,eval_aux2,eval_total",
                  epoch_rows_);
        write_csv("samples.csv",
                  "split,sample_id,shape,channel_count,decoder_f1,narrated_f1,output_f1,decoder_consistent_fields,"
                  "narrator_fields,decoder_tokens,eigen_sign_matches,descriptor_mae",
                  sample_rows_);
        write_csv("fields.csv",
                  "split,sample_id,field,decoder_slots_filled,decoder_consistent,output_source,decoder_violation,decoder_tokens,decoder_f1,"
                  "narrated_f1,output_f1,ground_truth_violation",
                  field_rows_);

        nlohmann::json metrics = nlohmann::json::object();
        for (const auto& [split, t] : totals_) {
            const double n = static_cast<double>(t.samples);
            const double fields = n * static_cast<double>(ObservationText::kFieldCount);
            nlohmann::json per_field = nlohmann::json::object();
            for (std::size_t f = 0; f < t.field_decoder_f1.size(); ++f) {
                per_field[ObservationText::field_names()[f]] = {
                    {"mean_decoder_f1", t.field_decoder_f1[f] / n},
                    {"decoder_consistent_rate", static_cast<double>(t.field_decoder_consistent[f]) / n}};
            }
            std::vector<double> component_accuracy;
            for (std::size_t c : t.sign_component_matches) {
                component_accuracy.push_back(static_cast<double>(c) / n);
            }
            metrics[split] = {{"samples", t.samples},
                              {"mean_decoder_f1", t.decoder_f1 / n},
                              {"mean_narrated_f1", t.narrated_f1 / n},
                              {"mean_output_f1", t.output_f1 / n},
                              {"decoder_consistent_fields", t.consistent_fields},
                              {"narrated_consistent_fields", t.narrated_consistent_fields},
                              {"output_consistent_fields", t.output_consistent_fields},
                              {"ground_truth_consistent_fields", t.ground_truth_consistent_fields},
                              {"total_fields", t.samples * ObservationText::kFieldCount},
                              {"decoder_consistent_rate", static_cast<double>(t.consistent_fields) / fields},
                              {"output_consistent_rate", static_cast<double>(t.output_consistent_fields) / fields},
                              {"ground_truth_consistent_rate", static_cast<double>(t.ground_truth_consistent_fields) / fields},
                              {"per_field", per_field},
                              {"eigen_sign_accuracy", t.sign_total > 0 ? static_cast<double>(t.sign_matches) /
                                                                             static_cast<double>(t.sign_total)
                                                                       : 0.0},
                              {"eigen_sign_component_accuracy", component_accuracy},
                              {"eigen_sign_vector_accuracy", static_cast<double>(t.sign_vector_matches) / n},
                              {"mean_descriptor_mae", t.descriptor_mae / n}};
        }

        nlohmann::json dataset = nlohmann::json::object();
        for (const auto& [name, records] : datasets) {
            dataset[name] = DatasetShapeSummary::from(*records).to_json();
        }

        const nlohmann::json run = {
            {"run", directory_.filename().string()},
            {"settings", {{"train_path", settings.train_path},
                          {"eval_path", settings.eval_path},
                          {"final_eval_path", settings.final_eval_path},
                          {"trial", settings.trial},
                          {"trial_description", settings.trial_description},
                          {"shuffled_training_text", settings.shuffled_training_text},
                          {"seed", settings.seed},
                          {"pretrain_epochs", settings.pretrain_epochs},
                          {"finetune_epochs", settings.finetune_epochs},
                          {"finetune_epochs_run", settings.finetune_epochs_run},
                          {"best_finetune_epoch", settings.best_finetune_epoch},
                          {"stopped_early", settings.stopped_early},
                          {"encoder_learning_rate", settings.encoder_learning_rate},
                          {"training_seconds", settings.training_seconds},
                          {"vocabulary_size", settings.vocabulary_size},
                          {"decoder_accelerator", settings.decoder_accelerator}}},
            {"config", config_json(config)},
            {"dataset", dataset},
            {"metrics", metrics},
            {"files", {{"loss_history.csv", "per-epoch mean losses on the training records and on the held-out records"},
                       {"samples.csv", "one row per record: shape, word F1 (decoder / narrated / output), consistency, "
                                       "eigen-sign matches, descriptor mean absolute error"},
                       {"fields.csv", "one row per record and field: decoder consistency, first violation, tokens, F1, "
                                      "first ground-truth violation"},
                       {"samples/", write_sample_files_
                                        ? "one JSON per record with measured facts, aux targets vs predictions, all texts, latents"
                                        : "not written for this run"}}}};
        write_text(directory_ / "run.json", run.dump(2));
    }

private:
    static nlohmann::json vector_json(const Eigen::VectorXd& v) {
        return std::vector<double>(v.data(), v.data() + v.size());
    }

    static nlohmann::json matrix_json(const Eigen::MatrixXd& m) {
        nlohmann::json rows = nlohmann::json::array();
        for (Eigen::Index r = 0; r < m.rows(); ++r) {
            rows.push_back(vector_json(m.row(r).transpose()));
        }
        return rows;
    }

    static nlohmann::json config_json(const DistillerConfig& c) {
        return {{"d_model", c.d_model},
                {"num_latents", c.num_latents},
                {"eigen_sign_dim", c.eigen_sign_dim},
                {"decoder_hidden_dim", c.decoder_hidden_dim},
                {"max_field_tokens", c.max_field_tokens},
                {"decoder_learning_rate", c.decoder_learning_rate},
                {"decoder_dropout", c.decoder_dropout},
                {"decoder_output_weight_decay", c.decoder_output_weight_decay},
                {"use_gpu", c.use_gpu},
                {"early_stopping_patience", c.early_stopping_patience},
                {"aux1_observation_weight", c.aux1_observation_weight},
                {"aux2_eigen_sign_weight", c.aux2_eigen_sign_weight},
                {"dead_activation_threshold", c.dead_activation_threshold},
                {"spectral_rank_tolerance", c.spectral_rank_tolerance},
                {"ray_tolerance", c.ray_tolerance},
                {"driver_correlation_threshold", c.driver_correlation_threshold},
                {"flat_output_ratio", c.flat_output_ratio},
                {"numerical_rank_tolerance", c.numerical_rank_tolerance},
                {"narrow_rank_limit", c.narrow_rank_limit},
                {"range_ratio_low", c.range_ratio_low},
                {"range_ratio_high", c.range_ratio_high},
                {"narration_style",
                 c.narration_style == NarrationStyle::kBatchStatistics ? "batch_statistics" : "measured_facts"},
                {"slot_binding", c.slot_binding == SlotBinding::kBatchStatistics ? "batch_statistics" : "shape_facts"}};
    }

    static void write_text(const std::filesystem::path& path, const std::string& text) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot write artifact: " + path.string());
        }
        out << text << "\n";
    }

    void write_csv(const std::string& name, const std::string& header, const std::vector<std::string>& rows) const {
        std::string text = header;
        for (const std::string& row : rows) {
            text += "\n" + row;
        }
        write_text(directory_ / name, text);
    }

    // Folder this run writes into, e.g. artifacts/run_20261013_224800.
    std::filesystem::path directory_;
    // Whether record_sample writes samples/<id>.json.
    bool write_sample_files_ = true;
    // loss_history.csv rows, one per epoch.
    std::vector<std::string> epoch_rows_;
    // samples.csv rows, one per record.
    std::vector<std::string> sample_rows_;
    // fields.csv rows, one per record x field.
    std::vector<std::string> field_rows_;
    // Aggregates per split name.
    std::map<std::string, SplitMetricTotals> totals_;
};

}  // namespace forward_analysis

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\include\forward_analysis\training\training_run_settings.hpp ===
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace forward_analysis {

// How a training run was launched, recorded alongside its artifacts.
struct TrainingRunSettings {
    // JSONL file the model trained on.
    std::string train_path;
    // JSONL file of held-out records used for early stopping and checkpoint selection.
    std::string eval_path;
    // JSONL file evaluated only after training (empty when the run has none).
    std::string final_eval_path;
    // Experiment label, e.g. "B" (empty for demo runs).
    std::string trial;
    // Free-form description of what the trial changes.
    std::string trial_description;
    // True when the training records' observation text was permuted across records (telemetry control).
    bool shuffled_training_text = false;
    // Seed passed to DualObjectiveDistillationBridge.
    std::uint64_t seed = 20261013ULL;
    // Step 1 epochs (auxiliary losses only, decoder frozen).
    std::size_t pretrain_epochs = 0;
    // Step 2 upper bound on epochs (CE + Aux1 + Aux2, everything trains).
    std::size_t finetune_epochs = 0;
    // Step 2 epochs actually run before early stopping (or finetune_epochs when it never triggered).
    std::size_t finetune_epochs_run = 0;
    // Step 2 epoch whose weights were restored (lowest held-out total loss).
    std::size_t best_finetune_epoch = 0;
    // True when step 2 stopped early on held-out token cross-entropy.
    bool stopped_early = false;
    // SGD step size for the adapter, Perceiver and heads.
    double encoder_learning_rate = 0.0;
    // Wall-clock seconds spent in steps 1 and 2.
    double training_seconds = 0.0;
    // Decoder vocabulary size after prepare_vocabulary.
    std::size_t vocabulary_size = 0;
    // Where decoder training passes ran: the OpenCL device description, or "cpu".
    std::string decoder_accelerator = "cpu";
};

}  // namespace forward_analysis

