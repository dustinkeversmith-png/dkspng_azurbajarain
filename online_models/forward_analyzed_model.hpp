#pragma once

// Canonical online model definition: ForwardAnalyzedModel (architecture "forward_analyzed_mlp").
// Uses offline_models/forward_analysis (TelemetryBuilder, ObservationExtractor, MeasuredFactsNarration,
// DualObjectiveDistillationBridge) to analyze its network during forward evaluation and append those
// analysis results to the prompt telemetry, checkpoint document, and sliding journal history.

#include "forward_analysis/analysis/observation_extractor.hpp"
#include "forward_analysis/bridge/dual_objective_distillation_bridge.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/description/measured_facts_narration.hpp"
#include "forward_analysis/description/observation_narrator.hpp"
#include "forward_analysis/telemetry/telemetry_builder.hpp"

#include "stevejob/messaging/sjc1_scheme.hpp"
#include "stevejob/models/model_template.hpp"
#include "stevejob/models/tournament_prompt.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

class ForwardAnalyzedModel : public ModelTemplate {
public:
    struct ForwardAnalysisSnapshot {
        forward_analysis::DistillationResult distilled;
        forward_analysis::ObservationText concise_text;
        forward_analysis::ObservationText batch_text;
        std::size_t dead_units_l1 = 0;
        std::size_t dead_units_l2 = 0;
        std::size_t total_dead_units = 0;
        std::size_t numerical_rank_l1 = 0;
        std::size_t numerical_rank_l2 = 0;
        double mean_effective_rank = 0.0;
        double strongest_association = 0.0;
        double sign_agreement_ratio = 0.0;
        double target_correlation = 0.0;
        double range_ratio = 0.0;
        double encoder_tokens = 0.0;
        std::string eigen_signs_token;
        std::string summary_token;
    };

    class ForwardAnalyzedOutput : public messaging::IOutputScheme {
    public:
        static const ForwardAnalyzedOutput& instance() {
            static const ForwardAnalyzedOutput scheme;
            return scheme;
        }

        messaging::EncodingDescriptor get_descriptor() const override {
            return messaging::EncodingDescriptor{
                "forward_analyzed_output",
                "Row-major fixed-4 floats joined by ';', paired with offline forward_analysis telemetry.",
                "Each row is one sample prediction y_hat = head(layer2(layer1(x))).",
                "Values are finite floats analyzed against sin/cos targets by forward_analysis."};
        }

        std::string serialize_output(const std::vector<std::vector<double>>& outputs) const override {
            std::ostringstream out;
            out.setf(std::ios::fixed);
            out.precision(4);
            for (std::size_t row = 0; row < outputs.size(); ++row) {
                if (row > 0) {
                    out << ';';
                }
                for (std::size_t col = 0; col < outputs[row].size(); ++col) {
                    if (col > 0) {
                        out << ' ';
                    }
                    out << outputs[row][col];
                }
            }
            return out.str();
        }

        std::vector<std::vector<double>> deserialize_output(const std::string& token,
                                                            std::size_t rows,
                                                            std::size_t cols) const override {
            std::vector<std::vector<double>> outputs;
            std::string pending = token;
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t split = pending.find(';');
                const std::string piece = pending.substr(0, split);
                std::istringstream in(piece);
                std::vector<double> values;
                double val = 0.0;
                while (in >> val) {
                    values.push_back(val);
                }
                if (values.size() != cols) {
                    throw std::invalid_argument("forward_analyzed_output row has the wrong column count");
                }
                outputs.push_back(std::move(values));
                if (split == std::string::npos) {
                    pending.clear();
                } else {
                    pending = pending.substr(split + 1);
                }
            }
            if (outputs.size() != rows) {
                throw std::invalid_argument("forward_analyzed_output token has the wrong row count");
            }
            return outputs;
        }
    };

    class AnalysisScalarCodec : public messaging::IEncodingScheme {
    public:
        messaging::EncodingDescriptor get_descriptor() const override { return info_; }

        std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const override {
            if (matrix.size() != 1) {
                throw std::invalid_argument(info_.scheme_name + " encodes one row");
            }
            return serialize_vector(matrix[0], 0, matrix[0].size());
        }

        std::vector<std::vector<double>> deserialize_matrix(const std::string& token,
                                                            std::size_t rows,
                                                            std::size_t cols) const override {
            if (rows != 1) {
                throw std::invalid_argument(info_.scheme_name + " decodes one row");
            }
            return {deserialize_vector(token, cols)};
        }

        std::string serialize_vector(const std::vector<double>& values,
                                     std::size_t start,
                                     std::size_t count) const override {
            return messaging::BiasFloatScheme::instance().serialize_vector(values, start, count);
        }

        std::vector<double> deserialize_vector(const std::string& text, std::size_t count) const override {
            return messaging::BiasFloatScheme::instance().deserialize_vector(text, count);
        }

    protected:
        explicit AnalysisScalarCodec(messaging::EncodingDescriptor info) : info_(std::move(info)) {}

    private:
        messaging::EncodingDescriptor info_;
    };

    class AnalysisTextCodec : public messaging::IEncodingScheme {
    public:
        messaging::EncodingDescriptor get_descriptor() const override { return info_; }

        std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const override {
            (void)matrix;
            throw std::invalid_argument(info_.scheme_name + " is a textual forward_analysis observation codec");
        }

        std::vector<std::vector<double>> deserialize_matrix(const std::string& token,
                                                            std::size_t rows,
                                                            std::size_t cols) const override {
            (void)token;
            (void)rows;
            (void)cols;
            throw std::invalid_argument(info_.scheme_name + " is a textual forward_analysis observation codec");
        }

        std::string serialize_vector(const std::vector<double>& values,
                                     std::size_t start,
                                     std::size_t count) const override {
            return messaging::BiasFloatScheme::instance().serialize_vector(values, start, count);
        }

        std::vector<double> deserialize_vector(const std::string& text, std::size_t count) const override {
            return messaging::BiasFloatScheme::instance().deserialize_vector(text, count);
        }

    protected:
        explicit AnalysisTextCodec(messaging::EncodingDescriptor info) : info_(std::move(info)) {}

    private:
        messaging::EncodingDescriptor info_;
    };

    class DeadUnitsCodec : public AnalysisScalarCodec {
    public:
        static const DeadUnitsCodec& instance() {
            static const DeadUnitsCodec scheme;
            return scheme;
        }

    private:
        DeadUnitsCodec()
            : AnalysisScalarCodec(
                  {"fa_dead_units_codec",
                   "Count of hidden units whose peak activation across the batch is <= 1e-6.",
                   "fa_dead_units = dead_units(layer1) + dead_units(layer2).",
                   "When dead units > 0, raise biases or adjust A/B signs to revive silent ReLU pathways."}) {}
    };

    class EffectiveRankCodec : public AnalysisScalarCodec {
    public:
        static const EffectiveRankCodec& instance() {
            static const EffectiveRankCodec scheme;
            return scheme;
        }

    private:
        EffectiveRankCodec()
            : AnalysisScalarCodec(
                  {"fa_effective_rank_codec",
                   "Mean spectral effective rank across hidden activation manifolds.",
                   "fa_effective_rank = mean_l count(sigma_i(H_l) >= 0.05 * sigma_0(H_l)).",
                   "Low rank indicates representational collapse onto a ray; diversify A/B codes."}) {}
    };

    class AssociationCodec : public AnalysisScalarCodec {
    public:
        static const AssociationCodec& instance() {
            static const AssociationCodec scheme;
            return scheme;
        }

    private:
        AssociationCodec()
            : AnalysisScalarCodec(
                  {"fa_input_association_codec",
                   "Peak absolute Pearson correlation between inputs and final hidden units in [0, 1].",
                   "fa_strongest_association = max_{c,k} |Pearson(X[:,c], H_last[:,k])|.",
                   "Higher association means hidden features strongly track primary input drivers."}) {}
    };

    class TargetCorrelationCodec : public AnalysisScalarCodec {
    public:
        static const TargetCorrelationCodec& instance() {
            static const TargetCorrelationCodec scheme;
            return scheme;
        }

    private:
        TargetCorrelationCodec()
            : AnalysisScalarCodec(
                  {"fa_target_correlation_codec",
                   "Pearson correlation between network predictions and targets in [-1, 1].",
                   "fa_target_correlation = Pearson(y_hat, y_target).",
                   "Push correlation toward +1.0 while keeping range_ratio near 1.0."}) {}
    };

    class SignAgreementCodec : public AnalysisScalarCodec {
    public:
        static const SignAgreementCodec& instance() {
            static const SignAgreementCodec scheme;
            return scheme;
        }

    private:
        SignAgreementCodec()
            : AnalysisScalarCodec(
                  {"fa_sign_agreement_codec",
                   "Fraction of batch samples where prediction sign matches target sign in [0, 1].",
                   "fa_sign_agreement = sign_agreements / sign_entries.",
                   "Low sign agreement means head or hidden signs are inverted."}) {}
    };

    class ObservationFieldCodec : public AnalysisTextCodec {
    public:
        static const ObservationFieldCodec& instance() {
            static const ObservationFieldCodec scheme;
            return scheme;
        }

    private:
        ObservationFieldCodec()
            : AnalysisTextCodec(
                  {"forward_analysis_distiller",
                   "Verified qualitative diagnostic sentence from offline_models/forward_analysis.",
                   "observations[f] = verify(decoder(Z, d_cond)) ? decoder_text : MeasuredFactsNarration(telemetry).",
                   "Use pathway_activity, feature_sensitivity, representational_structure, and functional_mapping to guide slice edits."}) {}
    };

    struct ForwardAnalyzedJournalScheme {
        storage::MetricField origin{"origin", storage::MetricType::Text, std::string("forward_analyzed_mlp"),
                                    "Architecture that opened this forward-analyzed checkpoint."};
        storage::MetricField delta{"delta", storage::MetricType::Number, 0.0,
                                   "Signed loss change against the last committed checkpoint."};
        storage::MetricField winner{"winner", storage::MetricType::Text, std::string("seed"),
                                    "Candidate slot that scored the lowest finite loss."};
        storage::MetricField seeded{"status", storage::MetricType::Text, std::string("seed"),
                                    "Trial result: seed, committed_natively, rejected, or rejected_divergent."};
        storage::MetricField step{"winner_step", storage::MetricType::Token, std::string(""),
                                  "Winning gamma and integer distance, written as g=<gamma> d=<distance>."};
        storage::MetricField slice{"slice", storage::MetricType::Text, std::string(""),
                                   "Slice id this trial edited."};
        storage::MetricField fa_dead_units{"fa_dead_units", storage::MetricType::Number, 0.0,
                                           "Forward-analysis count of dead hidden units across layers."};
        storage::MetricField fa_effective_rank{"fa_effective_rank", storage::MetricType::Number, 0.0,
                                               "Forward-analysis mean spectral effective rank across hidden layers."};
        storage::MetricField fa_target_corr{"fa_target_corr", storage::MetricType::Number, 0.0,
                                            "Forward-analysis Pearson correlation between predictions and targets."};
        storage::MetricField fa_sign_agreement{"fa_sign_agreement", storage::MetricType::Number, 0.0,
                                               "Forward-analysis fraction of samples with matching prediction/target sign."};
        storage::MetricField fa_summary{"fa_summary", storage::MetricType::Token, std::string(""),
                                        "Compact offline forward_analysis diagnostic summary token for the network state."};
        storage::MetricField fa_pathway{"fa_pathway", storage::MetricType::Text, std::string(""),
                                        "Offline forward_analysis pathway_activity narration for this network state."};
        storage::MetricField fa_mapping{"fa_mapping", storage::MetricType::Text, std::string(""),
                                        "Offline forward_analysis functional_mapping narration for this network state."};

        JournalConfig config() const {
            JournalConfig journal;
            journal.capacity = 6;
            journal.task = "forward_analyzed_regression";
            journal.description =
                "Sliding window of tournament trials augmented with offline_models/forward_analysis diagnostics. "
                "A checkpoint write requires loss to fall by more than 0.0001.";
            return journal;
        }

        std::string prompt_text() const {
            return "Journal scheme: task forward_analyzed_regression. Includes delta, winner, status, plus offline "
                   "forward_analysis metrics (fa_dead_units, fa_effective_rank, fa_target_corr, fa_sign_agreement, "
                   "fa_summary, fa_pathway, fa_mapping).";
        }

        storage::MetricField with_number(storage::MetricField field, double value) const {
            field.value = value;
            return field;
        }

        storage::MetricField with_text(storage::MetricField field, const std::string& value) const {
            field.value = value;
            return field;
        }
    };

    ForwardAnalyzedModel() : net_("ForwardAnalyzedModel") {
        config_.max_candidates = 3;
        config_.max_parameters = 22;
        config_.max_input_rows = 8;
        config_.max_input_cols = 4;
        config_.visible_history_count = 3;
        config_.int_bound = 25;
        config_.min_commit_drop = 1e-4;
        config_.divergence_threshold = 100.0;
        net_.architecture = "forward_analyzed_mlp";
        net_.add_layer("layer1", 4, 8, ReLUActivation{})
            .add_layer("layer2", 8, 4, ReLUActivation{})
            .add_layer("head", 4, 1, LinearActivation{});
        net_.initialize_default_weights();
        install(net_);
    }

    void adopt(NeuralNetwork model) {
        model.architecture = "forward_analyzed_mlp";
        install(model);
        net_ = std::move(model);
    }

    NeuralNetwork& network() override { return net_; }
    const NeuralNetwork& network() const override { return net_; }

    ModelEvaluation evaluate_model(int sample_count = 0) const override {
        return evaluate_with_forward_analysis(net_, resolve_samples(sample_count));
    }

    ModelEvaluation score_patch(const NeuralNetwork& candidate_net, int sample_count = 0) const override {
        return evaluate_with_forward_analysis(candidate_net, resolve_samples(sample_count));
    }

    ForwardAnalysisSnapshot analyze_network(const NeuralNetwork& candidate, int sample_count = 0) const {
        const int count = resolve_samples(sample_count);
        const forward_analysis::ForwardTelemetryTensor telemetry = build_forward_telemetry(candidate, count);
        forward_analysis::DistillerConfig fa_config;
        fa_config.narration_style = forward_analysis::NarrationStyle::kBatchStatistics;

        ForwardAnalysisSnapshot snap;
        snap.distilled = shared_bridge().distill_telemetry(telemetry);
        snap.concise_text = forward_analysis::ObservationNarrator::describe(snap.distilled.measured);
        snap.batch_text = forward_analysis::MeasuredFactsNarration::describe(snap.distilled.measured, fa_config);

        const auto& obs = snap.distilled.measured;
        if (!obs.layers.empty()) {
            snap.dead_units_l1 = obs.layers[0].dead_units.size();
            snap.numerical_rank_l1 = obs.layers[0].numerical_rank;
        }
        if (obs.layers.size() > 1) {
            snap.dead_units_l2 = obs.layers[1].dead_units.size();
            snap.numerical_rank_l2 = obs.layers[1].numerical_rank;
        }
        double eff_sum = 0.0;
        for (const auto& layer : obs.layers) {
            snap.total_dead_units += layer.dead_units.size();
            eff_sum += static_cast<double>(layer.effective_rank);
        }
        snap.mean_effective_rank =
            obs.layers.empty() ? 0.0 : eff_sum / static_cast<double>(obs.layers.size());
        snap.strongest_association = obs.strongest_association;
        snap.sign_agreement_ratio =
            obs.sign_entries == 0 ? 0.0 : static_cast<double>(obs.sign_agreements) / static_cast<double>(obs.sign_entries);
        if (!obs.outputs.empty()) {
            snap.target_correlation = std::isfinite(obs.outputs.front().correlation) ? obs.outputs.front().correlation : 0.0;
            snap.range_ratio = std::isfinite(obs.outputs.front().range_ratio) ? obs.outputs.front().range_ratio : 0.0;
        }
        snap.encoder_tokens = static_cast<double>(snap.distilled.encoder_token_count);

        std::ostringstream eig;
        for (Eigen::Index i = 0; i < obs.eigen_signs.size(); ++i) {
            if (i > 0) {
                eig << ",";
            }
            eig << (obs.eigen_signs(i) >= 0.0 ? "+1" : "-1");
        }
        snap.eigen_signs_token = eig.str();

        std::ostringstream summary;
        summary.setf(std::ios::fixed);
        summary.precision(3);
        summary << "dead=" << snap.dead_units_l1 << "/" << snap.dead_units_l2
                << "|rank=" << snap.numerical_rank_l1 << "/" << snap.numerical_rank_l2
                << "|assoc=" << snap.strongest_association
                << "|corr=" << snap.target_correlation
                << "|range=" << snap.range_ratio
                << "|sign=" << snap.sign_agreement_ratio
                << "|narrow=" << (obs.narrow_representation ? 1 : 0);
        snap.summary_token = summary.str();
        return snap;
    }

    std::vector<TelemetrySignal> derive_training_telemetry() const override {
        double count = 0.0;
        double magnitude_sum = 0.0;
        double square = 0.0;
        double peak = 0.0;
        double sparse = 0.0;
        for (const LayerSpec& layer : network().layers) {
            for (const std::vector<double>& row : layer.weights) {
                for (double value : row) {
                    const double magnitude = std::abs(value);
                    magnitude_sum += magnitude;
                    square += value * value;
                    peak = std::max(peak, magnitude);
                    if (magnitude < 0.05) {
                        sparse += 1.0;
                    }
                    count += 1.0;
                }
            }
        }
        const double mean = count == 0.0 ? 0.0 : magnitude_sum / count;
        const ForwardAnalysisSnapshot fa = analyze_network(network());

        return {
            {core::DescribedField{
                 "sparsity",
                 core::FieldType::Scalar,
                 count == 0.0 ? 0.0 : sparse / count,
                 "telemetry",
                 "Fraction of regression weights whose magnitude is below 0.05.",
                 "High sparsity is not a lower loss. Do not chase it with a large gamma."},
             nullptr},
            {core::DescribedField{
                 "variance",
                 core::FieldType::Scalar,
                 count == 0.0 ? 0.0 : square / count,
                 "telemetry",
                 "Mean square of the network weights.",
                 "A jump means the slice scale moved. Stay inside the gamma band."},
             nullptr},
            {core::DescribedField{
                 "conditioning",
                 core::FieldType::Scalar,
                 peak / std::max(mean, 1e-6),
                 "telemetry",
                 "Peak weight magnitude divided by the mean magnitude.",
                 "High conditioning means a few weights dominate. Take small integer steps on those coordinates."},
             nullptr},
            {core::DescribedField{
                 "fa_dead_units",
                 core::FieldType::Scalar,
                 static_cast<double>(fa.total_dead_units),
                 "telemetry",
                 "Total silent hidden units measured by offline_models/forward_analysis across hidden layers.",
                 "If dead units > 0, adjust biases or A/B signs to reactivate dead ReLU coordinates."},
             &DeadUnitsCodec::instance()},
            {core::DescribedField{
                 "fa_effective_rank",
                 core::FieldType::Scalar,
                 fa.mean_effective_rank,
                 "telemetry",
                 "Mean spectral effective rank across hidden layers from forward_analysis.",
                 "Keep effective rank >= 2 so hidden representations do not collapse onto a single ray."},
             &EffectiveRankCodec::instance()},
            {core::DescribedField{
                 "fa_strongest_association",
                 core::FieldType::Scalar,
                 fa.strongest_association,
                 "telemetry",
                 "Strongest input-to-final-hidden Pearson correlation from forward_analysis.",
                 "Higher association indicates stronger alignment between input drivers and hidden features."},
             &AssociationCodec::instance()},
            {core::DescribedField{
                 "fa_target_correlation",
                 core::FieldType::Scalar,
                 fa.target_correlation,
                 "telemetry",
                 "Pearson correlation between predictions and targets measured by forward_analysis.",
                 "Increase target correlation toward 1.0 while shrinking MSE."},
             &TargetCorrelationCodec::instance()},
            {core::DescribedField{
                 "fa_sign_agreement",
                 core::FieldType::Scalar,
                 fa.sign_agreement_ratio,
                 "telemetry",
                 "Fraction of samples where prediction sign matches target sign.",
                 "When sign agreement is low, prioritize correcting output amplitude and bias offset."},
             &SignAgreementCodec::instance()},
            {core::DescribedField{
                 "fa_summary",
                 core::FieldType::Token,
                 fa.summary_token,
                 "telemetry",
                 "Compact forward_analysis state token (dead units, numerical ranks, association, correlation, range ratio, sign agreement).",
                 "Use this compact readout alongside the journal's fa_summary history to see structural changes per trial."},
             &ObservationFieldCodec::instance()},
            {core::DescribedField{
                 "fa_pathway_activity",
                 core::FieldType::Token,
                 fa.concise_text.pathway_activity + " | " + fa.batch_text.pathway_activity,
                 "telemetry",
                 "Offline forward_analysis pathway_activity observation.",
                 "Diagnoses silent, gated, or ray-collapsed units across layer1 and layer2."},
             &ObservationFieldCodec::instance()},
            {core::DescribedField{
                 "fa_feature_sensitivity",
                 core::FieldType::Token,
                 fa.concise_text.feature_sensitivity + " | " + fa.batch_text.feature_sensitivity,
                 "telemetry",
                 "Offline forward_analysis feature_sensitivity observation.",
                 "Identifies which input coordinates drive the leading hidden principal axes."},
             &ObservationFieldCodec::instance()},
            {core::DescribedField{
                 "fa_representational_structure",
                 core::FieldType::Token,
                 fa.concise_text.representational_structure + " | " + fa.batch_text.representational_structure,
                 "telemetry",
                 "Offline forward_analysis representational_structure observation.",
                 "Describes activation manifold dimensionality and numerical ranks."},
             &ObservationFieldCodec::instance()},
            {core::DescribedField{
                 "fa_functional_mapping",
                 core::FieldType::Token,
                 fa.concise_text.functional_mapping + " | " + fa.batch_text.functional_mapping,
                 "telemetry",
                 "Offline forward_analysis functional_mapping observation.",
                 "Diagnoses whether predictions damp, invert, flatten, or track the targets."},
             &ObservationFieldCodec::instance()},
        };
    }

    core::DescribedField derive_gradient_probe(const std::string& slice_id) const override {
        return core::DescribedField{
            "gradient",
            core::FieldType::Token,
            probe_marks(slice_id),
            "gradient_probe",
            "Comma-separated directional marks: +2, +1, 0, -1, -2. One mark per A code, then one per B code.",
            "'+2' steep ascent: decrease that code by 6 to 12. '+1' gentle ascent: decrease it by 2 to 5. "
            "'-2' steep descent: increase that code by 6 to 12. '-1' gentle descent: increase it by 2 to 5. "
            "'0' is a flat dead zone. Combine with forward_analysis pathway_activity and functional_mapping to scale gamma."};
    }

    const messaging::IEncodingScheme& get_scheme_for_slice(const SliceRef& slice) const override {
        (void)slice;
        return messaging::TotrRank1Scheme::instance();
    }

    const messaging::IEncodingScheme& get_bias_scheme(const SliceRef& slice) const override {
        (void)slice;
        return messaging::BiasDeltaScheme::instance();
    }

    const messaging::IOutputScheme& get_output_scheme() const override { return ForwardAnalyzedOutput::instance(); }

    CandidateScheme candidate_scheme() const override { return CandidateScheme{}; }

    std::string candidate_roles(const storage::JournalRecord* last) const override {
        const std::string status = last == nullptr ? "" : last->metric_text("status");
        const double delta = last == nullptr ? 0.0 : last->metric_number("delta", 0.0);
        const char* mandate =
            "Never copy active w except when every probe mark is 0. Otherwise mutate at least 2 ints in A and B.\n";
        if (status == "rejected_divergent" || delta > 0.0) {
            return std::string(mandate) +
                   "- Previous step diverged or raised loss. Every candidate uses gamma <= 0.015.\n"
                   "- c1 gamma 0.005-0.010. c2 gamma 0.008-0.012. c3 gamma 0.010-0.015.";
        }
        if (delta < -0.1) {
            return std::string(mandate) +
                   "- Previous step dropped loss by more than 0.1. Settle instead of exploring.\n"
                   "- c1 and c2 gamma 0.01-0.03. c3 gamma 0.02-0.04.";
        }
        return std::string(mandate) +
               "- c1 (Gradient + Forward-Analysis Follower): move opposite +2/+1/-2/-1 while preserving active pathways. Gamma 0.015-0.035.\n"
               "- c2 (Manifold Alignment Step): follow the winning step and increase fa_target_corr. Gamma 0.030-0.055.\n"
               "- c3 (Basin & Rank Explorer): larger step on steep marks to expand effective rank and reduce damping. Gamma 0.050-0.080.";
    }

    std::string response_schema() const override {
        return TournamentPrompt::schema(
            candidate_scheme(),
            "1-2 sentences diagnosing probe signs and forward_analysis observations to justify step sizes");
    }

    std::string training_goal(double current_loss) const override {
        return "Target Loss: Minimize forward-analyzed regression error below 0.0100 using probe marks and "
               "offline_models/forward_analysis diagnostics. Current Best Loss: " +
               EvalScheme::format_loss(current_loss);
    }

    SchemeNotes scheme_notes() const override {
        return SchemeNotes{
            "This forward-analyzed regressor stores each slice as a rank-1 outer product. Move A opposite the probe and B with column marks.",
            "Bias codes are hundredths of the decoded bias. Use bias steps to revive dead units reported by fa_pathway_activity.",
            "Forward pass activations, predictions, targets, and residuals are distilled by offline_models/forward_analysis on every evaluation."};
    }

    SliceScheme slice_scheme() const override { return SliceScheme{}; }

    EvalScheme eval_scheme() const override { return EvalScheme{}; }

    const messaging::IContainerScheme& container_scheme() const override {
        return messaging::SJC1Compressor::instance();
    }

    JournalConfig journal_config() const override { return journal_scheme_.config(); }

    std::string journal_scheme_text() const override { return journal_scheme_.prompt_text(); }

    storage::JournalRecord seed_record(const ModelEvaluation& score, const std::string& checksum) const override {
        const ForwardAnalysisSnapshot fa = analyze_network(net_);
        storage::JournalRecord record;
        record.loss = score.primary_loss;
        record.checksum = checksum;
        storage::MetricFields::append_score(record, score);
        record.metrics.push_back(journal_scheme_.delta);
        record.metrics.push_back(journal_scheme_.winner);
        record.metrics.push_back(journal_scheme_.seeded);
        record.metrics.push_back(journal_scheme_.origin);
        append_forward_analysis_journal_fields(record, fa);
        return record;
    }

    storage::JournalRecord trial_record(const TrialNote& note) const override {
        const ForwardAnalysisSnapshot fa = analyze_network(net_);
        storage::JournalRecord record;
        record.loss = note.loss;
        record.checksum = note.checksum;
        record.metrics.push_back(journal_scheme_.with_number(journal_scheme_.delta, note.delta));
        if (note.record_winner) {
            record.metrics.push_back(journal_scheme_.with_text(journal_scheme_.winner, note.winner));
        }
        record.metrics.push_back(journal_scheme_.with_text(journal_scheme_.seeded, note.status));
        if (note.record_step) {
            record.metrics.push_back(journal_scheme_.with_text(journal_scheme_.step, note.winner_step));
        }
        if (note.record_slice) {
            record.metrics.push_back(journal_scheme_.with_text(journal_scheme_.slice, note.slice_id));
        }
        record.metrics.push_back(journal_scheme_.origin);
        if (note.score != nullptr) {
            storage::MetricFields::append_score(record, *note.score);
        }
        append_forward_analysis_journal_fields(record, fa);
        return record;
    }

    CheckpointDocument checkpoint_document(const ModelEvaluation& score, const std::string& checksum) const override {
        CheckpointDocument document = ModelTemplate::checkpoint_document(score, checksum);
        const ForwardAnalysisSnapshot fa = analyze_network(net_);
        document.fields.push_back(journal_scheme_.origin);
        document.fields.push_back(storage::MetricFields::number(
            "fa_dead_units", static_cast<double>(fa.total_dead_units), "Total dead hidden units from forward_analysis."));
        document.fields.push_back(storage::MetricFields::number(
            "fa_effective_rank", fa.mean_effective_rank, "Mean effective rank across hidden layers."));
        document.fields.push_back(storage::MetricFields::number(
            "fa_strongest_association", fa.strongest_association, "Peak input-to-hidden correlation."));
        document.fields.push_back(storage::MetricFields::number(
            "fa_target_correlation", fa.target_correlation, "Prediction-to-target correlation."));
        document.fields.push_back(storage::MetricFields::number(
            "fa_range_ratio", fa.range_ratio, "Prediction-to-target range ratio."));
        document.fields.push_back(storage::MetricFields::number(
            "fa_sign_agreement", fa.sign_agreement_ratio, "Prediction-to-target sign agreement fraction."));
        document.fields.push_back(storage::MetricFields::number(
            "fa_encoder_tokens", fa.encoder_tokens, "Multi-family token count encoded by PerceiverIOEncoder."));
        document.fields.push_back(
            storage::MetricFields::text("fa_eigen_signs", fa.eigen_signs_token, storage::MetricType::Token));
        document.fields.push_back(
            storage::MetricFields::text("fa_summary", fa.summary_token, storage::MetricType::Token));
        document.fields.push_back(
            storage::MetricFields::text("fa_pathway_activity", fa.concise_text.pathway_activity));
        document.fields.push_back(
            storage::MetricFields::text("fa_feature_sensitivity", fa.concise_text.feature_sensitivity));
        document.fields.push_back(
            storage::MetricFields::text("fa_representational_structure", fa.concise_text.representational_structure));
        document.fields.push_back(
            storage::MetricFields::text("fa_functional_mapping", fa.concise_text.functional_mapping));
        return document;
    }

    SchemaMap checkpoint_schema() const override {
        SchemaMap schema;
        schema.slot("weights", SlotShape::Matrix);
        schema.slot("biases", SlotShape::Vector);
        schema.summary(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.summary(FieldSpec{"regression_weight_mean", FieldSource::ParameterMean, "", false});
        schema.summary(FieldSpec{"regression_weight_variance", FieldSource::ParameterVariance, "", false});
        schema.summary(FieldSpec{"regression_weight_max", FieldSource::ParameterMax, "", false});
        schema.receipt(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.receipt(FieldSpec{"regression_weight_mean", FieldSource::ParameterMean, "", true});
        schema.receipt(FieldSpec{"regression_weight_variance", FieldSource::ParameterVariance, "", true});
        schema.receipt(FieldSpec{"output_count", FieldSource::OutputCount, "", false});
        schema.receipt(FieldSpec{"output_mean", FieldSource::OutputMean, "", true});
        schema.diagnostic(FieldSpec{"mean", FieldSource::LayerMean, "", false});
        schema.diagnostic(FieldSpec{"variance", FieldSource::LayerVariance, "", false});
        schema.diagnostic(FieldSpec{"regression_elements", FieldSource::LayerCount, "", false});
        schema.eval_input(EvalInput{"mean_squared_error", ValueKind::Number, "", 0});
        schema.eval_input(EvalInput{"residuals", ValueKind::NumberList, "residuals", 0});
        schema.eval_input(EvalInput{"anomalies", ValueKind::TextList, "anomalies", 80});
        schema.eval_field(FieldSpec{"mean_squared_error", FieldSource::CopiedNumber, "mean_squared_error", false});
        schema.eval_field(FieldSpec{"residual_count", FieldSource::ListCount, "residuals", false});
        schema.eval_field(FieldSpec{"residuals", FieldSource::CopiedNumberList, "residuals", false});
        schema.eval_field(FieldSpec{"anomalies", FieldSource::CopiedTextList, "anomalies", false});
        return schema;
    }

    ModelDirective get_directives() const {
        return ModelDirective{
            "4-in regression with offline_models/forward_analysis distillation.",
            "ForwardAnalyzedModel",
            "Use probe signs plus forward_analysis pathway, rank, and mapping diagnostics to choose c1/c2/c3."};
    }

private:
    NeuralNetwork net_;
    ForwardAnalyzedJournalScheme journal_scheme_;

    static forward_analysis::DualObjectiveDistillationBridge& shared_bridge() {
        static forward_analysis::DualObjectiveDistillationBridge bridge([] {
            forward_analysis::DistillerConfig cfg;
            cfg.narration_style = forward_analysis::NarrationStyle::kBatchStatistics;
            return cfg;
        }());
        return bridge;
    }

    int resolve_samples(int sample_count) const {
        const int count = sample_count < 1 ? eval_scheme().samples : sample_count;
        if (count < 1 || count > eval_scheme().samples) {
            throw std::invalid_argument("forward_analyzed evaluation uses 1 to " +
                                        std::to_string(eval_scheme().samples) + " samples");
        }
        return count;
    }

    void install(NeuralNetwork& model) const {
        model.slices = slice_scheme();
        model.evaluation = eval_scheme();
    }

    void append_forward_analysis_journal_fields(storage::JournalRecord& record,
                                                const ForwardAnalysisSnapshot& fa) const {
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.fa_dead_units, static_cast<double>(fa.total_dead_units)));
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.fa_effective_rank, fa.mean_effective_rank));
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.fa_target_corr, fa.target_correlation));
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.fa_sign_agreement, fa.sign_agreement_ratio));
        record.metrics.push_back(
            journal_scheme_.with_text(journal_scheme_.fa_summary, fa.summary_token));
        record.metrics.push_back(
            journal_scheme_.with_text(journal_scheme_.fa_pathway, fa.concise_text.pathway_activity));
        record.metrics.push_back(
            journal_scheme_.with_text(journal_scheme_.fa_mapping, fa.concise_text.functional_mapping));
    }

    static std::vector<double> sample_input(int index, int sample_count, std::size_t in_dim) {
        constexpr double kPi = 3.141592653589793;
        const double count = static_cast<double>(sample_count);
        std::vector<double> input(in_dim, 0.0);
        for (std::size_t axis = 0; axis < in_dim; ++axis) {
            if (axis == 0) {
                input[axis] = (static_cast<double>(index) / count) * kPi;
            } else if (axis == 1) {
                input[axis] = (static_cast<double>(sample_count - index) / count) * kPi;
            } else {
                const int phase = (index + static_cast<int>(axis) * 3) % sample_count;
                input[axis] = (static_cast<double>(phase) / count) * kPi;
            }
        }
        return input;
    }

    static forward_analysis::ForwardTelemetryTensor build_forward_telemetry(const NeuralNetwork& candidate,
                                                                            int sample_count) {
        if (candidate.layers.empty()) {
            throw std::runtime_error("Cannot build forward telemetry for an empty network.");
        }
        const std::size_t in_dim = candidate.layers.front().in_dim;
        const std::size_t out_dim = candidate.layers.back().out_dim;
        const std::size_t hidden_count =
            candidate.layers.size() > 1 ? candidate.layers.size() - 1 : 1;

        Eigen::MatrixXd inputs(sample_count, static_cast<Eigen::Index>(in_dim));
        Eigen::MatrixXd targets(sample_count, static_cast<Eigen::Index>(out_dim));
        Eigen::MatrixXd predictions(sample_count, static_cast<Eigen::Index>(out_dim));
        std::vector<Eigen::MatrixXd> layer_activations;
        layer_activations.reserve(hidden_count);
        for (std::size_t l = 0; l < hidden_count; ++l) {
            layer_activations.emplace_back(
                sample_count, static_cast<Eigen::Index>(candidate.layers[l].out_dim));
        }

        for (int index = 0; index < sample_count; ++index) {
            const std::vector<double> x = sample_input(index, sample_count, in_dim);
            for (std::size_t c = 0; c < in_dim; ++c) {
                inputs(index, static_cast<Eigen::Index>(c)) = x[c];
            }
            double target_val = 0.0;
            for (std::size_t axis = 0; axis < in_dim; ++axis) {
                target_val += (axis % 2 == 0) ? std::sin(x[axis]) : std::cos(x[axis]);
            }
            for (std::size_t k = 0; k < out_dim; ++k) {
                targets(index, static_cast<Eigen::Index>(k)) = target_val;
            }

            std::vector<double> current = x;
            for (std::size_t l = 0; l < candidate.layers.size(); ++l) {
                current = candidate.layers[l].forward(current);
                if (l < hidden_count) {
                    for (std::size_t u = 0; u < current.size(); ++u) {
                        layer_activations[l](index, static_cast<Eigen::Index>(u)) =
                            std::isfinite(current[u]) ? current[u] : 0.0;
                    }
                }
            }
            for (std::size_t k = 0; k < out_dim; ++k) {
                const double pred = k < current.size() && std::isfinite(current[k]) ? current[k] : 0.0;
                predictions(index, static_cast<Eigen::Index>(k)) = pred;
            }
        }

        forward_analysis::DistillerConfig fa_config;
        fa_config.narration_style = forward_analysis::NarrationStyle::kBatchStatistics;
        return forward_analysis::TelemetryBuilder::build(
            std::move(inputs), std::move(targets), std::move(predictions), std::move(layer_activations), fa_config);
    }

    static ModelEvaluation evaluate_with_forward_analysis(const NeuralNetwork& candidate, int sample_count) {
        if (candidate.layers.empty()) {
            throw std::runtime_error("Cannot evaluate an empty forward_analyzed network.");
        }
        const std::size_t in_dim = candidate.layers.front().in_dim;
        const std::size_t out_dim = candidate.layers.back().out_dim;
        const double count = static_cast<double>(sample_count);
        double squared = 0.0;
        double max_residual = 0.0;
        double residual_sum = 0.0;
        std::vector<double> residuals(static_cast<std::size_t>(sample_count));
        for (int index = 0; index < sample_count; ++index) {
            const std::vector<double> input = sample_input(index, sample_count, in_dim);
            const std::vector<double> predicted = candidate.forward(input);
            double sample_square = 0.0;
            double sample_peak = 0.0;
            for (std::size_t coord = 0; coord < out_dim; ++coord) {
                double target = 0.0;
                for (std::size_t axis = 0; axis < in_dim; ++axis) {
                    target += (axis % 2 == 0) ? std::sin(input[axis]) : std::cos(input[axis]);
                }
                const double residual = std::abs(predicted[coord] - target);
                sample_square += residual * residual;
                sample_peak = std::max(sample_peak, residual);
            }
            residuals[static_cast<std::size_t>(index)] = sample_peak;
            squared += sample_square / static_cast<double>(out_dim);
            residual_sum += sample_peak;
            max_residual = std::max(max_residual, sample_peak);
        }
        const double mean_residual = residual_sum / count;
        double moment = 0.0;
        for (double residual : residuals) {
            const double delta = residual - mean_residual;
            moment += delta * delta;
        }
        ModelEvaluation score = ModelEvaluation::from_grid(squared / count, max_residual, std::sqrt(moment / count));
        for (core::DescribedField& field : score.metrics) {
            if (field.key == "loss") {
                field.format_description = "Mean squared error against the sin/cos regression targets.";
                field.actionable_rule =
                    "Primary regression loss. Lower is better. A checkpoint write requires it to fall by more than 0.0001.";
            }
        }
        return score;
    }

    std::string probe_marks(const std::string& slice_id) const {
        const SliceRef slice = network().find_slice(slice_id);
        const std::string token = serialize_slice(slice);
        const std::string bias_token = serialize_bias(slice);
        const double baseline = evaluate_with_forward_analysis(net_, resolve_samples(0)).primary_loss;
        std::ostringstream hint;
        hint << "A:[";
        write_marks(hint, token, bias_token, "A:", slice, baseline);
        hint << "] B:[";
        write_marks(hint, token, bias_token, "B:", slice, baseline);
        hint << "]";
        return hint.str();
    }

    void write_marks(std::ostringstream& hint,
                     const std::string& token,
                     const std::string& bias_token,
                     const char* key,
                     const SliceRef& slice,
                     double baseline) const {
        const std::vector<int> codes = codes_of(token, key);
        const std::vector<double> bias_values = get_bias_scheme(slice).deserialize_vector(bias_token, slice.cols);
        for (std::size_t index = 0; index < codes.size(); ++index) {
            if (index > 0) {
                hint << ", ";
            }
            const int step = codes[index] >= 99 ? -1 : 1;
            const int probed = std::max(-99, std::min(99, codes[index] + step));
            NeuralNetwork trial = network();
            trial.patch_slice(
                slice.slice_id,
                get_scheme_for_slice(slice).deserialize_matrix(with_code(token, key, index, probed), slice.rows, slice.cols),
                bias_values);
            const double delta = evaluate_with_forward_analysis(trial, resolve_samples(0)).primary_loss - baseline;
            const double signed_delta = step > 0 ? delta : -delta;
            const double magnitude = std::abs(signed_delta);
            const double steep = std::max(0.02, 0.01 * std::abs(baseline));
            if (magnitude < 1e-8) {
                hint << '0';
            } else if (signed_delta > 0.0) {
                hint << (magnitude >= steep ? "+2" : "+1");
            } else {
                hint << (magnitude >= steep ? "-2" : "-1");
            }
        }
    }

    static std::vector<int> codes_of(const std::string& token, const char* key) {
        const std::size_t start = token.find(key);
        const std::size_t from = start + std::char_traits<char>::length(key);
        const std::size_t end = token.find('|', from);
        std::istringstream input(token.substr(from, end == std::string::npos ? std::string::npos : end - from));
        std::vector<int> codes;
        int value = 0;
        while (input >> value) {
            codes.push_back(value);
        }
        return codes;
    }

    static std::string with_code(const std::string& token, const char* key, std::size_t index, int value) {
        const std::vector<int> codes = codes_of(token, key);
        std::ostringstream replaced;
        for (std::size_t cursor = 0; cursor < codes.size(); ++cursor) {
            if (cursor > 0) {
                replaced << ' ';
            }
            replaced << (cursor == index ? value : codes[cursor]);
        }
        const std::size_t start = token.find(key) + std::char_traits<char>::length(key);
        const std::size_t end = token.find('|', start);
        return token.substr(0, start) + replaced.str() + (end == std::string::npos ? "" : token.substr(end));
    }
};

}  // namespace stevejob::models
