#pragma once

// Canonical online model definition: DemoScaleModel (architecture "demo_scale").
// Self-contained output scheme, telemetry encodings, journal scheme, per-learning
// LearningConfig, layer statistics, and checkpoint schema live on this class.

#include "stevejob/messaging/sjc1_scheme.hpp"
#include "stevejob/models/model_template.hpp"
#include "stevejob/models/tournament_prompt.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

class DemoScaleModel : public ModelTemplate {
public:
    class ScaleOutput : public messaging::IOutputScheme {
    public:
        static const ScaleOutput& instance() {
            static const ScaleOutput scheme;
            return scheme;
        }

        messaging::EncodingDescriptor get_descriptor() const override {
            return messaging::EncodingDescriptor{
                "scale_output",
                "One decimal, the linear head's scale prediction.",
                "prediction = head(mix(input)).",
                "The value is finite. The prompt does not list the target grid."};
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
                    throw std::invalid_argument("scale_output row has the wrong column count");
                }
                outputs.push_back(std::move(values));
                if (split == std::string::npos) {
                    pending.clear();
                } else {
                    pending = pending.substr(split + 1);
                }
            }
            if (outputs.size() != rows) {
                throw std::invalid_argument("scale_output token has the wrong row count");
            }
            return outputs;
        }
    };

    class SpanDecimalScheme : public messaging::IEncodingScheme {
    public:
        static const SpanDecimalScheme& instance() {
            static const SpanDecimalScheme scheme;
            return scheme;
        }

        messaging::EncodingDescriptor get_descriptor() const override {
            return messaging::EncodingDescriptor{
                "span_decimal",
                "One finite decimal.",
                "span = highest weight - lowest weight.",
                "Read the decimal as a span. It is not a TOTR token."};
        }

        std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const override {
            if (matrix.size() != 1) {
                throw std::invalid_argument("span_decimal encodes one row");
            }
            return serialize_vector(matrix[0], 0, matrix[0].size());
        }

        std::vector<std::vector<double>> deserialize_matrix(const std::string& token,
                                                            std::size_t rows,
                                                            std::size_t cols) const override {
            if (rows != 1) {
                throw std::invalid_argument("span_decimal decodes one row");
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
    };

    class MixSignScheme : public messaging::IEncodingScheme {
    public:
        static const MixSignScheme& instance() {
            static const MixSignScheme scheme;
            return scheme;
        }

        messaging::EncodingDescriptor get_descriptor() const override {
            return messaging::EncodingDescriptor{
                "mix_sign",
                "+1 or -1.",
                "sign = +1 when the first mix weight is non-negative, otherwise -1.",
                "+1 means hold can keep the mix. -1 means shift should reverse the head step."};
        }

        std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const override {
            (void)matrix;
            throw std::invalid_argument("mix_sign encodes one mark");
        }

        std::vector<std::vector<double>> deserialize_matrix(const std::string& token,
                                                            std::size_t rows,
                                                            std::size_t cols) const override {
            (void)token;
            (void)rows;
            (void)cols;
            throw std::invalid_argument("mix_sign decodes one mark");
        }

        std::string serialize_vector(const std::vector<double>& values,
                                     std::size_t start,
                                     std::size_t count) const override {
            (void)count;
            return values.at(start) >= 0.0 ? "+1" : "-1";
        }

        std::vector<double> deserialize_vector(const std::string& text, std::size_t count) const override {
            if (count != 1) {
                throw std::invalid_argument("mix_sign decodes one mark");
            }
            if (text.find("-1") != std::string::npos) {
                return {-1.0};
            }
            return {1.0};
        }
    };

    struct ScaleJournalScheme {
        storage::MetricField origin{"origin", storage::MetricType::Text, std::string("demo_scale"),
                                    "Architecture that opened this checkpoint."};
        storage::MetricField seeded{"status", storage::MetricType::Text, std::string("seed"),
                                    "Opening row for a DemoScale checkpoint."};

        JournalConfig config() const {
            JournalConfig journal;
            journal.capacity = 5;
            journal.task = "scale";
            journal.description =
                "DemoScale journal scheme. origin names the architecture. status is seed on the opening row.";
            return journal;
        }

        std::string prompt_text() const {
            return "Journal scheme: task scale. origin is text and names the architecture. status is text and is seed on the opening row.";
        }

        storage::JournalRecord open_record(double loss, const std::string& checksum) const {
            storage::JournalRecord record;
            record.loss = loss;
            record.checksum = checksum;
            record.metrics.push_back(origin);
            record.metrics.push_back(seeded);
            return record;
        }

        storage::MetricField status_field(const std::string& status) const {
            storage::MetricField field = seeded;
            field.value = status;
            field.prompt_description = "Trial status stored by the DemoScale journal scheme.";
            return field;
        }
    };

    DemoScaleModel() : net_("DemoScaleModel") {
        config_.max_candidates = 2;
        config_.max_parameters = 12;
        config_.max_input_rows = 4;
        config_.max_input_cols = 3;
        net_.architecture = "demo_scale";
        net_.add_layer("mix", 3, 2, TanhActivation{}).add_layer("head", 2, 1, LinearActivation{});
        net_.initialize_default_weights();
        install(net_);
    }

    void adopt(NeuralNetwork model) {
        install(model);
        net_ = std::move(model);
    }

    NeuralNetwork& network() override { return net_; }
    const NeuralNetwork& network() const override { return net_; }

    ModelEvaluation evaluate_model(int sample_count = 0) const override {
        return scale_loss(net_, resolve_samples(sample_count));
    }

    ModelEvaluation score_patch(const NeuralNetwork& candidate_net, int sample_count = 0) const override {
        return scale_loss(candidate_net, resolve_samples(sample_count));
    }

    std::vector<TelemetrySignal> derive_training_telemetry() const override {
        double lowest = 0.0;
        double highest = 0.0;
        bool seen = false;
        for (const LayerSpec& layer : network().layers) {
            for (const std::vector<double>& row : layer.weights) {
                for (double value : row) {
                    if (!seen) {
                        lowest = highest = value;
                        seen = true;
                    } else {
                        lowest = std::min(lowest, value);
                        highest = std::max(highest, value);
                    }
                }
            }
        }
        const double span = seen ? highest - lowest : 0.0;
        const std::string sign = mix_sign();
        return {
            {core::DescribedField{"scale_span", core::FieldType::Scalar, span, "telemetry",
                                  "Highest scale weight minus the lowest scale weight.",
                                  "A wide span means the mix layer is unbalanced. Prefer the hold candidate."},
             &SpanDecimalScheme::instance()},
            {core::DescribedField{"mix_sign", core::FieldType::Token, sign, "telemetry",
                                  "Sign of the first mix weight.",
                                  "Use the mix_sign codec. +1 keeps the mix. -1 reverses the head step."},
             &MixSignScheme::instance()},
        };
    }

    core::DescribedField derive_gradient_probe(const std::string& slice_id) const override {
        return core::DescribedField{
            "scale_probe",
            core::FieldType::Token,
            slice_id + ":" + mix_sign(),
            "gradient_probe",
            "One mark for the first mix weight. +1 means that weight is non-negative.",
            "hold keeps the mix integers. shift moves the head integers by 1 or 2 in the opposite direction of the mark."};
    }

    const messaging::IEncodingScheme& get_scheme_for_slice(const SliceRef& slice) const override {
        (void)slice;
        return messaging::TotrRank1Scheme::instance();
    }

    const messaging::IEncodingScheme& get_bias_scheme(const SliceRef& slice) const override {
        (void)slice;
        return messaging::BiasFloatScheme::instance();
    }

    const messaging::IOutputScheme& get_output_scheme() const override { return ScaleOutput::instance(); }

    CandidateScheme candidate_scheme() const override {
        CandidateScheme scheme;
        scheme.slots = {"hold", "shift"};
        return scheme;
    }

    std::string candidate_roles(const storage::JournalRecord* last) const override {
        (void)last;
        return "DemoScale candidates: hold keeps gamma in 0.010-0.020. shift uses gamma in 0.020-0.040. "
               "Do not add a third candidate.";
    }

    std::string response_schema() const override {
        return TournamentPrompt::schema(candidate_scheme(), "one sentence naming whether hold or shift should win");
    }

    std::string training_goal(double current_loss) const override {
        return "DemoScale goal: shrink the scale loss below 0.0500. Current Best Loss: " +
               EvalScheme::format_loss(current_loss);
    }

    SchemeNotes scheme_notes() const override {
        return SchemeNotes{
            "DemoScale stores mix and head as rank-1 tokens. hold edits mix. shift edits head.",
            "Bias values are plain floats, one per output of the active slice.",
            "The scale target is half the sum of the three inputs. The head prediction uses scale_output."};
    }

    SliceScheme slice_scheme() const override { return SliceScheme{}; }

    EvalScheme eval_scheme() const override {
        EvalScheme scheme;
        scheme.samples = 4;
        return scheme;
    }

    const messaging::IContainerScheme& container_scheme() const override {
        return messaging::SJC1Compressor::instance();
    }

    JournalConfig journal_config() const override { return journal_scheme_.config(); }

    std::string journal_scheme_text() const override { return journal_scheme_.prompt_text(); }

    storage::JournalRecord seed_record(const ModelEvaluation& score, const std::string& checksum) const override {
        return journal_scheme_.open_record(score.primary_loss, checksum);
    }

    storage::JournalRecord trial_record(const TrialNote& note) const override {
        storage::JournalRecord record;
        record.loss = note.loss;
        record.checksum = note.checksum;
        record.metrics.push_back(journal_scheme_.origin);
        record.metrics.push_back(journal_scheme_.status_field(note.status));
        if (note.score != nullptr) {
            storage::MetricFields::append_score(record, *note.score);
        }
        return record;
    }

    CheckpointDocument checkpoint_document(const ModelEvaluation& score, const std::string& checksum) const override {
        CheckpointDocument document = ModelTemplate::checkpoint_document(score, checksum);
        document.fields.push_back(journal_scheme_.origin);
        return document;
    }

    SchemaMap checkpoint_schema() const override {
        SchemaMap schema;
        schema.slot("weights", SlotShape::Matrix);
        schema.slot("biases", SlotShape::Vector);
        schema.summary(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.summary(FieldSpec{"scale_weight_mean", FieldSource::ParameterMean, "", false});
        schema.summary(FieldSpec{"scale_weight_min", FieldSource::ParameterMin, "", false});
        schema.summary(FieldSpec{"scale_weight_max", FieldSource::ParameterMax, "", false});
        schema.receipt(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.receipt(FieldSpec{"scale_weight_mean", FieldSource::ParameterMean, "", true});
        schema.receipt(FieldSpec{"output_count", FieldSource::OutputCount, "", false});
        schema.receipt(FieldSpec{"output_mean", FieldSource::OutputMean, "", true});
        schema.diagnostic(FieldSpec{"mean", FieldSource::LayerMean, "", false});
        schema.diagnostic(FieldSpec{"variance", FieldSource::LayerVariance, "", false});
        schema.diagnostic(FieldSpec{"scale_elements", FieldSource::LayerCount, "", false});
        schema.eval_input(EvalInput{"mean_squared_error", ValueKind::Number, "", 0});
        schema.eval_input(EvalInput{"residuals", ValueKind::NumberList, "residuals", 0});
        schema.eval_field(FieldSpec{"mean_squared_error", FieldSource::CopiedNumber, "mean_squared_error", false});
        schema.eval_field(FieldSpec{"residual_count", FieldSource::ListCount, "residuals", false});
        schema.eval_field(FieldSpec{"residuals", FieldSource::CopiedNumberList, "residuals", false});
        return schema;
    }

    ModelDirective get_directives() const {
        return ModelDirective{"3-in scale, tanh mix, linear head.", "DemoScaleModel", "hold the mix. shift the head."};
    }

private:
    NeuralNetwork net_;
    ScaleJournalScheme journal_scheme_;

    int resolve_samples(int sample_count) const { return sample_count < 1 ? eval_scheme().samples : sample_count; }

    void install(NeuralNetwork& model) const {
        model.slices = slice_scheme();
        model.evaluation = eval_scheme();
    }

    std::string mix_sign() const {
        if (network().layers.empty() || network().layers.front().weights.empty() ||
            network().layers.front().weights.front().empty()) {
            return "0";
        }
        return MixSignScheme::instance().serialize_vector({network().layers.front().weights.front().front()}, 0, 1);
    }

    static ModelEvaluation scale_loss(const NeuralNetwork& candidate, int sample_count) {
        if (candidate.layers.empty() || candidate.layers.front().in_dim != 3) {
            throw std::invalid_argument("demo scale evaluation expects a 3-input network");
        }
        const double count = static_cast<double>(sample_count);
        double squared = 0.0;
        double peak = 0.0;
        for (int index = 0; index < sample_count; ++index) {
            const double step = static_cast<double>(index + 1) / count;
            const std::vector<double> input{step, step * 0.5, step * 0.25};
            const std::vector<double> predicted = candidate.forward(input);
            const double target = 0.5 * (input[0] + input[1] + input[2]);
            const double residual = (predicted.empty() ? 0.0 : predicted.front()) - target;
            squared += residual * residual;
            peak = std::max(peak, std::abs(residual));
        }
        ModelEvaluation score = ModelEvaluation::from_grid(squared / count, peak, 0.0);
        for (core::DescribedField& field : score.metrics) {
            if (field.key == "loss") {
                field.format_description = "Mean squared error against half the sum of the three inputs.";
                field.actionable_rule = "This is the scale loss. Lower is better.";
            }
        }
        return score;
    }
};

}  // namespace stevejob::models
