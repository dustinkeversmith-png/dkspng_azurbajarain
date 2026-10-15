#pragma once
#include "stevejob/core/config.hpp"
#include "stevejob/core/metric_descriptor.hpp"
#include "stevejob/math/linear_algebra.hpp"
#include "stevejob/math/tensor_stats.hpp"
#include "stevejob/models/candidate_scheme.hpp"
#include "stevejob/models/checkpoint_registry.hpp"
#include "stevejob/models/evaluation_score.hpp"
#include "stevejob/models/neural_network.hpp"
#include "stevejob/messaging/container_scheme.hpp"
#include "stevejob/messaging/encoding_scheme.hpp"
#include "stevejob/messaging/output_scheme.hpp"
#include "stevejob/models/eval_scheme.hpp"
#include "stevejob/models/journal_spec.hpp"
#include "stevejob/models/slice_scheme.hpp"
#include "stevejob/storage/json_text.hpp"
#include "stevejob/storage/metric_fields.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stevejob::models {

using LearningConfig = stevejob::config::LearningConfig;

struct ModelDirective {
    std::string task;
    std::string model_type;
    std::string rules;
};

struct SchemeNotes {
    std::string weights;
    std::string bias;
    std::string output;
};

struct TelemetrySignal {
    core::DescribedField field;
    const messaging::IEncodingScheme* encoding = nullptr;
};

class ModelTemplate {
public:
    virtual ~ModelTemplate() = default;

    const LearningConfig& config() const { return config_; }
    LearningConfig& config() { return config_; }
    void set_config(const LearningConfig& cfg) { config_ = cfg; }

    virtual NeuralNetwork& network() = 0;
    virtual const NeuralNetwork& network() const = 0;

    virtual ModelEvaluation evaluate_model(int sample_count = 0) const = 0;
    virtual ModelEvaluation score_patch(const NeuralNetwork& candidate_net, int sample_count = 0) const = 0;

    // 2. Extensible Multi-Field Telemetry with Descriptors
    virtual std::vector<TelemetrySignal> derive_training_telemetry() const = 0;

    // 3. Gradient Probing as an Extensible Telemetry Option
    virtual core::DescribedField derive_gradient_probe(const std::string& slice_id) const = 0;

    virtual bool probe_is_flat(const std::string& slice_id) const {
        const std::string hint = derive_gradient_probe(slice_id).to_string();
        return hint.find('+') == std::string::npos && hint.find('-') == std::string::npos;
    }

    // 4. Decoupled Schemes
    virtual const messaging::IEncodingScheme& get_scheme_for_slice(const SliceRef& slice) const = 0;
    virtual const messaging::IEncodingScheme& get_bias_scheme(const SliceRef& slice) const = 0;
    virtual const messaging::IOutputScheme& get_output_scheme() const = 0;

    virtual CandidateScheme candidate_scheme() const { return CandidateScheme{}; }

    virtual std::string candidate_roles(const storage::JournalRecord* last) const = 0;
    virtual std::string response_schema() const = 0;
    virtual std::string training_goal(double current_loss) const = 0;
    virtual SchemeNotes scheme_notes() const = 0;
    virtual SliceScheme slice_scheme() const = 0;
    virtual EvalScheme eval_scheme() const = 0;
    virtual const messaging::IContainerScheme& container_scheme() const = 0;

    virtual JournalConfig journal_config() const { return JournalConfig{}; }

    virtual std::string journal_scheme_text() const { return ""; }

    virtual storage::JournalRecord seed_record(const ModelEvaluation& score, const std::string& checksum) const = 0;

    virtual storage::JournalRecord trial_record(const TrialNote& note) const = 0;

    virtual SchemaMap checkpoint_schema() const { return SchemaMap::default_schema(); }

    virtual math::TensorStats compute_layer_stats(const std::vector<std::vector<double>>& matrix) const {
        return math::TensorStats::from_matrix(matrix);
    }

    virtual std::vector<FieldValue> layer_diagnostic_fields(const std::vector<std::vector<double>>& matrix) const {
        const math::TensorStats stats = compute_layer_stats(matrix);
        return checkpoint_schema().compile_diagnostics(
            stats.mean(), stats.variance(), stats.min_val(), stats.max_val(), stats.count());
    }

    virtual double validate_dampening(double value) const {
        if (!std::isfinite(value) || value < config_.min_dampening || value > config_.max_dampening) {
            std::ostringstream msg;
            msg << "dampening must be in [" << config_.min_dampening << ", " << config_.max_dampening << "]";
            throw std::invalid_argument(msg.str());
        }
        return value;
    }

    virtual double parse_dampening_text(const std::string& raw) const {
        const std::string trimmed = trim_ws(raw);
        if (trimmed.empty()) {
            return 1.0;
        }
        try {
            std::size_t idx = 0;
            const double direct = std::stod(trimmed, &idx);
            if (idx == trimmed.size()) {
                return validate_dampening(direct);
            }
        } catch (const std::exception&) {
        }
        const storage::JsonText parsed = extract_json_object(trimmed, "dampening response");
        const storage::JsonText* field = parsed.find("dampening");
        if (field == nullptr || field->kind != storage::JsonText::Kind::Number) {
            throw std::invalid_argument("dampening must be a number");
        }
        return validate_dampening(field->number);
    }

    virtual double validate_learning_rate(double lr) const {
        if (!std::isfinite(lr) || lr <= 0.0 || lr > config_.max_learning_rate) {
            std::ostringstream msg;
            msg << "learning_rate must be in (0, " << config_.max_learning_rate << "]";
            throw std::invalid_argument(msg.str());
        }
        return lr;
    }

    virtual double validate_weight_decay(double decay) const {
        if (!std::isfinite(decay) || decay < 0.0 || decay > 1.0) {
            throw std::invalid_argument("weight_decay must be in [0, 1]");
        }
        return decay;
    }

    virtual storage::JsonText parse_gradients_text(const std::string& raw) const {
        const storage::JsonText parsed = extract_json_object(raw, "gradient response");
        const storage::JsonText* inner = parsed.find("gradients");
        if (inner != nullptr && inner->kind == storage::JsonText::Kind::Object) {
            return *inner;
        }
        return parsed;
    }

    virtual void validate_inputs(const std::vector<std::vector<double>>& inputs) const {
        const std::size_t max_rows = config_.max_input_rows;
        const std::size_t max_cols = config_.max_input_cols;
        const std::string message = "inputs must be 1 to " + std::to_string(max_rows) + " samples with at most " +
                                    std::to_string(max_cols) + " features";
        if (inputs.empty() || inputs.size() > max_rows) {
            throw std::invalid_argument(message);
        }
        for (const auto& row : inputs) {
            if (row.size() > max_cols) {
                throw std::invalid_argument(message);
            }
        }
    }

    virtual std::vector<std::vector<double>> evaluate_layer_inputs(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& weights,
        const std::vector<double>& biases) const {
        validate_inputs(inputs);
        return math::linear_outputs(inputs, weights, biases);
    }

    virtual CheckpointDocument checkpoint_document(const ModelEvaluation& score, const std::string& checksum) const {
        CheckpointDocument document;
        document.fields.push_back(storage::MetricFields::number("loss", score.primary_loss, "Scored loss."));
        for (const core::DescribedField& field : score.metrics) {
            if (field.key == "loss" || !std::holds_alternative<double>(field.value)) {
                continue;
            }
            document.fields.push_back(
                storage::MetricFields::number(field.key, std::get<double>(field.value), field.actionable_rule));
        }
        document.fields.push_back(storage::MetricFields::text("checksum", checksum, storage::MetricType::Token));
        document.fields.push_back(storage::MetricFields::text("architecture", network().architecture));
        return document;
    }

    void bind_journal(std::size_t capacity_override = 0) {
        const JournalConfig config = journal_config();
        const std::size_t capacity = capacity_override == 0 ? config.capacity : capacity_override;
        journal_ = storage::SlidingJournal(capacity, config.description);
        journal_.set_task(config.task);
    }

    void take_journal(storage::SlidingJournal journal) { journal_ = std::move(journal); }

    storage::SlidingJournal& journal() { return journal_; }
    const storage::SlidingJournal& journal() const { return journal_; }

    void append_seed(const ModelEvaluation& score, const std::string& checksum) {
        journal_.append_record(seed_record(score, checksum));
    }

    // Serialization wrappers
    std::string serialize_slice(const SliceRef& slice) const {
        return get_scheme_for_slice(slice).serialize_matrix(network().layers[slice.layer_index].extract_block(slice));
    }
    std::string serialize_bias(const SliceRef& slice) const {
        return get_bias_scheme(slice).serialize_vector(network().layers[slice.layer_index].biases, slice.col_offset, slice.cols);
    }
    void apply_slice_patch(const SliceRef& slice, const std::string& w_token, const std::string& b_token) {
        const std::vector<std::vector<double>> block =
            get_scheme_for_slice(slice).deserialize_matrix(w_token, slice.rows, slice.cols);
        const std::vector<double> bias = get_bias_scheme(slice).deserialize_vector(b_token, slice.cols);
        network().patch_slice(slice.slice_id, block, bias);
    }

    static storage::JsonText extract_json_object(const std::string& raw, const std::string& label) {
        const std::string trimmed = trim_ws(raw);
        if (trimmed.empty()) {
            throw std::invalid_argument(label + " did not contain a JSON object");
        }
        try {
            const storage::JsonText direct = storage::JsonText::parse(trimmed);
            if (direct.kind == storage::JsonText::Kind::Object) {
                return direct;
            }
        } catch (const std::exception&) {
        }
        const std::size_t start = trimmed.find('{');
        const std::size_t end = trimmed.rfind('}');
        if (start == std::string::npos || end == std::string::npos || end <= start) {
            throw std::invalid_argument(label + " did not contain a JSON object");
        }
        const storage::JsonText sliced = storage::JsonText::parse(trimmed.substr(start, end - start + 1));
        if (sliced.kind != storage::JsonText::Kind::Object) {
            throw std::invalid_argument(label + " must be a JSON object");
        }
        return sliced;
    }

protected:
    LearningConfig config_{};

private:
    static std::string trim_ws(const std::string& s) {
        const std::size_t first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        const std::size_t last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    }

    storage::SlidingJournal journal_{1};
};

}  // namespace stevejob::models

