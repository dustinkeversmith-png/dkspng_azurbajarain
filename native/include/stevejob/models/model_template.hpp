#pragma once
#include "stevejob/core/metric_descriptor.hpp"
#include "stevejob/models/candidate_scheme.hpp"
#include "stevejob/models/evaluation_score.hpp"
#include "stevejob/models/neural_network.hpp"
#include "stevejob/messaging/container_scheme.hpp"
#include "stevejob/messaging/encoding_scheme.hpp"
#include "stevejob/messaging/output_scheme.hpp"
#include "stevejob/messaging/totr_engine.hpp"
#include "stevejob/models/eval_scheme.hpp"
#include "stevejob/models/journal_spec.hpp"
#include "stevejob/models/slice_scheme.hpp"
#include "stevejob/storage/metric_fields.hpp"

#include <string>
#include <utility>
#include <vector>

namespace stevejob::models {

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

private:
    storage::SlidingJournal journal_{1};
};

}  // namespace stevejob::models
