#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

enum class SlotShape { Matrix, Vector };

enum class ValueKind { Number, Count, NumberList, TextList };

enum class FieldSource {
    ParameterCount,
    ParameterMean,
    ParameterMin,
    ParameterMax,
    ParameterVariance,
    OutputCount,
    OutputMean,
    LayerMean,
    LayerVariance,
    LayerMin,
    LayerMax,
    LayerCount,
    CopiedNumber,
    CopiedNumberList,
    CopiedTextList,
    ListCount
};

struct ParameterSlot {
    std::string key;
    SlotShape shape = SlotShape::Matrix;
};

struct FieldSpec {
    std::string key;
    FieldSource source = FieldSource::ParameterCount;
    std::string from_key;
    bool omit_when_empty = false;
};

struct EvalInput {
    std::string key;
    ValueKind kind = ValueKind::Number;
    std::string cap;
    int clip = 0;
};

struct FieldValue {
    std::string key;
    ValueKind kind = ValueKind::Number;
    bool present = true;
    double number = 0.0;
    std::size_t count = 0;
    std::vector<double> numbers;
    std::vector<std::string> texts;
};

struct OutputSample {
    std::size_t count = 0;
    double total = 0.0;
};

struct EvalSample {
    std::map<std::string, double> numbers;
    std::map<std::string, std::vector<double>> number_lists;
    std::map<std::string, std::vector<std::string>> text_lists;
};

class SchemaMap {
public:
    void slot(std::string key, SlotShape shape) { slots_.push_back(ParameterSlot{std::move(key), shape}); }

    void summary(FieldSpec field) { summary_.push_back(std::move(field)); }

    void receipt(FieldSpec field) { receipt_.push_back(std::move(field)); }

    void diagnostic(FieldSpec field) { diagnostics_.push_back(std::move(field)); }

    void eval_input(EvalInput input) { eval_inputs_.push_back(std::move(input)); }

    void eval_field(FieldSpec field) { eval_fields_.push_back(std::move(field)); }

    const std::vector<ParameterSlot>& slots() const { return slots_; }

    const std::vector<EvalInput>& eval_inputs() const { return eval_inputs_; }

    std::vector<FieldValue> compile_summary(const std::vector<double>& parameters) const {
        return compile_numbers(summary_, parameters, OutputSample{});
    }

    std::vector<FieldValue> compile_receipt(const std::vector<double>& parameters, const OutputSample& outputs) const {
        return compile_numbers(receipt_, parameters, outputs);
    }

    std::vector<FieldValue> compile_diagnostics(double mean, double variance, double low, double high, std::size_t count) const {
        std::vector<FieldValue> fields;
        fields.reserve(diagnostics_.size());
        for (const FieldSpec& spec : diagnostics_) {
            FieldValue field;
            field.key = spec.key;
            field.kind = spec.source == FieldSource::LayerCount ? ValueKind::Count : ValueKind::Number;
            if (spec.source == FieldSource::LayerCount) {
                field.count = count;
            } else if (spec.source == FieldSource::LayerMean) {
                field.number = mean;
            } else if (spec.source == FieldSource::LayerVariance) {
                field.number = variance;
            } else if (spec.source == FieldSource::LayerMin) {
                field.number = low;
            } else if (spec.source == FieldSource::LayerMax) {
                field.number = high;
            }
            fields.push_back(std::move(field));
        }
        return fields;
    }

    std::vector<FieldValue> compile_eval(const EvalSample& sample) const {
        std::vector<FieldValue> fields;
        fields.reserve(eval_fields_.size());
        for (const FieldSpec& spec : eval_fields_) {
            fields.push_back(compile_eval_field(spec, sample));
        }
        return fields;
    }

private:
    struct ParameterRollup {
        std::size_t count = 0;
        double total = 0.0;
        double low = 0.0;
        double high = 0.0;
        double variance = 0.0;
        bool empty = true;
    };

    std::vector<FieldValue> compile_numbers(const std::vector<FieldSpec>& specs,
                                            const std::vector<double>& parameters,
                                            const OutputSample& outputs) const {
        const ParameterRollup rollup = rollup_parameters(parameters);
        std::vector<FieldValue> fields;
        fields.reserve(specs.size());
        for (const FieldSpec& spec : specs) {
            const bool empty = spec.source == FieldSource::OutputMean ? outputs.count == 0 : rollup.empty;
            if (spec.omit_when_empty && empty) {
                continue;
            }
            fields.push_back(compile_number(spec, rollup, outputs));
        }
        return fields;
    }

    static FieldValue compile_number(const FieldSpec& spec, const ParameterRollup& rollup, const OutputSample& outputs) {
        FieldValue field;
        field.key = spec.key;
        field.kind = spec.source == FieldSource::ParameterCount || spec.source == FieldSource::OutputCount ? ValueKind::Count
                                                                                                            : ValueKind::Number;
        if (spec.source == FieldSource::ParameterCount) {
            field.count = rollup.count;
        } else if (spec.source == FieldSource::ParameterMean) {
            field.number = rollup.count == 0 ? 0.0 : rollup.total / static_cast<double>(rollup.count);
        } else if (spec.source == FieldSource::ParameterMin) {
            field.number = rollup.low;
        } else if (spec.source == FieldSource::ParameterMax) {
            field.number = rollup.high;
        } else if (spec.source == FieldSource::ParameterVariance) {
            field.number = rollup.variance;
        } else if (spec.source == FieldSource::OutputCount) {
            field.count = outputs.count;
        } else if (spec.source == FieldSource::OutputMean) {
            field.number = outputs.count == 0 ? 0.0 : outputs.total / static_cast<double>(outputs.count);
        }
        return field;
    }

    static FieldValue compile_eval_field(const FieldSpec& spec, const EvalSample& sample) {
        FieldValue field;
        field.key = spec.key;
        if (spec.source == FieldSource::CopiedNumber) {
            field.kind = ValueKind::Number;
            const auto it = sample.numbers.find(spec.from_key);
            field.number = it == sample.numbers.end() ? 0.0 : it->second;
        } else if (spec.source == FieldSource::CopiedNumberList) {
            field.kind = ValueKind::NumberList;
            const auto it = sample.number_lists.find(spec.from_key);
            if (it != sample.number_lists.end()) {
                field.numbers = it->second;
            }
        } else if (spec.source == FieldSource::CopiedTextList) {
            field.kind = ValueKind::TextList;
            const auto it = sample.text_lists.find(spec.from_key);
            if (it != sample.text_lists.end()) {
                field.texts = it->second;
            }
        } else if (spec.source == FieldSource::ListCount) {
            field.kind = ValueKind::Count;
            const auto numbers = sample.number_lists.find(spec.from_key);
            if (numbers != sample.number_lists.end()) {
                field.count = numbers->second.size();
            }
            const auto texts = sample.text_lists.find(spec.from_key);
            if (texts != sample.text_lists.end()) {
                field.count = texts->second.size();
            }
        }
        return field;
    }

    static ParameterRollup rollup_parameters(const std::vector<double>& parameters) {
        ParameterRollup rollup;
        if (parameters.empty()) {
            return rollup;
        }
        rollup.empty = false;
        rollup.low = parameters[0];
        rollup.high = parameters[0];
        for (double value : parameters) {
            rollup.total += value;
            if (value < rollup.low) {
                rollup.low = value;
            }
            if (value > rollup.high) {
                rollup.high = value;
            }
            ++rollup.count;
        }
        const double mean = rollup.total / static_cast<double>(rollup.count);
        double moment = 0.0;
        for (double value : parameters) {
            const double delta = value - mean;
            moment += delta * delta;
        }
        rollup.variance = moment / static_cast<double>(rollup.count);
        return rollup;
    }

    std::vector<ParameterSlot> slots_;
    std::vector<FieldSpec> summary_;
    std::vector<FieldSpec> receipt_;
    std::vector<FieldSpec> diagnostics_;
    std::vector<EvalInput> eval_inputs_;
    std::vector<FieldSpec> eval_fields_;
};

class CheckpointRegistry {
public:
    static const char* model_tag() { return "architecture"; }

    static CheckpointRegistry& instance() {
        static CheckpointRegistry registry;
        return registry;
    }

    void add(std::string model_id, SchemaMap schema) { schemas_[std::move(model_id)] = std::move(schema); }

    const SchemaMap& find(const std::string& model_id) const {
        const auto it = schemas_.find(model_id);
        if (it == schemas_.end()) {
            return fallback();
        }
        return it->second;
    }

    const SchemaMap& fallback() const { return schemas_.at("legacy_layers"); }

private:
    CheckpointRegistry() { register_builtins(); }

    static void legacy_parameters(SchemaMap& schema) {
        schema.slot("weights", SlotShape::Matrix);
        schema.slot("biases", SlotShape::Vector);
    }

    static void legacy_eval(SchemaMap& schema) {
        schema.eval_input(EvalInput{"mean_squared_error", ValueKind::Number, "", 0});
        schema.eval_input(EvalInput{"residuals", ValueKind::NumberList, "residuals", 0});
        schema.eval_input(EvalInput{"anomalies", ValueKind::TextList, "anomalies", 80});
        schema.eval_field(FieldSpec{"mean_squared_error", FieldSource::CopiedNumber, "mean_squared_error", false});
        schema.eval_field(FieldSpec{"residual_count", FieldSource::ListCount, "residuals", false});
        schema.eval_field(FieldSpec{"residuals", FieldSource::CopiedNumberList, "residuals", false});
        schema.eval_field(FieldSpec{"anomalies", FieldSource::CopiedTextList, "anomalies", false});
    }

    static SchemaMap legacy_layers() {
        SchemaMap schema;
        legacy_parameters(schema);
        schema.summary(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.summary(FieldSpec{"weight_mean", FieldSource::ParameterMean, "", false});
        schema.summary(FieldSpec{"weight_min", FieldSource::ParameterMin, "", false});
        schema.summary(FieldSpec{"weight_max", FieldSource::ParameterMax, "", false});
        schema.receipt(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.receipt(FieldSpec{"weight_mean", FieldSource::ParameterMean, "", true});
        schema.receipt(FieldSpec{"weight_min", FieldSource::ParameterMin, "", true});
        schema.receipt(FieldSpec{"weight_max", FieldSource::ParameterMax, "", true});
        schema.receipt(FieldSpec{"output_count", FieldSource::OutputCount, "", false});
        schema.receipt(FieldSpec{"output_mean", FieldSource::OutputMean, "", true});
        schema.diagnostic(FieldSpec{"mean", FieldSource::LayerMean, "", false});
        schema.diagnostic(FieldSpec{"variance", FieldSource::LayerVariance, "", false});
        schema.diagnostic(FieldSpec{"max", FieldSource::LayerMax, "", false});
        schema.diagnostic(FieldSpec{"min", FieldSource::LayerMin, "", false});
        schema.diagnostic(FieldSpec{"total_elements", FieldSource::LayerCount, "", false});
        legacy_eval(schema);
        return schema;
    }

    static SchemaMap regression_layers() {
        SchemaMap schema;
        legacy_parameters(schema);
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
        legacy_eval(schema);
        return schema;
    }

    static SchemaMap reconstruction_layers() {
        SchemaMap schema;
        legacy_parameters(schema);
        schema.summary(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.summary(FieldSpec{"reconstruction_weight_mean", FieldSource::ParameterMean, "", false});
        schema.summary(FieldSpec{"reconstruction_weight_min", FieldSource::ParameterMin, "", false});
        schema.summary(FieldSpec{"latent_peak", FieldSource::ParameterMax, "", false});
        schema.receipt(FieldSpec{"parameter_count", FieldSource::ParameterCount, "", false});
        schema.receipt(FieldSpec{"reconstruction_weight_mean", FieldSource::ParameterMean, "", true});
        schema.receipt(FieldSpec{"output_count", FieldSource::OutputCount, "", false});
        schema.receipt(FieldSpec{"reconstruction_output_mean", FieldSource::OutputMean, "", true});
        schema.diagnostic(FieldSpec{"mean", FieldSource::LayerMean, "", false});
        schema.diagnostic(FieldSpec{"latent_peak", FieldSource::LayerMax, "", false});
        schema.diagnostic(FieldSpec{"latent_elements", FieldSource::LayerCount, "", false});
        schema.eval_input(EvalInput{"reconstruction_loss", ValueKind::Number, "", 0});
        schema.eval_input(EvalInput{"residuals", ValueKind::NumberList, "residuals", 0});
        schema.eval_field(FieldSpec{"reconstruction_loss", FieldSource::CopiedNumber, "reconstruction_loss", false});
        schema.eval_field(FieldSpec{"residual_count", FieldSource::ListCount, "residuals", false});
        schema.eval_field(FieldSpec{"residuals", FieldSource::CopiedNumberList, "residuals", false});
        return schema;
    }

    void register_builtins() {
        add("legacy_layers", legacy_layers());
        add("mlp_regressor", regression_layers());
        add("bottleneck_ae", reconstruction_layers());
    }

    std::map<std::string, SchemaMap> schemas_;
};

}  // namespace stevejob::models
