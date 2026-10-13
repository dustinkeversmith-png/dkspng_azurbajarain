#pragma once

#include "stevejob/models/model_template.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <sstream>
#include <string>

namespace stevejob::compiler {

struct PromptCompilerOptions {
    std::size_t visible_history_count = 3;
    int int_bound = 25;
    std::string retry_note;
};

class PromptCompiler {
public:
    static std::string compile_slice_prompt(const models::ModelTemplate& model,
                                            const std::string& slice_id,
                                            const storage::SlidingJournal& journal,
                                            double current_loss,
                                            const PromptCompilerOptions& options) {
        const models::SliceRef slice = model.network().find_slice(slice_id);
        std::ostringstream prompt;

        prompt << "=== TASK SPECIFICATION ===\n";
        prompt << "Model: " << model.network().architecture << "\n";
        prompt << "Slice: " << slice.slice_id << " [" << slice.rows << "x" << slice.cols << "]\n";
        prompt << model.training_goal(current_loss) << "\n\n";

        prompt << "=== RECENT TRAJECTORY & JOURNAL ===\n";
        prompt << journal.get_general_description() << "\n";
        const std::string journal_scheme = model.journal_scheme_text();
        if (!journal_scheme.empty()) {
            prompt << journal_scheme << "\n";
        }
        prompt << journal.format_visible(options.visible_history_count) << "\n";

        prompt << "=== SYSTEM TELEMETRY & DIAGNOSTICS ===\n";
        for (const models::TelemetrySignal& signal : model.derive_training_telemetry()) {
            const stevejob::core::DescribedField& field = signal.field;
            prompt << "[" << field.key << "]: " << field.to_string() << "\n";
            prompt << "  -> Interpretation: " << field.format_description << "\n";
            prompt << "  -> Guidance: " << field.actionable_rule << "\n";
            if (signal.encoding != nullptr) {
                const stevejob::messaging::EncodingDescriptor info = signal.encoding->get_descriptor();
                prompt << "  -> Encoding: " << info.scheme_name << "\n";
                prompt << "     Format: " << info.format_description << "\n";
                prompt << "     Equation: " << info.equation << "\n";
                prompt << "     Codec: " << info.actionable_rules << "\n";
            }
        }
        prompt << "\n";

        prompt << "=== DIRECTIONAL GRADIENT PROBE ===\n";
        const stevejob::core::DescribedField probe = model.derive_gradient_probe(slice_id);
        prompt << "Probe Data: " << probe.to_string() << "\n";
        prompt << "Format Definition: " << probe.format_description << "\n";
        prompt << "Actionable Rules: " << probe.actionable_rule << "\n\n";

        prompt << "=== PARAMETER ENCODINGS & ACTIVE STATE ===\n";
        const stevejob::messaging::IEncodingScheme& weights = model.get_scheme_for_slice(slice);
        const stevejob::messaging::IEncodingScheme& bias = model.get_bias_scheme(slice);
        const stevejob::messaging::EncodingDescriptor weight_info = weights.get_descriptor();
        const stevejob::messaging::EncodingDescriptor bias_info = bias.get_descriptor();
        const models::SchemeNotes notes = model.scheme_notes();
        prompt << "Weight Scheme: " << weight_info.scheme_name << "\n";
        prompt << "  Format: " << weight_info.format_description << "\n";
        prompt << "  Equation: " << weight_info.equation << "\n";
        prompt << "  Codec: " << weight_info.actionable_rules << "\n";
        prompt << "  Model: " << notes.weights << "\n";
        prompt << "Bias Scheme: " << bias_info.scheme_name << "\n";
        prompt << "  Format: " << bias_info.format_description << "\n";
        prompt << "  Equation: " << bias_info.equation << "\n";
        prompt << "  Codec: " << bias_info.actionable_rules << "\n";
        prompt << "  Model: " << notes.bias << "\n";
        prompt << "Output Scheme: " << model.get_output_scheme().get_output_description() << "\n";
        prompt << "  Model: " << notes.output << "\n";
        prompt << "Active Slice: w:\"" << model.serialize_slice(slice) << "\" | b:\"" << model.serialize_bias(slice) << "\"\n\n";

        prompt << "=== CANDIDATE ROLES & CONSTRAINTS ===\n";
        prompt << weights.constraint_line(options.int_bound);
        prompt << bias.constraint_line(options.int_bound);
        prompt << model.candidate_roles(journal.latest()) << "\n\n";
        prompt << model.response_schema();

        if (!options.retry_note.empty()) {
            prompt << "\n[CORRECTION REQUIRED: " << options.retry_note << "]";
        }
        return prompt.str();
    }
};

}  // namespace stevejob::compiler
