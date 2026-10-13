#pragma once

#include "stevejob/messaging/sjc1_scheme.hpp"
#include "stevejob/models/model_template.hpp"
#include "stevejob/models/tournament_prompt.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace stevejob::models {

class BottleneckAutoencoder : public ModelTemplate {
public:
    BottleneckAutoencoder() : net_("BottleneckAutoencoder") {
        net_.architecture = "bottleneck_ae";
        net_.add_layer("encode", 8, 2, ReLUActivation{})
            .add_layer("decode", 2, 8, LinearActivation{});
        net_.initialize_default_weights();
        install(net_);
    }

    void adopt(NeuralNetwork model) {
        install(model);
        net_ = std::move(model);
    }

    NeuralNetwork& network() override { return net_; }
    const NeuralNetwork& network() const override { return net_; }

    ModelEvaluation evaluate_model(int sample_count = 0) const override { return score_patch(net_, sample_count); }

    ModelEvaluation score_patch(const NeuralNetwork& model, int sample_count = 0) const override {
        sample_count = sample_count < 1 ? eval_scheme().samples : sample_count;
        if (sample_count < 1 || sample_count > eval_scheme().samples) {
            throw std::invalid_argument("reconstruction evaluation uses 1 to " + std::to_string(eval_scheme().samples) + " samples");
        }
        const std::size_t in_dim = model.layers.front().in_dim;
        const double count = static_cast<double>(sample_count);
        double squared = 0.0;
        double max_residual = 0.0;
        double residual_sum = 0.0;
        double dead_units = 0.0;
        double latent_units = 0.0;
        std::vector<double> residuals(static_cast<std::size_t>(sample_count));
        for (int index = 0; index < sample_count; ++index) {
            const std::vector<double> input = sample_input(index, sample_count, in_dim);
            const std::vector<double> latent = model.layers.front().forward(input);
            for (double unit : latent) {
                latent_units += 1.0;
                if (unit <= 0.0) {
                    dead_units += 1.0;
                }
            }
            const std::vector<double> predicted = model.forward(input);
            double sample_square = 0.0;
            double sample_peak = 0.0;
            for (std::size_t coord = 0; coord < in_dim; ++coord) {
                const double residual = std::abs(predicted[coord] - input[coord]);
                sample_square += residual * residual;
                sample_peak = std::max(sample_peak, residual);
            }
            residuals[static_cast<std::size_t>(index)] = sample_peak;
            squared += sample_square / static_cast<double>(in_dim);
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
        const double saturation = latent_units == 0.0 ? 0.0 : dead_units / latent_units;
        for (core::DescribedField& field : score.metrics) {
            if (field.key == "loss") {
                field.format_description = "Mean squared reconstruction error. The target row is the input row.";
                field.actionable_rule =
                    "This is the reconstruction loss. Lower is better. A checkpoint write requires it to fall by more than 0.0001.";
            }
        }
        score.metrics.push_back(core::DescribedField{
            "bottleneck_saturation",
            core::FieldType::Scalar,
            saturation,
            "eval_metric",
            "Fraction of 2-wide ReLU latent units that output 0 on the eval grid.",
            "Saturation near 1 means the bottleneck is collapsed. Prefer a step that reconstructs without killing both latent units."});
        return score;
    }

    std::vector<TelemetrySignal> derive_training_telemetry() const override {
        double count = 0.0;
        double sparse = 0.0;
        for (const LayerSpec& layer : network().layers) {
            for (const std::vector<double>& row : layer.weights) {
                for (double value : row) {
                    if (std::abs(value) < 0.05) {
                        sparse += 1.0;
                    }
                    count += 1.0;
                }
            }
        }
        return {
            {core::DescribedField{
                 "sparsity",
                 core::FieldType::Scalar,
                 count == 0.0 ? 0.0 : sparse / count,
                 "telemetry",
                 "Fraction of autoencoder weights whose magnitude is below 0.05.",
                 "Keep the encoder able to reach both latent units. High sparsity is not a lower reconstruction loss."},
             nullptr},
            {core::DescribedField{
                 "latent_utilization",
                 core::FieldType::Scalar,
                 latent_utilization(network(), eval_scheme().samples),
                 "telemetry",
                 "Fraction of bottleneck ReLU units that fire on the reconstruction grid.",
                 "Utilization near 0 means the 2-wide code is unused. Do not drive both latent units to zero."},
             nullptr},
        };
    }

    core::DescribedField derive_gradient_probe(const std::string& slice_id) const override {
        return core::DescribedField{
            "gradient",
            core::FieldType::Token,
            probe_marks(slice_id),
            "gradient_probe",
            "Comma-separated reconstruction marks: +2, +1, 0, -1, -2. One mark per A code, then one per B code.",
            "'+2' and '+1' are ascent on reconstruction loss: decrease that code, by 6 to 12 for +2 and by 2 to 5 for +1. "
            "'-2' and '-1' are descent: increase that code, by 6 to 12 for -2 and by 2 to 5 for -1. "
            "'0' is a flat dead zone. When every mark is 0, keep A and B integers and raise each bias by 0.2 to 0.5. "
            "Keep the 2-wide bottleneck alive."};
    }

    const messaging::IEncodingScheme& get_scheme_for_slice(const SliceRef& slice) const override {
        (void)slice;
        return messaging::TotrRank1Scheme::instance();
    }

    const messaging::IEncodingScheme& get_bias_scheme(const SliceRef& slice) const override {
        (void)slice;
        return messaging::BiasDeltaScheme::instance();
    }

    const messaging::IOutputScheme& get_output_scheme() const override { return ReconstructionOutput::instance(); }

    std::string training_goal(double current_loss) const override {
        return "Target Loss: Minimize reconstruction error below 0.0100. Current Best Loss: " +
               EvalScheme::format_loss(current_loss);
    }

    SchemeNotes scheme_notes() const override {
        return SchemeNotes{
            "This autoencoder stores each full layer as a rank-1 outer product. Keep both latent units alive while following the reconstruction probe.",
            "Bias codes are hundredths of the decoded bias. A dead probe keeps A and B and raises each decoded bias by 0.2 to 0.5.",
            "The reconstruction target is the input row. The prompt does not include that grid."};
    }

    SliceScheme slice_scheme() const override { return SliceScheme{4, 8, 4}; }

    EvalScheme eval_scheme() const override { return EvalScheme{}; }

    const messaging::IContainerScheme& container_scheme() const override { return messaging::SJC1Compressor::instance(); }

    storage::JournalRecord seed_record(const ModelEvaluation& score, const std::string& checksum) const override {
        storage::JournalRecord record;
        record.loss = score.primary_loss;
        record.checksum = checksum;
        storage::MetricFields::append_score(record, score);
        record.metrics.push_back(storage::MetricFields::number("delta", 0.0));
        record.metrics.push_back(storage::MetricFields::text("winner", "seed"));
        record.metrics.push_back(storage::MetricFields::text("status", "seed"));
        return record;
    }

    storage::JournalRecord trial_record(const TrialNote& note) const override {
        storage::JournalRecord record;
        record.loss = note.loss;
        record.checksum = note.checksum;
        record.metrics.push_back(storage::MetricFields::number("delta", note.delta));
        if (note.record_winner) {
            record.metrics.push_back(storage::MetricFields::text("winner", note.winner));
        }
        record.metrics.push_back(storage::MetricFields::text("status", note.status));
        if (note.record_step) {
            record.metrics.push_back(storage::MetricFields::text("winner_step", note.winner_step, storage::MetricType::Token));
        }
        if (note.record_slice) {
            record.metrics.push_back(storage::MetricFields::text("slice", note.slice_id));
        }
        if (note.score != nullptr) {
            storage::MetricFields::append_score(record, *note.score);
        }
        return record;
    }

    std::string candidate_roles(const storage::JournalRecord* last) const override {
        const std::string status = last == nullptr ? "" : last->metric_text("status");
        const double delta = last == nullptr ? 0.0 : last->metric_number("delta", 0.0);
        const char* mandate =
            "Never copy active w except when every probe mark is 0. Otherwise mutate at least 2 ints in A and B. "
            "Keep both latent units alive.\n";
        if (status == "rejected_divergent" || delta > 0.0) {
            return std::string(mandate) +
                   "- Previous reconstruction step diverged or raised loss. Every candidate uses gamma <= 0.015.\n"
                   "- c1 gamma 0.005-0.010. c2 gamma 0.008-0.012. c3 gamma 0.010-0.015.";
        }
        if (delta < -0.1) {
            return std::string(mandate) +
                   "- Previous reconstruction step dropped loss by more than 0.1. Settle and protect the bottleneck.\n"
                   "- c1 and c2 gamma 0.01-0.03. c3 gamma 0.02-0.04.";
        }
        return std::string(mandate) +
               "- c1 (Gradient Follower): move opposite +2/+1/-2/-1. Gamma 0.015-0.035.\n"
               "- c2 (Momentum Step): follow the winning reconstruction step. Gamma 0.030-0.055.\n"
               "- c3 (Basin Explorer): larger step on steep marks. Gamma 0.050-0.080.";
    }

    std::string response_schema() const override {
        return TournamentPrompt::schema(
            candidate_scheme(),
            "1-2 sentences diagnosing the reconstruction probe and keeping the 2-wide bottleneck alive");
    }

    ModelDirective get_directives() const {
        return ModelDirective{
            "8-in reconstruction, 2-wide ReLU bottleneck, MSE.",
            "BottleneckAutoencoder",
            "c1 step 1-3. c2 follow Grad and raise sparsity. c3 g .05-.08 ints -25..25. Keep the bottleneck nonzero."};
    }

private:
    class ReconstructionOutput : public messaging::IOutputScheme {
    public:
        static const ReconstructionOutput& instance() {
            static const ReconstructionOutput scheme;
            return scheme;
        }

        messaging::EncodingDescriptor get_descriptor() const override {
            return messaging::EncodingDescriptor{
                "reconstruction_output",
                "Row-major fixed-4 floats. Rows are joined by ';'.",
                "Each stored row is one reconstruction sample.",
                "Values are finite floats."};
        }

        std::string serialize_output(const std::vector<std::vector<double>>& outputs) const override { return pack_rows(outputs); }

        std::vector<std::vector<double>> deserialize_output(const std::string& token, std::size_t rows, std::size_t cols) const override {
            return unpack_rows(token, rows, cols);
        }
    };

    NeuralNetwork net_;

    void install(NeuralNetwork& model) const {
        model.slices = slice_scheme();
        model.evaluation = eval_scheme();
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

    std::string probe_marks(const std::string& slice_id) const {
        const SliceRef slice = network().find_slice(slice_id);
        const std::string token = network().slice_token(slice);
        const double baseline = evaluate_model().primary_loss;
        std::ostringstream hint;
        hint << "A:[";
        write_marks(hint, token, "A:", slice, baseline);
        hint << "] B:[";
        write_marks(hint, token, "B:", slice, baseline);
        hint << "]";
        return hint.str();
    }

    void write_marks(std::ostringstream& hint,
                     const std::string& token,
                     const char* key,
                     const SliceRef& slice,
                     double baseline) const {
        const std::vector<int> codes = codes_of(token, key);
        for (std::size_t index = 0; index < codes.size(); ++index) {
            if (index > 0) {
                hint << ", ";
            }
            const int step = codes[index] >= 99 ? -1 : 1;
            const int probed = std::max(-99, std::min(99, codes[index] + step));
            NeuralNetwork trial = network();
            trial.patch_slice(slice.slice_id,
                              messaging::TotrEngine::decode_matrix(with_code(token, key, index, probed), slice.rows, slice.cols),
                              messaging::TotrEngine::decode_vector(network().slice_bias(slice), slice.cols));
            const double delta = score_patch(trial).primary_loss - baseline;
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

    static double latent_utilization(const NeuralNetwork& model, int sample_count) {
        if (model.layers.empty() || sample_count < 1) {
            return 0.0;
        }
        const std::size_t in_dim = model.layers.front().in_dim;
        double alive = 0.0;
        double total = 0.0;
        for (int index = 0; index < sample_count; ++index) {
            const std::vector<double> input = sample_input(index, sample_count, in_dim);
            for (double unit : model.layers.front().forward(input)) {
                total += 1.0;
                if (unit > 0.0) {
                    alive += 1.0;
                }
            }
        }
        return total == 0.0 ? 0.0 : alive / total;
    }
};

}  // namespace stevejob::models
