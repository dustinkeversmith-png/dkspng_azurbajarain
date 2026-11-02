#pragma once

#include "burger_dataset.hpp"
#include "burger_network.hpp"

#include "stevejob/messaging/sjc1_scheme.hpp"
#include "stevejob/models/model_template.hpp"
#include "stevejob/models/tournament_prompt.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

class BurgerOnlineModel : public ModelTemplate {
public:
    class BurgerOutputScheme final : public messaging::IOutputScheme {
    public:
        static const BurgerOutputScheme& instance() {
            static const BurgerOutputScheme inst;
            return inst;
        }

        messaging::EncodingDescriptor get_descriptor() const override {
            return {"burger_pde_output",
                    "Spatiotemporal velocity field prediction u(x,t).",
                    "u_hat = net(gamma(x, t))",
                    "Minimize MSE on Cole-Hopf targets while keeping PDE residual small."};
        }

        std::string serialize_output(const std::vector<std::vector<double>>& out) const override {
            std::ostringstream ss;
            ss.setf(std::ios::fixed);
            ss.precision(6);
            ss << (out.empty() || out[0].empty() ? 0.0 : out[0][0]);
            return ss.str();
        }

        std::vector<std::vector<double>> deserialize_output(const std::string&,
                                                            std::size_t rows,
                                                            std::size_t cols) const override {
            return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
        }
    };

    struct BurgerJournalScheme {
        storage::MetricField origin{"origin", storage::MetricType::Text, std::string("burger_base"),
                                    "Architecture that opened this Burgers checkpoint."};
        storage::MetricField delta{"delta", storage::MetricType::Number, 0.0,
                                   "Signed loss change against the last committed checkpoint."};
        storage::MetricField winner{"winner", storage::MetricType::Text, std::string("seed"),
                                    "Candidate slot that scored the lowest finite loss."};
        storage::MetricField seeded{"status", storage::MetricType::Text, std::string("seed"),
                                    "Trial result: seed, committed_natively, rejected, or rejected_divergent."};
        storage::MetricField step{"winner_step", storage::MetricType::Token, std::string(""),
                                  "Winning gamma and integer distance."};
        storage::MetricField slice{"slice", storage::MetricType::Text, std::string(""), "Edited slice id."};
        storage::MetricField extrap_mse{"extrap_mse", storage::MetricType::Number, 0.0,
                                        "Extrapolation MSE on t in [0.75, 1.0]."};
        storage::MetricField pde_residual{"pde_residual", storage::MetricType::Number, 0.0,
                                          "Mean interior Burgers PDE residual on a coarse grid."};

        JournalConfig config() const {
            JournalConfig journal;
            journal.capacity = 8;
            journal.task = "burger_pde_regression";
            journal.description =
                "Sliding window of Burgers tournament trials. Commit requires loss drop > 0.0001.";
            return journal;
        }

        std::string prompt_text() const {
            return "Journal scheme: task burger_pde_regression with delta, winner, status, extrap_mse, pde_residual.";
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

    explicit BurgerOnlineModel(int r_enc = 8, int width = 68, std::uint32_t seed = 2026)
        : r_enc_(r_enc), width_(width), seed_(seed), net_("burger_base_online") {
        config_.max_candidates = 3;
        config_.max_parameters = 12000;
        config_.max_blob_bytes = 2u * 1024u * 1024u;
        config_.max_input_rows = 128;
        config_.max_input_cols = static_cast<std::size_t>(r_enc_);
        config_.visible_history_count = 3;
        config_.int_bound = 25;
        config_.min_commit_drop = 1e-4;
        config_.divergence_threshold = 100.0;

        net_.architecture = "burger_base";
        // Tanh is the closest SJC1-supported smooth activation to SiLU for PDE-friendly derivatives.
        net_.add_layer("l1", static_cast<std::size_t>(r_enc_), static_cast<std::size_t>(width_), TanhActivation{})
            .add_layer("l2", static_cast<std::size_t>(width_), static_cast<std::size_t>(width_), TanhActivation{})
            .add_layer("l3", static_cast<std::size_t>(width_), static_cast<std::size_t>(width_), TanhActivation{})
            .add_layer("head", static_cast<std::size_t>(width_), 1, LinearActivation{});
        xavier_init(net_, seed_);
        install(net_);

        burger::BurgerDatasetGenerator::generate_splits(
            train_eval_data_, extrap_eval_data_, 128, 64, r_enc_, seed_);
    }

    void adopt(NeuralNetwork model) {
        model.architecture = "burger_base";
        install(model);
        net_ = std::move(model);
    }

    NeuralNetwork& network() override { return net_; }
    const NeuralNetwork& network() const override { return net_; }

    ModelEvaluation evaluate_model(int = 0) const override {
        return score_on(net_);
    }

    ModelEvaluation score_patch(const NeuralNetwork& candidate_net, int = 0) const override {
        return score_on(candidate_net);
    }

    std::vector<TelemetrySignal> derive_training_telemetry() const override {
        const double train_mse = mse_on(net_, train_eval_data_);
        const double extrap = mse_on(net_, extrap_eval_data_);
        const double pde = mean_pde_residual(net_);
        return {
            {core::DescribedField{"burgers_train_mse",
                                  core::FieldType::Scalar,
                                  train_mse,
                                  "telemetry",
                                  "Interpolation MSE on Cole-Hopf Burgers samples (t in [0, 0.75]).",
                                  "Prefer candidates that reduce this MSE."},
             nullptr},
            {core::DescribedField{"burgers_extrap_mse",
                                  core::FieldType::Scalar,
                                  extrap,
                                  "telemetry",
                                  "Extrapolation MSE on t in [0.75, 1.0].",
                                  "Watch for generalization; do not overfit the train window only."},
             nullptr},
            {core::DescribedField{"burgers_pde_residual",
                                  core::FieldType::Scalar,
                                  pde,
                                  "telemetry",
                                  "Mean squared Burgers residual u_t + u u_x - nu u_xx.",
                                  "Lower residual means better physics consistency."},
             nullptr},
        };
    }

    core::DescribedField derive_gradient_probe(const std::string& slice_id) const override {
        return core::DescribedField{
            "gradient",
            core::FieldType::Token,
            probe_marks(slice_id),
            "gradient_probe",
            "Directional marks +2/+1/0/-1/-2 for A then B factors on the active slice.",
            "Move integers opposite ascent marks; raise bias when marks are flat."};
    }

    const messaging::IEncodingScheme& get_scheme_for_slice(const SliceRef&) const override {
        return messaging::TotrRank1Scheme::instance();
    }

    const messaging::IEncodingScheme& get_bias_scheme(const SliceRef&) const override {
        return messaging::BiasDeltaScheme::instance();
    }

    const messaging::IOutputScheme& get_output_scheme() const override {
        return BurgerOutputScheme::instance();
    }

    CandidateScheme candidate_scheme() const override { return CandidateScheme{}; }

    std::string candidate_roles(const storage::JournalRecord* last) const override {
        const std::string status = last == nullptr ? "" : last->metric_text("status");
        const double delta = last == nullptr ? 0.0 : last->metric_number("delta", 0.0);
        const char* mandate =
            "Never copy active w except when every probe mark is 0. Otherwise mutate at least 2 ints in A and B.\n";
        if (status == "rejected_divergent" || delta > 0.0) {
            return std::string(mandate) +
                   "- Previous step diverged or raised loss. Use gamma <= 0.015 for all candidates.";
        }
        return std::string(mandate) +
               "- c1 conservative gamma 0.010-0.025.\n"
               "- c2 balanced gamma 0.025-0.045.\n"
               "- c3 exploratory gamma 0.045-0.075.";
    }

    std::string response_schema() const override {
        return TournamentPrompt::schema(candidate_scheme(),
                                        "1-2 sentences on Burgers residual / MSE and probe signs");
    }

    std::string training_goal(double current_loss) const override {
        std::ostringstream ss;
        ss.setf(std::ios::fixed);
        ss.precision(6);
        ss << "Approximate 1D viscous Burgers u(x,t) via Cole-Hopf targets. "
           << "Reduce train MSE from " << current_loss
           << " while keeping extrapolation and PDE residual controlled.";
        return ss.str();
    }

    SchemeNotes scheme_notes() const override {
        return {"Rank-1 outer-product weight slices for Burgers MLP layers.",
                "Bias codes are hundredths of decoded bias.",
                "Scalar field prediction u(x,t) from Fourier features of (x,t)."};
    }

    SliceScheme slice_scheme() const override {
        SliceScheme s;
        s.block = 16;
        s.long_side = static_cast<std::size_t>(width_);
        s.short_side = 16;
        return s;
    }

    EvalScheme eval_scheme() const override {
        EvalScheme s;
        s.samples = static_cast<int>(train_eval_data_.size());
        return s;
    }

    const messaging::IContainerScheme& container_scheme() const override {
        return messaging::SJC1Compressor::instance();
    }

    JournalConfig journal_config() const override { return journal_scheme_.config(); }

    std::string journal_scheme_text() const override { return journal_scheme_.prompt_text(); }

    storage::JournalRecord seed_record(const ModelEvaluation& score, const std::string& checksum) const override {
        storage::JournalRecord record;
        record.loss = score.primary_loss;
        record.checksum = checksum;
        storage::MetricFields::append_score(record, score);
        record.metrics.push_back(journal_scheme_.delta);
        record.metrics.push_back(journal_scheme_.winner);
        record.metrics.push_back(journal_scheme_.seeded);
        record.metrics.push_back(journal_scheme_.origin);
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.extrap_mse, mse_on(net_, extrap_eval_data_)));
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.pde_residual, mean_pde_residual(net_)));
        return record;
    }

    ModelDirective get_directives() const {
        return ModelDirective{
            "1D viscous Burgers PDE regression via Cole-Hopf targets.",
            "BurgerOnlineModel",
            "Use probe marks plus train/extrap MSE and PDE residual telemetry to choose c1/c2/c3."};
    }

    storage::JournalRecord trial_record(const TrialNote& note) const override {
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
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.extrap_mse, mse_on(net_, extrap_eval_data_)));
        record.metrics.push_back(
            journal_scheme_.with_number(journal_scheme_.pde_residual, mean_pde_residual(net_)));
        return record;
    }

private:
    int r_enc_ = 8;
    int width_ = 68;
    std::uint32_t seed_ = 2026;
    NeuralNetwork net_;
    BurgerJournalScheme journal_scheme_;
    std::vector<burger::DataPoint> train_eval_data_;
    std::vector<burger::DataPoint> extrap_eval_data_;

    void install(NeuralNetwork& model) const {
        model.slices = slice_scheme();
        model.evaluation = eval_scheme();
    }

    static void xavier_init(NeuralNetwork& model, std::uint32_t seed) {
        std::mt19937 rng(seed);
        for (LayerSpec& layer : model.layers) {
            const double limit =
                std::sqrt(6.0 / static_cast<double>(layer.in_dim + layer.out_dim));
            std::uniform_real_distribution<double> dist(-limit, limit);
            for (std::size_t r = 0; r < layer.weights.size(); ++r) {
                for (std::size_t c = 0; c < layer.weights[r].size(); ++c) {
                    layer.weights[r][c] = dist(rng);
                }
            }
            std::fill(layer.biases.begin(), layer.biases.end(), 0.0);
        }
    }

    static double mse_on(const NeuralNetwork& target, const std::vector<burger::DataPoint>& data) {
        double sum = 0.0;
        for (const auto& dp : data) {
            const std::vector<double> out = target.forward(dp.encoded_input);
            const double pred = out.empty() ? 0.0 : out[0];
            const double diff = pred - dp.u_true;
            sum += diff * diff;
        }
        return sum / static_cast<double>(std::max<std::size_t>(1, data.size()));
    }

    static double mean_pde_residual(const NeuralNetwork& target, int r_enc = 8) {
        const double nu = 0.01 / burger::BurgerDatasetGenerator::kPi;
        const double h = 1e-4;
        auto enc = [&](double x, double t) {
            return burger::BurgerDatasetGenerator::encode_coordinates(x, t, r_enc);
        };
        auto predict = [&](double x, double t) {
            const auto out = target.forward(enc(x, t));
            return out.empty() ? 0.0 : out[0];
        };
        double accum = 0.0;
        const int grid = 12;
        for (int i = 0; i < grid; ++i) {
            const double x = -0.8 + 1.6 * (static_cast<double>(i) / static_cast<double>(grid));
            const double t = 0.1 + 0.5 * (static_cast<double>(i) / static_cast<double>(grid));
            const double u = predict(x, t);
            const double u_t = (predict(x, t + h) - predict(x, t - h)) / (2.0 * h);
            const double u_x = (predict(x + h, t) - predict(x - h, t)) / (2.0 * h);
            const double u_xx = (predict(x + h, t) - 2.0 * u + predict(x - h, t)) / (h * h);
            const double res = u_t + u * u_x - nu * u_xx;
            accum += res * res;
        }
        return accum / static_cast<double>(grid);
    }

    ModelEvaluation score_on(const NeuralNetwork& candidate) const {
        const double mse = mse_on(candidate, train_eval_data_);
        ModelEvaluation score = ModelEvaluation::from_grid(mse, std::sqrt(mse), 0.01);
        for (core::DescribedField& field : score.metrics) {
            if (field.key == "loss") {
                field.format_description = "Train MSE vs Cole-Hopf Burgers targets.";
                field.actionable_rule = "Primary Burgers loss. Commit requires drop > 0.0001.";
            }
        }
        return score;
    }

    std::string probe_marks(const std::string& slice_id) const {
        const SliceRef slice = network().find_slice(slice_id);
        const std::string token = serialize_slice(slice);
        const std::string bias_token = serialize_bias(slice);
        const double baseline = score_on(net_).primary_loss;
        // Cap probe coordinates to keep Azure compile latency / token cost bounded.
        constexpr std::size_t kMaxProbe = 8;
        std::ostringstream hint;
        hint << "A:[";
        write_marks(hint, token, bias_token, "A:", slice, baseline, kMaxProbe);
        hint << "] B:[";
        write_marks(hint, token, bias_token, "B:", slice, baseline, kMaxProbe);
        hint << "]";
        return hint.str();
    }

    void write_marks(std::ostringstream& hint,
                     const std::string& token,
                     const std::string& bias_token,
                     const char* key,
                     const SliceRef& slice,
                     double baseline,
                     std::size_t max_probe) const {
        const std::vector<int> codes = codes_of(token, key);
        const std::vector<double> bias_values =
            get_bias_scheme(slice).deserialize_vector(bias_token, slice.cols);
        const std::size_t count = std::min(codes.size(), max_probe);
        for (std::size_t index = 0; index < count; ++index) {
            if (index > 0) {
                hint << ", ";
            }
            const int step = codes[index] >= 99 ? -1 : 1;
            const int probed = std::max(-99, std::min(99, codes[index] + step));
            NeuralNetwork trial = network();
            trial.patch_slice(
                slice.slice_id,
                get_scheme_for_slice(slice).deserialize_matrix(
                    with_code(token, key, index, probed), slice.rows, slice.cols),
                bias_values);
            const double delta = score_on(trial).primary_loss - baseline;
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
        if (start == std::string::npos) {
            return {};
        }
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
