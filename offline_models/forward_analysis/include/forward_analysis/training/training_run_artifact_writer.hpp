#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/analysis/observation_json_writer.hpp"
#include "forward_analysis/bridge/distillation_result.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/dataset/dataset_shape_summary.hpp"
#include "forward_analysis/dataset/training_sample.hpp"
#include "forward_analysis/decoder/observation_delexicalizer.hpp"
#include "forward_analysis/decoder/shape_slot_lexicon.hpp"
#include "forward_analysis/description/observation_consistency.hpp"
#include "forward_analysis/description/observation_narrator.hpp"
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
    TrainingRunArtifactWriter(const std::filesystem::path& root, const std::string& run_name)
        : directory_((root / run_name).lexically_normal()) {
        std::filesystem::create_directories(directory_ / "samples");
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

    void record_sample(const std::string& split, const TrainingSample& sample, const DistillationResult& result) {
        const ObservationText& truth = sample.ground_truth_targets.observations;
        const ObservationText narrated = ObservationNarrator::describe(result.measured);
        const Eigen::VectorXd target_descriptor = ObservationDescriptor::encode(sample.measured);
        const Eigen::VectorXd& target_signs = sample.ground_truth_targets.eigen_signs;

        std::size_t sign_matches = 0;
        for (Eigen::Index i = 0; i < std::min(target_signs.size(), result.predicted_eigen_signs.size()); ++i) {
            sign_matches += target_signs(i) == result.predicted_eigen_signs(i) ? 1 : 0;
        }
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
            consistent_fields += consistent ? 1 : 0;
            decoder_tokens += result.decoder_token_counts[f];

            fields[name] = {{"target_slot_text", LexemeSplitter::join(ObservationDelexicalizer::delexicalize(
                                                     truth.field(f), ShapeSlotLexicon::from(sample.measured)))},
                            {"decoder_slot_text", result.decoder_slot_observations.field(f)},
                            {"decoder_slots_filled", filled},
                            {"decoder_text", result.decoder_observations.field(f)},
                            {"decoder_consistent", consistent},
                            {"decoder_violation", violation},
                            {"decoder_tokens", result.decoder_token_counts[f]},
                            {"narrated_text", narrated.field(f)},
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
                 CsvFieldFormatter::number(narrated_f1), CsvFieldFormatter::number(output_f1)}));
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

        SplitMetricTotals& totals = totals_[split];
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
        write_text(directory_ / "samples" / (sample.sample_id + ".json"), detail.dump(2));
    }

    void finish(const TrainingRunSettings& settings,
                const DistillerConfig& config,
                const std::vector<TrainingSample>& train,
                const std::vector<TrainingSample>& eval) const {
        write_csv("loss_history.csv",
                  "phase,epoch,train_token_ce,train_aux1,train_aux2,train_total,eval_token_ce,eval_aux1,eval_aux2,eval_total",
                  epoch_rows_);
        write_csv("samples.csv",
                  "split,sample_id,shape,channel_count,decoder_f1,narrated_f1,output_f1,decoder_consistent_fields,"
                  "narrator_fields,decoder_tokens,eigen_sign_matches,descriptor_mae",
                  sample_rows_);
        write_csv("fields.csv",
                  "split,sample_id,field,decoder_slots_filled,decoder_consistent,output_source,decoder_violation,decoder_tokens,decoder_f1,"
                  "narrated_f1,output_f1",
                  field_rows_);

        nlohmann::json metrics = nlohmann::json::object();
        for (const auto& [split, t] : totals_) {
            const double n = static_cast<double>(t.samples);
            metrics[split] = {{"samples", t.samples},
                              {"mean_decoder_f1", t.decoder_f1 / n},
                              {"mean_narrated_f1", t.narrated_f1 / n},
                              {"mean_output_f1", t.output_f1 / n},
                              {"decoder_consistent_fields", t.consistent_fields},
                              {"total_fields", t.samples * ObservationText::kFieldCount},
                              {"eigen_sign_accuracy", t.sign_total > 0 ? static_cast<double>(t.sign_matches) /
                                                                             static_cast<double>(t.sign_total)
                                                                       : 0.0},
                              {"mean_descriptor_mae", t.descriptor_mae / n}};
        }

        const nlohmann::json run = {
            {"run", directory_.filename().string()},
            {"settings", {{"train_path", settings.train_path},
                          {"eval_path", settings.eval_path},
                          {"seed", settings.seed},
                          {"pretrain_epochs", settings.pretrain_epochs},
                          {"finetune_epochs", settings.finetune_epochs},
                          {"finetune_epochs_run", settings.finetune_epochs_run},
                          {"best_finetune_epoch", settings.best_finetune_epoch},
                          {"stopped_early", settings.stopped_early},
                          {"encoder_learning_rate", settings.encoder_learning_rate},
                          {"training_seconds", settings.training_seconds},
                          {"vocabulary_size", settings.vocabulary_size}}},
            {"config", config_json(config)},
            {"dataset", {{"train", DatasetShapeSummary::from(train).to_json()},
                         {"eval", DatasetShapeSummary::from(eval).to_json()}}},
            {"metrics", metrics},
            {"files", {{"loss_history.csv", "per-epoch mean losses on the training records and on the held-out records"},
                       {"samples.csv", "one row per record: shape, word F1 (decoder / narrated / output), consistency, "
                                       "eigen-sign matches, descriptor mean absolute error"},
                       {"fields.csv", "one row per record and field: decoder consistency, first violation, tokens, F1"},
                       {"samples/", "one JSON per record with measured facts, aux targets vs predictions, all texts, latents"}}}};
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
                {"early_stopping_patience", c.early_stopping_patience},
                {"aux1_observation_weight", c.aux1_observation_weight},
                {"aux2_eigen_sign_weight", c.aux2_eigen_sign_weight},
                {"dead_activation_threshold", c.dead_activation_threshold},
                {"spectral_rank_tolerance", c.spectral_rank_tolerance},
                {"ray_tolerance", c.ray_tolerance},
                {"driver_correlation_threshold", c.driver_correlation_threshold},
                {"flat_output_ratio", c.flat_output_ratio}};
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
