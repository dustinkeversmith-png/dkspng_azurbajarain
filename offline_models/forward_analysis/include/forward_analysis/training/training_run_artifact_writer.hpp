#pragma once

#include "forward_analysis/analysis/observation_descriptor.hpp"
#include "forward_analysis/analysis/observation_json_writer.hpp"
#include "forward_analysis/bridge/distillation_result.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/domains/analysis_domain_traits.hpp"
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

        // Domain head predictions serialized as JSON.
        nlohmann::json domain_pred_json = nlohmann::json::object();
        for (const auto& [head_id, pred] : result.domain_predictions) {
            domain_pred_json[head_id] = {
                {"domain", AnalysisDomainTraits::name(pred.domain)},
                {"calibrated_outputs", vector_json(pred.calibrated_outputs)},
                {"discrete_predictions", vector_json(pred.discrete_predictions)},
                {"confidence", pred.confidence}};
        }
        const nlohmann::json detail = {
            {"split", split},
            {"sample_id", sample.sample_id},
            {"shape", shape},
            {"channel_count", result.channel_count},
            {"encoder_token_count", result.encoder_token_count},
            {"measured_observations", ObservationJsonWriter::to_json(result.measured)},
            {"descriptor", {{"target", vector_json(target_descriptor)},
                            {"predicted", vector_json(result.predicted_observation_descriptor)},
                            {"mean_absolute_error", descriptor_mae}}},
            {"eigen_signs", {{"target", vector_json(target_signs)},
                             {"predicted", vector_json(result.predicted_eigen_signs)},
                             {"matches", sign_matches}}},
            {"domain_predictions", domain_pred_json},
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
                {"slot_binding", c.slot_binding == SlotBinding::kBatchStatistics ? "batch_statistics" : "shape_facts"},
                {"feature_ablation_level", static_cast<int>(c.feature_ablation_level) + 1},
                {"enable_structural_decoder_constraints", c.enable_structural_decoder_constraints}};
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
