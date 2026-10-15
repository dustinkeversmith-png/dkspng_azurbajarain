// forward_experiment: one training trial with separate selection and final-evaluation sets.
//
//   train.jsonl   -> step 1 (aux only) and step 2 (CE + Aux1 + Aux2) training records
//   select.jsonl  -> held-out losses after every epoch; early stopping and checkpoint selection use only this set
//   final.jsonl   -> distilled and scored once, after training (never seen during training or selection)
//
// Usage: forward_experiment key=value ...
//   trial=B description="..." train=PATH select=PATH final=PATH out=artifacts/experiments
//   pretrain=5 finetune=12 patience=3 aux2=0.2 dropout=0.15 hidden=256 decay=0.1 lr=0.05
//   shuffle_text=0 sample_files=0 seed=20261013 slots=shape|batch
// shuffle_text=1 permutes the observation text across training records (telemetry and eigen labels stay with their
// record), so the decoder trains on text that does not describe the telemetry it reads.

#include "forward_analysis/forward_analysis.hpp"  // umbrella header: every class in the library

#include <chrono>     // std::chrono::steady_clock, to time training
#include <exception>  // std::exception, caught at the bottom of main
#include <iostream>   // std::cout / std::cerr
#include <map>        // std::map of command-line options
#include <string>     // std::string option values and paths
#include <utility>    // std::swap, std::pair
#include <vector>     // std::vector<TrainingSample>

// CMake defines both folders as absolute paths; these fallbacks apply only when compiled without CMake.
#ifndef FORWARD_ANALYSIS_DATA_DIR
#define FORWARD_ANALYSIS_DATA_DIR "data"
#endif
#ifndef FORWARD_ANALYSIS_ARTIFACT_DIR
#define FORWARD_ANALYSIS_ARTIFACT_DIR "artifacts"
#endif

using namespace forward_analysis;  // all library types live in this namespace

int main(int argc, char** argv) {
    try {
        // Every "key=value" argument, e.g. {"trial": "B", "pretrain": "5"}.
        std::map<std::string, std::string> options;
        for (int a = 1; a < argc; ++a) {
            // One argument such as "finetune=12".
            const std::string argument = argv[a];
            // Position of '=' splitting key from value.
            const std::size_t equals = argument.find('=');
            if (equals == std::string::npos) {
                throw std::invalid_argument("expected key=value, got " + argument);
            }
            options[argument.substr(0, equals)] = argument.substr(equals + 1);
        }
        // Value of an option, or fallback when it was not given.
        const auto option = [&options](const std::string& key, const std::string& fallback) {
            const auto it = options.find(key);
            return it == options.end() ? fallback : it->second;
        };

        // Default data folder (offline_models/forward_analysis/data) set by CMake.
        const std::string data_dir = FORWARD_ANALYSIS_DATA_DIR;
        // Trial label written into run.json, e.g. "B".
        const std::string trial = option("trial", "B");
        // What the trial changes, e.g. "selection on in-distribution validation".
        const std::string description = option("description", "");
        // Training records.
        const std::string train_path = option("train", data_dir + "/llm_train.jsonl");
        // Records for per-epoch held-out losses, early stopping and checkpoint selection.
        const std::string select_path = option("select", data_dir + "/llm_validation.jsonl");
        // Records scored only after training.
        const std::string final_path = option("final", data_dir + "/llm_eval_ood.jsonl");
        // Parent folder of the run folder.
        const std::string out_root = option("out", std::string(FORWARD_ANALYSIS_ARTIFACT_DIR) + "/experiments");
        // Step 1 epochs (auxiliary losses only, decoder frozen).
        const std::size_t pretrain_epochs = std::stoul(option("pretrain", "5"));
        // Step 2 epoch limit.
        const std::size_t finetune_epochs = std::stoul(option("finetune", "12"));
        // SGD step size for adapter, Perceiver and heads.
        const double encoder_learning_rate = std::stod(option("lr", "0.05"));
        // Permute the training text across records (1) or keep each record's own text (0).
        const bool shuffle_text = option("shuffle_text", "0") == "1";
        // Write samples/<id>.json per record (1) or only the CSVs and run.json (0).
        const bool sample_files = option("sample_files", "0") == "1";
        // Bridge initialization seed.
        const std::uint64_t seed = std::stoull(option("seed", "20261013"));

        // Model configuration: library defaults with the trial's overrides.
        DistillerConfig config;
        config.early_stopping_patience = std::stoul(option("patience", "3"));
        config.aux2_eigen_sign_weight = std::stod(option("aux2", "0.2"));
        config.decoder_dropout = std::stod(option("dropout", "0.15"));
        config.decoder_hidden_dim = std::stoul(option("hidden", "256"));
        config.decoder_output_weight_decay = std::stod(option("decay", "0.1"));
        config.slot_binding = option("slots", "shape") == "batch" ? SlotBinding::kBatchStatistics : SlotBinding::kShapeFacts;
        // Feature ablation level in [1, 5] (default 5 = complete multi-family feature architecture).
        const int ablation_level = std::stoi(option("ablation", "5"));
        if (ablation_level >= 1 && ablation_level <= 5) {
            config.feature_ablation_level = static_cast<FeatureAblationLevel>(ablation_level - 1);
        }
        config.enable_structural_decoder_constraints = option("constraints", "1") != "0";

        // Reads and measures JSONL records.
        const TrainingDatasetImporter importer(config);
        // Training records (text permuted below when shuffle_text is set).
        std::vector<TrainingSample> train = importer.load_jsonl(train_path);
        // Selection records.
        std::vector<TrainingSample> select = importer.load_jsonl(select_path);
        // Final-evaluation records.
        std::vector<TrainingSample> final_eval = importer.load_jsonl(final_path);
        // Optional sample count cap for rapid ablation studies (0 = full dataset).
        const std::size_t max_train = std::stoul(option("max_train", "0"));
        // Optional selection/final sample count cap (0 = full dataset).
        const std::size_t max_eval = std::stoul(option("max_eval", "0"));
        if (max_train > 0 && train.size() > max_train) {
            train.resize(max_train);
        }
        if (max_eval > 0 && select.size() > max_eval) {
            select.resize(max_eval);
        }
        if (max_eval > 0 && final_eval.size() > max_eval) {
            final_eval.resize(max_eval);
        }
        if (shuffle_text) {
            // Fisher-Yates over the observation text only.
            DeterministicRng rng(seed + 99ULL);
            for (std::size_t i = train.size(); i > 1; --i) {
                // Record whose text moves to position i - 1.
                const std::size_t j = static_cast<std::size_t>(rng.next_u64() % i);
                std::swap(train[i - 1].ground_truth_targets.observations, train[j].ground_truth_targets.observations);
            }
        }
        std::cout << "trial " << trial << ": train " << train.size() << ", select " << select.size() << ", final "
                  << final_eval.size() << (shuffle_text ? " (training text shuffled)" : "") << "\n";

        // The trainable pipeline.
        DualObjectiveDistillationBridge bridge(config, seed);
        bridge.prepare_vocabulary(train);
        std::cout << "vocabulary " << bridge.decoder().tokenizer().vocab_size() << ", decoder "
                  << bridge.decoder().accelerator() << "\n";
        // Run folder, e.g. artifacts/experiments/trial_B_run_20261014_140000.
        TrainingRunArtifactWriter artifacts(out_root, "trial_" + trial + "_" + TrainingRunArtifactWriter::timestamped_name(),
                                            sample_files);
        // Start of steps 1 and 2.
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t epoch = 1; epoch <= pretrain_epochs; ++epoch) {
            // Training-set mean losses of this step 1 epoch.
            const LossBreakdown loss = bridge.run_pretraining_phase(train, 1, encoder_learning_rate).front();
            artifacts.record_epoch("pretrain", epoch, loss, bridge.evaluate_dataset(select));
        }
        // Step 2 history, restored epoch and early-stop flag.
        const FinetuneOutcome finetune = bridge.run_finetuning_phase(train, finetune_epochs, encoder_learning_rate, select);
        for (const EpochRecord& record : finetune.history) {
            artifacts.record_epoch("finetune", record.epoch, record.train, record.eval);
            std::cout << "  epoch " << record.epoch << " train CE " << record.train.token_cross_entropy << ", select CE "
                      << record.eval.token_cross_entropy << ", select total " << record.eval.total_loss
                      << (record.best_so_far ? " *" : "") << "\n";
        }
        // Seconds spent in steps 1 and 2.
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "restored epoch " << finetune.best_epoch << " of " << finetune.history.size() << " ("
                  << seconds << " s)\n";

        // Distill and score every split with the restored weights.
        for (const auto& [split, records] : std::vector<std::pair<std::string, const std::vector<TrainingSample>*>>{
                 {"select", &select}, {"final", &final_eval}, {"train", &train}}) {
            for (const TrainingSample& sample : *records) {
                artifacts.record_sample(split, sample, bridge.distill_telemetry(sample.telemetry), config.slot_binding);
            }
        }

        // How this run was launched.
        TrainingRunSettings settings;
        settings.train_path = train_path;
        settings.eval_path = select_path;
        settings.final_eval_path = final_path;
        settings.trial = trial;
        settings.trial_description = description;
        settings.shuffled_training_text = shuffle_text;
        settings.seed = seed;
        settings.pretrain_epochs = pretrain_epochs;
        settings.finetune_epochs = finetune_epochs;
        settings.finetune_epochs_run = finetune.history.size();
        settings.best_finetune_epoch = finetune.best_epoch;
        settings.stopped_early = finetune.stopped_early;
        settings.encoder_learning_rate = encoder_learning_rate;
        settings.training_seconds = seconds;
        settings.vocabulary_size = bridge.decoder().tokenizer().vocab_size();
        settings.decoder_accelerator = bridge.decoder().accelerator();
        artifacts.finish(settings, config, {{"train", &train}, {"select", &select}, {"final", &final_eval}});
        std::cout << "artifacts written to " << artifacts.directory().string() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "forward_experiment failed: " << error.what() << "\n";
        return 1;
    }
}
