// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\demos\forward_pipeline_demo.cpp ===
// forward_pipeline_demo: one linear walk through the whole forward-analysis pipeline.
//
//   JSONL records (any shape)                                       e.g. syn_math_train_001: N=8 in=16 hidden=[24,12] out=4
//     -> TrainingDatasetImporter      -> TrainingSample             telemetry + ground truth + measured facts
//     -> ChannelFeatureExtractor      -> C x 25 channel tokens      C = 16 inputs + 36 hidden units + 3*4 outputs = 64
//     -> LinearInputAdapter           -> C x 256 sequence           64 x 256
//     -> PerceiverIOEncoder           -> Z, M x 256 latents         8 x 256 for every network, whatever its shape
//     -> AuxiliaryAlignmentHeads      -> 14-number descriptor + 4 eigen signs
//     -> AutoregressiveCausalDecoder  -> four observation text fields
//     -> ObservationConsistency       -> any field contradicting the measured facts is replaced by ObservationNarrator text
//     -> TrainingRunArtifactWriter    -> artifacts/run_YYYYMMDD_HHMMSS/ with run.json, CSVs and one JSON per record
//
// Usage: forward_pipeline_demo [train.jsonl] [eval.jsonl] [artifact_dir] [pretrain_epochs] [finetune_epochs]
//        (defaults: the data/ and artifacts/ folders next to CMakeLists.txt, 40 pretrain and 15 finetune epochs)

#include "forward_analysis/forward_analysis.hpp"  // umbrella header: every class in the library

#include <chrono>     // std::chrono::steady_clock, to time the training run
#include <exception>  // std::exception, caught at the bottom of main
#include <iomanip>    // std::setprecision, for 3-decimal number printing
#include <iostream>   // std::cout / std::cerr
#include <string>     // std::string paths and text fields
#include <vector>     // std::vector<TrainingSample>, std::vector<LossBreakdown>

// CMake defines FORWARD_ANALYSIS_DATA_DIR as the absolute path to offline_models/forward_analysis/data.
// This fallback only applies when the file is compiled without CMake.
#ifndef FORWARD_ANALYSIS_DATA_DIR
#define FORWARD_ANALYSIS_DATA_DIR "data"
#endif
// CMake defines FORWARD_ANALYSIS_ARTIFACT_DIR as offline_models/forward_analysis/artifacts.
#ifndef FORWARD_ANALYSIS_ARTIFACT_DIR
#define FORWARD_ANALYSIS_ARTIFACT_DIR "artifacts"
#endif

using namespace forward_analysis;  // all library types live in this namespace

int main(int argc, char** argv) {
    try {
        // ---------------------------------------------------------------------------------------------
        // 0. Settings
        // ---------------------------------------------------------------------------------------------

        // Training file: first command-line argument, or data/llm_train.jsonl (12 records, one JSON object per line).
        const std::string train_path = argc > 1 ? argv[1] : std::string(FORWARD_ANALYSIS_DATA_DIR) + "/llm_train.jsonl";
        // Held-out file: second command-line argument, or data/llm_eval.jsonl (4 records never seen during training).
        const std::string eval_path = argc > 2 ? argv[2] : std::string(FORWARD_ANALYSIS_DATA_DIR) + "/llm_eval.jsonl";
        // Parent folder for run outputs: third argument, or artifacts/. Each run gets its own run_<timestamp> subfolder.
        const std::string artifact_root = argc > 3 ? argv[3] : std::string(FORWARD_ANALYSIS_ARTIFACT_DIR);

        // Prints an Eigen vector on one line with 3 digits, e.g. a sign vector as "[1.000, -1.000, 1.000, 1.000]".
        const Eigen::IOFormat row(3, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
        // Step 1 length: passes over the training records with only the auxiliary losses (decoder frozen).
        // Fourth command-line argument, or 40.
        const std::size_t pretrain_epochs = argc > 4 ? std::stoul(argv[4]) : 40;
        // Step 2 upper bound: passes over the training records with CE + 0.3 Aux1 + 0.2 Aux2 (everything trains).
        // Training stops earlier once held-out token CE has not decreased for config.early_stopping_patience (5) epochs.
        // Fifth command-line argument, or 15.
        const std::size_t finetune_epochs = argc > 5 ? std::stoul(argv[5]) : 15;
        // SGD step size for the adapter, Perceiver and heads (the decoder uses Adam at config.decoder_learning_rate).
        const double encoder_learning_rate = 0.05;
        // Every double printed from here on shows 3 decimals, e.g. 0.545.
        std::cout << std::fixed << std::setprecision(3);

        // ---------------------------------------------------------------------------------------------
        // 1. Dataset loader: every record carries its own shape.
        // ---------------------------------------------------------------------------------------------

        // Model sizes and thresholds: d_model 256, num_latents 8, eigen_sign_dim 4, max_field_tokens 160,
        // dead_activation_threshold 1e-6, driver_correlation_threshold 0.75, ...
        DistillerConfig config;
        // Reads JSONL. For each line it validates the arrays, builds the telemetry tensor and measures facts.
        const TrainingDatasetImporter importer(config);

        // 12 TrainingSamples. Each one holds:
        //   sample_id            "syn_math_train_001"
        //   telemetry.inputs     N x d_in            e.g. 8 x 16
        //   telemetry.targets    N x d_out           e.g. 8 x 4
        //   telemetry.predictions N x d_out          e.g. 8 x 4
        //   telemetry.residuals  N x d_out           predictions - targets, derived on load
        //   telemetry.layer_activations  one N x width matrix per hidden layer, e.g. {8 x 24, 8 x 12}
        //   ground_truth_targets.eigen_signs  4 values of +1/-1, e.g. [1, 1, 1, 1]
        //   ground_truth_targets.observations four text fields (pathway_activity, feature_sensitivity,
        //                                     representational_structure, functional_mapping)
        //   measured            NetworkObservations computed from the numbers (dead units, rays, gating, drivers, outputs)
        const std::vector<TrainingSample> train = importer.load_jsonl(train_path);
        // 4 held-out TrainingSamples in the same layout, e.g. syn_math_eval_002: N=4 in=6 hidden=[9] out=2.
        const std::vector<TrainingSample> eval = importer.load_jsonl(eval_path);
        // Shape ranges across the batch, e.g.
        // "12 samples; rows 4-8, inputs 6-16, hidden layers 1-2 of width 8-24, outputs 1-4, channel tokens 17-68".
        std::cout << "train: " << DatasetShapeSummary::from(train).to_string() << "\n";
        // The same ranges for the held-out set, e.g. "4 samples; rows 4-8, inputs 6-16, ... channel tokens 21-57".
        std::cout << "eval:  " << DatasetShapeSummary::from(eval).to_string() << "\n";
        // One line per training record showing how its shape becomes a token count:
        // channel_count = d_in + sum(hidden widths) + 3 * d_out (each output gives target, prediction and residual tokens).
        for (const TrainingSample& sample : train) {
            // e.g. "syn_math_train_001  N=8 in=16 hidden=[24,12] out=4  -> 64 channel tokens"   (16 + 36 + 12 = 64)
            //      "syn_math_train_005  N=4 in=6 hidden=[8] out=1       -> 17 channel tokens"   (6 + 8 + 3 = 17)
            std::cout << "  " << sample.sample_id << "  " << sample.telemetry.shape().to_string() << "  -> "
                      << sample.telemetry.shape().channel_count() << " channel tokens\n";
        }

        // ---------------------------------------------------------------------------------------------
        // 2. First sample through channel features -> linear adapter -> Perceiver IO -> heads (untrained).
        //    These are standalone, freshly initialized components, separate from the bridge trained in section 5.
        // ---------------------------------------------------------------------------------------------

        // syn_math_train_001: N=8 in=16 hidden=[24,12] out=4.
        const TrainingSample& first = train.front();

        // Owns the LinearInputAdapter (256 x 25 weights + 256 biases) that turns channel features into d_model vectors.
        ChannelSequenceFormatter formatter(config);
        // C x 25 = 64 x 25. One row per column of telemetry, in this order: 16 inputs, 24 + 12 hidden units,
        // then target/prediction/residual for each of the 4 outputs. The 25 columns are: role one-hot (5),
        // depth, index sin/cos/fraction, tanh mean/std/min/max, zero/positive/negative fractions, correlation
        // with hidden axes 1 and 2, correlation with closed rows, correlation with the paired column, sign
        // agreement or residual ratio, log group width, log N, ray flag, constant flag. All values lie in [-1, 1].
        const Eigen::MatrixXd features = formatter.channel_features(first.telemetry);
        // C x 256 = 64 x 256: features * W^T + b. Row count follows the network; width is always d_model.
        const Eigen::MatrixXd sequence = formatter.format_sequence(features);
        // Cross-attention encoder: 8 learned latent queries (8 x 256) attend over the variable-length sequence.
        const PerceiverIOEncoder perceiver(config);
        // Forward pass that keeps intermediates for backprop:
        //   attn_weights 8 x 64   (each latent's softmax over the 64 channel tokens; each row sums to 1)
        //   context      8 x 256  (attention-weighted values)
        //   latents_z    8 x 256  (LayerNorm(queries + W_o * context + b_o)) — fixed size for any network
        const PerceiverForwardCache cache = perceiver.forward_with_cache(sequence);
        // Aux1 head (14 x 256 linear) and Aux2 head (4 x 256 linear + sigmoid), both reading the mean of Z's 8 rows.
        const AuxiliaryAlignmentHeads heads(config);
        // Untrained predictions and losses against this record's targets:
        //   target descriptor 14 numbers from the measured facts, target signs 4 values from the file.
        //   e.g. aux1 0.545 (half mean squared error), aux2 0.791 (binary cross-entropy), signs [1, -1, -1, 1].
        const AuxiliaryEvaluation untrained = heads.evaluate(
            cache.latents_z, ObservationDescriptor::encode(first.measured), first.ground_truth_targets.eigen_signs);
        // e.g. "syn_math_train_001: channel features 64x25 -> adapter sequence 64x256 -> attention 8x64 -> Z 8x256"
        std::cout << "\n" << first.sample_id << ": channel features " << features.rows() << "x" << features.cols()
                  << " -> adapter sequence " << sequence.rows() << "x" << sequence.cols() << " -> attention "
                  << cache.attn_weights.rows() << "x" << cache.attn_weights.cols() << " -> Z " << cache.latents_z.rows()
                  << "x" << cache.latents_z.cols() << "\n";
        // e.g. "untrained aux1 0.545, aux2 0.791, predicted signs [1.000, -1.000, -1.000, 1.000]"
        std::cout << "untrained aux1 " << untrained.aux1_observation_loss << ", aux2 " << untrained.aux2_eigen_sign_loss
                  << ", predicted signs " << untrained.predicted_eigen_signs.transpose().format(row) << "\n";

        // ---------------------------------------------------------------------------------------------
        // 3. Facts measured from the telemetry (the Aux1 target and narrator input).
        // ---------------------------------------------------------------------------------------------

        // Pretty-printed NetworkObservations for train_001, e.g.
        //   layers[0]: width 24, dead_units [3, 8, 14, 19, 22], effective_rank 1, lies_on_ray true
        //   layers[1]: width 12, dead_units [1, 5, 9], lies_on_ray true
        //   drivers: x9 falls, x12 / x0 / x5 rise (leading axis, strength ~0.98-0.99)
        //   gating: no closed samples;  outputs: 4 entries with correlation, amplitude_ratio, sign/order flags
        //   outputs_move_in_lockstep true;  eigen_signs [1, 1, 1, 1]
        std::cout << "\nmeasured observations:\n" << ObservationJsonWriter::write(first.measured) << "\n";
        // The 14-number Aux1 target compressed from those facts, e.g.
        // [0.208, 0.250, 0.053, 1.000, 0, 0, 0, 1.000, 0.994, 0, 0, 0.757, 0.497, 0]
        //  dead fraction first/last layer, rank/live, ray, gated, closed fraction, gate found,
        //  leading drivers/4, strongest driver, outputs keeping sign, keeping order, mean amplitude, mean correlation, flat.
        std::cout << "observation descriptor " << ObservationDescriptor::encode(first.measured).transpose().format(row)
                  << "\n";
        // Counts training records whose file eigen_signs equal the measured signs (descending eigenvalues of the
        // zero-diagonal covariance of the first four inputs).
        std::size_t sign_matches = 0;
        // Counts ground-truth text fields (12 records x 4 fields = 48) whose claims agree with the measured facts.
        std::size_t ground_truth_consistent_fields = 0;
        // Cross-check every training record's ground truth against what the telemetry actually shows.
        for (const TrainingSample& sample : train) {
            // Both are 4-vectors of +1/-1, e.g. file [1, 1, -1, 1] vs measured [1, 1, -1, 1].
            if (sample.ground_truth_targets.eigen_signs.isApprox(sample.measured.eigen_signs)) {
                // Signs agree.
                ++sign_matches;
            } else {
                // Signs disagree: print both vectors so the difference is visible.
                std::cout << "eigen_signs differ on " << sample.sample_id << ": file "
                          << sample.ground_truth_targets.eigen_signs.transpose().format(row) << ", measured "
                          << sample.measured.eigen_signs.transpose().format(row) << "\n";
            }
            // Check each of the 4 text fields separately (f = 0 pathway, 1 feature, 2 representational, 3 functional).
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                // Empty string when every checkable claim holds, otherwise the first contradiction, e.g.
                // "x13 is neither the gate coordinate nor a measured driver".
                const std::string reason = ObservationConsistency::violation(
                    sample.ground_truth_targets.observations.field(f), sample.measured);
                if (reason.empty()) {
                    // Field consistent with the measurements.
                    ++ground_truth_consistent_fields;
                } else {
                    // e.g. "ground truth syn_math_train_006.functional_mapping: x13 is neither the gate coordinate nor a measured driver"
                    std::cout << "ground truth " << sample.sample_id << "." << ObservationText::field_names()[f]
                              << ": " << reason << "\n";
                }
            }
        }
        // e.g. "file eigen_signs equal to measured eigen_signs on 12/12 train samples; ground-truth fields passing ... 40/48"
        std::cout << "file eigen_signs equal to measured eigen_signs on " << sign_matches << "/" << train.size()
                  << " train samples; ground-truth fields passing the consistency check "
                  << ground_truth_consistent_fields << "/" << train.size() * ObservationText::kFieldCount << "\n";

        // ---------------------------------------------------------------------------------------------
        // 4. Narration of the measured facts versus the ground-truth observation text.
        // ---------------------------------------------------------------------------------------------

        // Four fields written only from measured facts (no learning), e.g. pathway_activity:
        // "Five of the twenty-four first-layer units (units 3, 8, 14, 19 and 22) output exactly zero on every
        //  sample, leaving nineteen live units. ..."
        const ObservationText narrated = ObservationNarrator::describe(first.measured);
        // The hand-written target text for train_001 as a JSON object with the 4 field names as keys.
        std::cout << "\nground truth:\n" << first.ground_truth_targets.observations.to_json().dump(2) << "\n";
        // Word-overlap F1 (0 = no shared words, 1 = same words) averaged over the 4 fields, e.g. 0.513, then the narration.
        std::cout << "narrated from measured facts (word F1 "
                  << ObservationSimilarity::mean_f1(narrated, first.ground_truth_targets.observations) << "):\n"
                  << narrated.to_json().dump(2) << "\n";

        // ---------------------------------------------------------------------------------------------
        // 5. Step 1 (aux only, decoder frozen) and Step 2 (CE + 0.3 Aux1 + 0.2 Aux2).
        // ---------------------------------------------------------------------------------------------

        // The trainable pipeline: ChannelSequenceFormatter -> PerceiverIOEncoder -> {AuxiliaryAlignmentHeads, decoder}.
        DualObjectiveDistillationBridge bridge(config);
        // Rewrites each training field with slot tokens bound to that record's measured facts, e.g.
        //   "Nineteen of the twenty-four first-layer units ... the other five (units 3, 8, 14, 19 and 22)"
        //   -> "<COUNT:LIVE_L1|W> of the <DIM:W_1|w> <ORD:LAYER_1|wl> units ... ( units <LIST:DEAD_L1> )"
        // then registers every lexeme so the decoder's embedding (vocab x 256) and output (vocab x 256)
        // matrices have their final size before training.
        bridge.prepare_vocabulary(train);
        // e.g. "vocabulary 1100 tokens" (4 specials <PAD> <BOS> <EOS> <UNK>, words, punctuation and slot tokens).
        std::cout << "\nvocabulary " << bridge.decoder().tokenizer().vocab_size() << " tokens\n";
        // Where decoder training passes run (config.use_gpu and an OpenCL GPU with double precision), e.g.
        // "decoder accelerator NVIDIA GeForce RTX 4070 (NVIDIA CUDA, OpenCL 3.0 CUDA)" or "decoder accelerator cpu".
        std::cout << "decoder accelerator " << bridge.decoder().accelerator() << "\n";
        // The first training field after delexicalization, as the decoder is trained to write it.
        std::cout << "slot-token target " << first.sample_id << ".pathway_activity: "
                  << LexemeSplitter::join(DualObjectiveDistillationBridge::target_lexemes(first, 0)) << "\n";
        // Creates <artifact_root>/run_YYYYMMDD_HHMMSS/samples/ and collects everything recorded below.
        TrainingRunArtifactWriter artifacts(artifact_root, TrainingRunArtifactWriter::timestamped_name());
        // One LossBreakdown per step 1 epoch (40 entries): mean token CE (0 here), Aux1, Aux2, total over the training records.
        std::vector<LossBreakdown> pretrain;
        // Start the stopwatch for steps 1 and 2.
        const auto started = std::chrono::steady_clock::now();
        // Step 1, one epoch per iteration: 12 records, each features -> Z -> heads; gradients of
        // 0.3 Aux1 + 0.2 Aux2 flow back through the Perceiver and adapter.
        for (std::size_t epoch = 1; epoch <= pretrain_epochs; ++epoch) {
            // run_pretraining_phase with 1 epoch returns a single LossBreakdown (training-set means).
            pretrain.push_back(bridge.run_pretraining_phase(train, 1, encoder_learning_rate).front());
            // Same losses on the 4 held-out records with no weight updates; one loss_history.csv row,
            // e.g. "pretrain,1,0,0.1447,0.7672,0.1969,6.4483,0.0574,0.5732,6.5802"
            //       (train CE is 0 because the decoder is frozen; eval CE is measured but not trained).
            artifacts.record_epoch("pretrain", epoch, pretrain.back(), bridge.evaluate_dataset(eval));
        }
        // Step 2: teacher-forces the decoder on all 4 slot-token fields (each up to 160 tokens) with dropout 0.15,
        // so CE gradients reach every latent row of Z through the decoder's cross-attention. After each epoch the
        // held-out losses are measured; the phase stops when held-out token CE has not decreased for 5 epochs and
        // restores the weights of the epoch with the lowest held-out total loss.
        // FinetuneOutcome holds:
        //   history      one EpochRecord {epoch, train, eval, best_so_far} per epoch run, e.g. 9 entries
        //   best_epoch   e.g. 4 (the weights now in the bridge)
        //   best_eval    held-out losses at best_epoch
        //   stopped_early true when fewer than finetune_epochs ran
        const FinetuneOutcome finetune = bridge.run_finetuning_phase(train, finetune_epochs, encoder_learning_rate, eval);
        // One loss_history.csv row per fine-tuning epoch, e.g. "finetune,4,1.2345,0.0123,0.3456,1.3,2.9876,...".
        for (const EpochRecord& record : finetune.history) {
            artifacts.record_epoch("finetune", record.epoch, record.train, record.eval);
        }
        // Wall-clock seconds for both steps including the per-epoch held-out evaluation, e.g. 20.0.
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        // First vs last epoch of step 1, e.g. "step 1 aux1 0.140 -> 0.011, aux2 0.767 -> 0.385".
        std::cout << "step 1 aux1 " << pretrain.front().aux1_observation_loss << " -> " << pretrain.back().aux1_observation_loss
                  << ", aux2 " << pretrain.front().aux2_eigen_sign_loss << " -> " << pretrain.back().aux2_eigen_sign_loss << "\n";
        // First vs last epoch of step 2 on the training records, e.g. "step 2 train token CE 5.1 -> 1.2, total 5.2 -> 1.3".
        std::cout << "step 2 train token CE " << finetune.history.front().train.token_cross_entropy << " -> "
                  << finetune.history.back().train.token_cross_entropy << ", total "
                  << finetune.history.front().train.total_loss << " -> " << finetune.history.back().train.total_loss
                  << "  (" << seconds << " s)\n";
        // Held-out token CE per epoch, '*' marking each new lowest held-out total, e.g. "4.10* 3.80* 3.75* 3.90 ...".
        std::cout << "step 2 held-out token CE:";
        for (const EpochRecord& record : finetune.history) {
            std::cout << " " << record.eval.token_cross_entropy << (record.best_so_far ? "*" : "");
        }
        // e.g. "ran 9 of 15 epochs (stopped early), restored epoch 4 with held-out total 3.812"
        std::cout << "\nran " << finetune.history.size() << " of " << finetune_epochs << " epochs ("
                  << (finetune.stopped_early ? "stopped early" : "no early stop") << "), restored epoch "
                  << finetune.best_epoch << " with held-out total " << finetune.best_eval.total_loss << "\n";

        // ---------------------------------------------------------------------------------------------
        // 6. Step 3 on the training records: how much of the text the decoder reproduces by itself.
        // ---------------------------------------------------------------------------------------------

        // Sum of per-record mean F1 between the decoder's raw text and the ground truth.
        double train_decoder_f1 = 0.0;
        // How many of the 48 raw decoder fields pass the consistency check.
        std::size_t train_consistent_fields = 0;
        // Generate text for each training record (records the model has seen).
        for (const TrainingSample& sample : train) {
            // telemetry -> features (C x 25) -> sequence (C x 256) -> Z (8 x 256) -> heads + greedy decoding of 4 fields.
            const DistillationResult result = bridge.distill_telemetry(sample.telemetry);
            // Writes samples/syn_math_train_XXX.json and adds one samples.csv row and 4 fields.csv rows.
            artifacts.record_sample("train", sample, result);
            // Add this record's raw-decoder F1 (averaged over its 4 fields).
            train_decoder_f1 += ObservationSimilarity::mean_f1(result.decoder_observations, sample.ground_truth_targets.observations);
            // decoder_field_consistent is 4 booleans, one per field.
            for (bool ok : result.decoder_field_consistent) {
                // Count the fields whose raw decoder text agrees with the measured facts.
                train_consistent_fields += ok ? 1 : 0;
            }
        }
        // e.g. "train decoder word F1 0.973, consistent fields 39/48" — high because these records were memorized.
        std::cout << "train decoder word F1 " << train_decoder_f1 / static_cast<double>(train.size()) << ", consistent fields "
                  << train_consistent_fields << "/" << train.size() * ObservationText::kFieldCount << "\n";

        // ---------------------------------------------------------------------------------------------
        // 7. Step 3 on held-out records.
        // ---------------------------------------------------------------------------------------------

        // Each eval record has a shape the model never trained on, e.g. eval_001 N=8 in=16 hidden=[20,12] out=3.
        for (const TrainingSample& sample : eval) {
            // DistillationResult holds:
            //   channel_count                     e.g. 57 (16 + 32 + 9)
            //   latent_embeddings                 8 x 256
            //   predicted_observation_descriptor  14 numbers
            //   predicted_eigen_signs             4 values of +1/-1
            //   measured                          NetworkObservations from this telemetry
            //   decoder_slot_observations         4 fields as generated, slots unfilled, e.g. "<COUNT:DEAD_L1|W> of the ..."
            //   decoder_slots_filled              4 booleans: every slot had a value in this network's measurements
            //   decoder_observations              4 fields with slots filled from the measurements, e.g. "Seven of the ..."
            //   decoder_field_consistent          4 booleans (slots filled and the consistency check passed)
            //   observations                      final 4 fields: decoder text if consistent, otherwise narrator text
            //   decoder_token_counts              4 counts, each in [2, 160]
            const DistillationResult result = bridge.distill_telemetry(sample.telemetry);
            // Writes samples/syn_math_eval_XXX.json and adds one samples.csv row and 4 fields.csv rows.
            artifacts.record_sample("eval", sample, result);
            // The hand-written observation text for this held-out record, used only for scoring.
            const ObservationText& truth = sample.ground_truth_targets.observations;
            // e.g. "=== syn_math_eval_001  N=8 in=16 hidden=[20,12] out=3  (57 channel tokens)"
            std::cout << "\n=== " << sample.sample_id << "  " << sample.telemetry.shape().to_string() << "  ("
                      << result.channel_count << " channel tokens)\n";
            // Aux2 output vs file, e.g. "predicted signs [1.000, 1.000, 1.000, 1.000], target [1.000, -1.000, 1.000, -1.000]"
            std::cout << "predicted signs " << result.predicted_eigen_signs.transpose().format(row) << ", target "
                      << sample.ground_truth_targets.eigen_signs.transpose().format(row) << "\n";
            // Three lines per field: raw decoder text, final output text and ground truth, each with its F1.
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                // e.g. "[pathway_activity]"
                std::cout << "[" << ObservationText::field_names()[f] << "]\n"
                          // e.g. "  slots   (filled): <COUNT:LIVE_L1|W> of the <DIM:W_1|w> units ..."
                          << "  slots   (" << (result.decoder_slots_filled[f] ? "filled" : "unfillable")
                          << "): " << result.decoder_slot_observations.field(f) << "\n"
                          // e.g. "  decoder (inconsistent, F1 0.362): Seventeen of the twenty-four units ..."
                          << "  decoder (" << (result.decoder_field_consistent[f] ? "consistent" : "inconsistent")
                          << ", F1 " << ObservationSimilarity::field_f1(result.decoder_observations.field(f), truth.field(f))
                          << "): " << result.decoder_observations.field(f) << "\n"
                          // e.g. "  output  (F1 0.762): Seven of the twenty-four first-layer units (units 1, 5, 9, ...) ..."
                          << "  output  (F1 " << ObservationSimilarity::field_f1(result.observations.field(f), truth.field(f))
                          << "): " << result.observations.field(f) << "\n"
                          // e.g. "  truth: Seventeen of the twenty-four units fire on every sample and seven ... never fire."
                          << "  truth: " << truth.field(f) << "\n";
            }
            // Record-level summary, e.g. "mean F1 decoder 0.394, output 0.432".
            std::cout << "mean F1 decoder " << ObservationSimilarity::mean_f1(result.decoder_observations, truth) << ", output "
                      << ObservationSimilarity::mean_f1(result.observations, truth) << "\n";
        }

        // ---------------------------------------------------------------------------------------------
        // 8. Write the run artifacts.
        // ---------------------------------------------------------------------------------------------

        // How this run was launched; stored under "settings" in run.json.
        TrainingRunSettings settings;
        // Input files, e.g. ".../data/llm_train.jsonl" and ".../data/llm_eval.jsonl".
        settings.train_path = train_path;
        settings.eval_path = eval_path;
        // Epoch counts and step size used in section 5 (40, 80, 0.05).
        settings.pretrain_epochs = pretrain_epochs;
        settings.finetune_epochs = finetune_epochs;
        // Epochs actually run, the restored epoch and whether early stopping triggered, e.g. 9, 4, true.
        settings.finetune_epochs_run = finetune.history.size();
        settings.best_finetune_epoch = finetune.best_epoch;
        settings.stopped_early = finetune.stopped_early;
        settings.encoder_learning_rate = encoder_learning_rate;
        // Measured training time and decoder vocabulary size (e.g. 32.0 s, 589 tokens).
        settings.training_seconds = seconds;
        settings.vocabulary_size = bridge.decoder().tokenizer().vocab_size();
        // OpenCL device description or "cpu".
        settings.decoder_accelerator = bridge.decoder().accelerator();
        // Writes loss_history.csv (120 rows), samples.csv (16 rows), fields.csv (64 rows) and run.json
        // (settings, config, dataset shape ranges, per-split mean F1 / consistency / eigen-sign accuracy).
        artifacts.finish(settings, config, train, eval);
        // e.g. "artifacts written to C:/.../offline_models/forward_analysis/artifacts/run_20261013_224800"
        std::cout << "\nartifacts written to " << artifacts.directory().string() << "\n";
        // Success exit code.
        return 0;
    } catch (const std::exception& error) {
        // Any failure (missing file, malformed JSONL line reported as "path:line: reason", shape mismatch) ends up here.
        std::cerr << "forward_pipeline_demo failed: " << error.what() << "\n";
        // Failure exit code.
        return 1;
    }
}

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\experiments\forward_experiment.cpp ===
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

        // Reads and measures JSONL records.
        const TrainingDatasetImporter importer(config);
        // Training records (text permuted below when shuffle_text is set).
        std::vector<TrainingSample> train = importer.load_jsonl(train_path);
        // Selection records.
        const std::vector<TrainingSample> select = importer.load_jsonl(select_path);
        // Final-evaluation records.
        const std::vector<TrainingSample> final_eval = importer.load_jsonl(final_path);
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

// === C:\Users\Cutie Magic 500\stevejob\stevejob\offline_models\forward_analysis\tests\test_forward_analysis.cpp ===
#include "forward_analysis/forward_analysis.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FORWARD_ANALYSIS_DATA_DIR
#define FORWARD_ANALYSIS_DATA_DIR "data"
#endif
#ifndef FORWARD_ANALYSIS_FIXTURE_DIR
#define FORWARD_ANALYSIS_FIXTURE_DIR "tests/fixtures"
#endif

using namespace forward_analysis;

struct ForwardAnalysisTestSuite {
    // Number of require() checks that passed.
    std::size_t passed = 0;
    // Number of require() checks that failed.
    std::size_t failed = 0;
    // Model configuration shared by every check.
    DistillerConfig config;
    // Loaded training records (12, varying shapes).
    std::vector<TrainingSample> train;
    // Loaded held-out records (4, varying shapes).
    std::vector<TrainingSample> eval;

    void require(bool condition, const std::string& label) {
        if (condition) {
            ++passed;
        } else {
            ++failed;
            std::cout << "  FAIL " << label << "\n";
        }
    }

    void require_throws(const std::function<void()>& action, const std::string& label) {
        bool threw = false;
        try {
            action();
        } catch (const std::exception&) {
            threw = true;
        }
        require(threw, label);
    }

    static std::string data_path(const std::string& name) {
        return std::string(FORWARD_ANALYSIS_DATA_DIR) + "/" + name;
    }

    static std::string fixture_path(const std::string& name) {
        return std::string(FORWARD_ANALYSIS_FIXTURE_DIR) + "/" + name;
    }

    // The first `count` records of a JSONL file.
    static std::vector<TrainingSample> load_head(const TrainingDatasetImporter& importer, const std::string& path, std::size_t count) {
        std::ifstream in(path);
        std::string line;
        std::string head;
        for (std::size_t n = 0; n < count && std::getline(in, line); ++n) {
            head += line + "\n";
        }
        std::istringstream stream(head);
        return importer.read_jsonl(stream, path);
    }

    const TrainingSample& sample(const std::string& id) const {
        for (const std::vector<TrainingSample>* set : {&train, &eval}) {
            for (const TrainingSample& s : *set) {
                if (s.sample_id == id) {
                    return s;
                }
            }
        }
        throw std::runtime_error("missing sample " + id);
    }

    std::vector<const TrainingSample*> all_samples() const {
        std::vector<const TrainingSample*> out;
        for (const TrainingSample& s : train) {
            out.push_back(&s);
        }
        for (const TrainingSample& s : eval) {
            out.push_back(&s);
        }
        return out;
    }

    void verify_math() {
        std::cout << "math\n";
        Eigen::VectorXd a(4), b(4), ties(4);
        a << 1, 2, 3, 4;
        b << 2, 4, 6, 8;
        ties << 5, 1, 5, 2;
        require(std::abs(StatisticsOps::pearson(a, b) - 1.0) < 1e-12, "pearson of proportional vectors is 1");
        require(std::abs(StatisticsOps::pearson(a, Eigen::VectorXd::Constant(4, 3.0))) < 1e-12, "pearson with constant is 0");
        const Eigen::VectorXd r = StatisticsOps::ranks(ties);
        require(r(0) == 2.5 && r(2) == 2.5 && r(1) == 0.0 && r(3) == 1.0, "ranks average ties");
        require(StatisticsOps::same_order(a, b) && !StatisticsOps::same_order(a, -b), "same_order");

        Eigen::MatrixXd ray(3, 3);
        ray << 1, 2, 0, 2, 4, 0, 0.5, 1, 0;
        const Eigen::VectorXd s = SpectralOps::uncentered_singular_values(ray);
        require(s(1) < 1e-12 * s(0), "rank-1 matrix has one non-zero uncentered singular value");
        require(SpectralOps::principal_scores(ray, 2).rows() == 3 && SpectralOps::principal_scores(ray, 2).cols() == 2,
                "principal_scores is N x components");

        DeterministicRng rng(5);
        const Eigen::MatrixXd x = rng.xavier_matrix(3, 6) * 3.0;
        const Eigen::MatrixXd upstream = rng.xavier_matrix(3, 6);
        const Eigen::MatrixXd analytic = MatrixOps::layer_norm_rows_backward(
            MatrixOps::layer_norm_rows(x), MatrixOps::row_inverse_std(x), upstream);
        double worst = 0.0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            for (Eigen::Index j = 0; j < x.cols(); ++j) {
                Eigen::MatrixXd hi = x, lo = x;
                hi(i, j) += 1e-6;
                lo(i, j) -= 1e-6;
                const double numeric =
                    ((MatrixOps::layer_norm_rows(hi) - MatrixOps::layer_norm_rows(lo)).cwiseProduct(upstream)).sum() / 2e-6;
                worst = std::max(worst, std::abs(numeric - analytic(i, j)));
            }
        }
        require(worst < 1e-5, "layer_norm_rows_backward matches finite differences");

        AdamMoment moment;
        Eigen::MatrixXd param = Eigen::MatrixXd::Constant(2, 2, 3.0);
        for (std::uint64_t step = 1; step <= 400; ++step) {
            moment.apply(param, param, 0.05, step);
        }
        require(param.cwiseAbs().maxCoeff() < 0.5, "Adam minimizes a quadratic");
        Eigen::MatrixXd grown = Eigen::MatrixXd::Ones(3, 2);
        moment.apply(grown, grown, 0.01, 401);
        require(grown.rows() == 3, "AdamMoment grows with its parameter");
    }

    void verify_text_helpers() {
        std::cout << "text helpers\n";
        std::size_t v = 0;
        require(NumberWords::word(24) == "twenty-four" && NumberWords::parse("Twenty-four", v) && v == 24, "NumberWords");
        require(EnglishListFormatter::coordinates({0, 5, 12}) == "x0, x5 and x12", "EnglishListFormatter");
        const std::string text = "Units 3, 8 and 14 (of twenty-four) follow x0/x7; the rest: none.";
        require(LexemeSplitter::join(LexemeSplitter::split(text)) == text, "LexemeSplitter round trip");
        ObservationText same;
        same.pathway_activity = same.feature_sensitivity = same.representational_structure = same.functional_mapping = "a b c";
        ObservationText other = same;
        other.pathway_activity = "d e";
        require(std::abs(ObservationSimilarity::mean_f1(same, same) - 1.0) < 1e-12, "identical text has F1 1");
        require(ObservationSimilarity::field_f1("a b c", "d e") == 0.0, "disjoint text has F1 0");
        require(std::abs(ObservationSimilarity::mean_f1(other, same) - 0.75) < 1e-12, "mean F1 averages fields");
    }

    void verify_builder() {
        std::cout << "telemetry builder\n";
        const Eigen::MatrixXd in = Eigen::MatrixXd::Ones(4, 3);
        const Eigen::MatrixXd out = Eigen::MatrixXd::Ones(4, 2);
        const Eigen::MatrixXd hidden = Eigen::MatrixXd::Ones(4, 5);
        const ForwardTelemetryTensor t = TelemetryBuilder::build(in, out, out * 0.5, {hidden});
        require(t.shape().to_string() == "N=4 in=3 hidden=[5] out=2", "TelemetryShape string");
        require(t.shape().channel_count() == 3 + 5 + 3 * 2, "channel count = inputs + hidden units + 3 x outputs");
        require(t.residuals.isApprox(-0.5 * out), "residuals derived");
        require_throws([&] { TelemetryBuilder::build(in, Eigen::MatrixXd::Ones(3, 2), out, {hidden}); }, "row mismatch rejected");
        require_throws([&] { TelemetryBuilder::build(in, out, Eigen::MatrixXd::Ones(4, 3), {hidden}); }, "width mismatch rejected");
        require_throws([&] { TelemetryBuilder::build(in, out, out, {}); }, "missing hidden layer rejected");
        require_throws([&] { TelemetryBuilder::build(in, out, out, {Eigen::MatrixXd(4, 0)}); }, "zero-unit layer rejected");
        Eigen::MatrixXd bad = in;
        bad(0, 0) = std::nan("");
        require_throws([&] { TelemetryBuilder::build(bad, out, out, {hidden}); }, "non-finite rejected");
    }

    void verify_importer() {
        std::cout << "dataset importer\n";
        const TrainingDatasetImporter importer(config);
        train = importer.load_jsonl(fixture_path("llm_train.jsonl"));
        eval = importer.load_jsonl(fixture_path("llm_eval.jsonl"));
        require(train.size() == 40 && eval.size() == 10, "40 train and 10 eval records");
        std::set<std::size_t> input_dims;
        std::set<std::size_t> layer_counts;
        std::set<std::size_t> output_dims;
        for (const TrainingSample* s : all_samples()) {
            input_dims.insert(s->telemetry.shape().input_dim);
            layer_counts.insert(s->telemetry.num_layers());
            output_dims.insert(s->telemetry.shape().output_dim);
            require(s->ground_truth_targets.eigen_signs.size() == 4, s->sample_id + " has four eigen signs");
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                require(!s->ground_truth_targets.observations.field(f).empty(), s->sample_id + " field non-empty");
            }
        }
        require(input_dims.size() >= 5 && layer_counts.size() == 2 && output_dims.size() == 4, "shapes vary across records");
        const DatasetShapeSummary summary = DatasetShapeSummary::from(train);
        require(summary.min_input_dim == 6 && summary.max_input_dim == 16, "summary input range 6-16");

        const std::string good = R"({"sample_id":"s","telemetry":{"inputs":[[1,2],[3,4]],"targets":[[1],[2]],"predictions":[[1],[1]],"layer_activations":[[[0,1],[0,2]]]},"ground_truth_targets":{"target_text":{"pathway_activity":"a","feature_sensitivity":"b","representational_structure":"c","functional_mapping":"d"}},"extra":1})";
        std::istringstream one(good + "\n");
        const std::vector<TrainingSample> parsed = importer.read_jsonl(one, "inline");
        require(parsed.size() == 1 && parsed.front().ground_truth_targets.eigen_signs.size() == 4,
                "missing eigen_signs fall back to measured, unknown keys ignored");

        std::istringstream broken(good + "\n{not json}\n");
        bool located = false;
        try {
            importer.read_jsonl(broken, "inline");
        } catch (const std::exception& e) {
            located = std::string(e.what()).rfind("inline:2:", 0) == 0;
        }
        require(located, "parse errors report source:line");
        std::string no_field = good;
        no_field.replace(no_field.find("\"functional_mapping\":\"d\""), 24, "\"other\":\"d\"");
        std::istringstream missing(no_field);
        require_throws([&] { importer.read_jsonl(missing, "inline"); }, "missing observation field rejected");
        std::istringstream empty("\n\n");
        require_throws([&] { importer.read_jsonl(empty, "inline"); }, "empty dataset rejected");
    }

    void verify_extractor() {
        std::cout << "observation extractor\n";
        const NetworkObservations& t1 = sample("syn_math_train_001").measured;
        require(t1.layers[0].dead_units == std::vector<std::size_t>({3, 8, 14, 19, 22}), "train_001 dead units");
        require(t1.layers[0].lies_on_ray && t1.layers[1].lies_on_ray, "train_001 both layers on a ray");
        require(t1.outputs_move_in_lockstep, "train_001 outputs in lockstep");

        const NetworkObservations& t3 = sample("syn_math_train_003").measured;
        require(t3.gating.present() && t3.gating.has_gate_coordinate && t3.gating.gate_coordinate == 3, "train_003 gated by x3");
        require(t3.gating.closed_samples == std::vector<std::size_t>({1, 4, 5}), "train_003 closed samples");
        require(t3.gating.deeper_layers_constant, "train_003 deeper layer constant on closed samples");

        const NetworkObservations& t4 = sample("syn_math_train_004").measured;
        require(t4.outputs[0].sign_preserved && t4.outputs[0].order_preserved && t4.outputs[0].amplitude_ratio < 0.7,
                "train_004 output keeps sign and order but is damped");

        require(sample("syn_math_train_002").measured.layers[0].dead_units.size() == 11, "train_002 eleven dead units");
        const NetworkObservations& e4 = sample("syn_math_eval_004").measured;
        require(e4.layers[0].lies_on_ray && e4.layers[0].dead_units.size() == 7, "eval_004 ray with seven dead units");

        bool found_x10 = false;
        for (const InputDriver& d : sample("syn_math_train_010").measured.drivers) {
            found_x10 = found_x10 || d.coordinate == 10;
        }
        require(found_x10, "train_010 x10 drives the hidden plane");

        for (const TrainingSample* s : all_samples()) {
            require(s->measured.eigen_signs.size() == 4 && s->measured.eigen_signs.cwiseAbs().isApprox(Eigen::VectorXd::Ones(4)),
                    s->sample_id + " four measured eigen signs of +/-1");
            const Eigen::VectorXd d = ObservationDescriptor::encode(s->measured);
            require(d.size() == static_cast<Eigen::Index>(ObservationDescriptor::dim()) && d.allFinite(),
                    s->sample_id + " descriptor finite");
        }
    }

    // Measurement, checker and narrator against the corrected dataset (manifest corrected-2026-10-14).
    void verify_corrected_dataset() {
        std::cout << "corrected dataset measurements\n";
        Eigen::MatrixXd inputs(4, 4);
        inputs << 1, 2, 0, 1,
                  2, 1, 1, 0,
                  3, 4, 0, 2,
                  4, 3, 1, 1;
        Eigen::MatrixXd centered = inputs.rowwise() - inputs.colwise().mean();
        Eigen::MatrixXd covariance = centered.transpose() * centered / 3.0;
        covariance.diagonal().setZero();
        const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(covariance).eigenvalues().reverse();
        const Eigen::VectorXd expected = eigenvalues.unaryExpr([](double v) { return v >= 0.0 ? 1.0 : -1.0; });
        require(ObservationExtractor::eigen_signs(inputs, config).isApprox(expected),
                "eigen signs are the descending eigenvalue signs of the zero-diagonal input covariance");
        require(ObservationExtractor::eigen_signs(inputs.leftCols(2), config).tail(2).isApprox(Eigen::VectorXd::Ones(2)),
                "eigen signs beyond the input width stay +1");

        const TrainingDatasetImporter importer(config);
        for (const char* file : {"llm_train.jsonl", "llm_validation.jsonl", "llm_eval_ood.jsonl"}) {
            const std::string path = data_path(file);
            if (!std::filesystem::exists(path)) {
                std::cout << "  " << file << " not present, skipped\n";
                continue;
            }
            const std::vector<TrainingSample> head = load_head(importer, path, 200);
            std::size_t sign_match = 0;
            std::size_t truth_pass = 0;
            std::size_t narrated_exact = 0;
            // Fields whose batch-statistic slot lexemes relexicalize to the ground-truth lexemes.
            std::size_t slot_round_trips = 0;
            for (const TrainingSample& s : head) {
                sign_match += s.ground_truth_targets.eigen_signs.isApprox(s.measured.eigen_signs) ? 1 : 0;
                const ObservationText narrated = BatchStatisticsNarrator::describe(s.measured, config);
                for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                    const std::string& truth = s.ground_truth_targets.observations.field(f);
                    truth_pass += ObservationConsistency::verify_field(truth, s.measured) ? 1 : 0;
                    narrated_exact += narrated.field(f) == truth ? 1 : 0;
                    // Ground-truth field with every batch-statistic number replaced by its slot.
                    const std::vector<std::string> slots =
                        FieldDelexicalizer::delexicalize(truth, s.measured, SlotBinding::kBatchStatistics);
                    // Slots filled back from the measured telemetry.
                    std::string refilled;
                    slot_round_trips += ObservationDelexicalizer::relexicalize(slots, ShapeSlotLexicon::from(s.measured), refilled) &&
                                                LexemeSplitter::split(refilled) == LexemeSplitter::split(truth)
                                            ? 1
                                            : 0;
                    require(ObservationConsistency::verify_field(narrated.field(f), s.measured),
                            s.sample_id + " batch narration " + ObservationText::field_names()[f] + " consistent");
                }
            }
            std::cout << "  " << file << " first " << head.size() << ": eigen signs " << sign_match << ", ground-truth fields consistent "
                      << truth_pass << "/" << head.size() * 4 << ", narration identical to ground truth " << narrated_exact
                      << ", batch slots round-trip " << slot_round_trips << "\n";
            require(slot_round_trips + 4 >= head.size() * 4, std::string(file) + " batch-statistic slots refill to the ground truth");
            require(sign_match == head.size(), std::string(file) + " file eigen signs equal measured eigen signs");
            require(truth_pass + 4 >= head.size() * 4, std::string(file) + " ground truth passes the consistency check");
            require(narrated_exact + 4 >= head.size() * 4, std::string(file) + " batch narration reproduces the ground truth");
        }

        const TrainingSample& t4 = sample("syn_math_train_004");
        const NetworkObservations& m = t4.measured;
        const std::string truth = BatchStatisticsNarrator::describe(m, config).pathway_activity;
        require(ObservationConsistency::verify_field(truth, m), "batch pathway narration consistent");
        require(!ObservationConsistency::verify_field("Across 99 observations, the hidden widths are 3.", m),
                "wrong observation count rejected");
        require(!ObservationConsistency::verify_field(
                    "Layer 1 has " + std::to_string(m.layers[0].dead_units.size() + 1) + " coordinates that remain zero throughout the batch.", m),
                "wrong always-zero count rejected");
        require(!ObservationConsistency::verify_field("Layer 7 has 0 coordinates that remain zero throughout the batch.", m),
                "missing layer rejected");
        require(!ObservationConsistency::verify_field(
                    "Prediction and target signs agree in " + std::to_string(m.sign_agreements + 1) + " of " +
                        std::to_string(m.sign_entries) + " entries.", m),
                "wrong sign agreement rejected");
        require(ObservationConsistency::verify_field(
                    "Prediction and target signs agree in " + std::to_string(m.sign_agreements) + " of " +
                        std::to_string(m.sign_entries) + " entries.", m),
                "measured sign agreement accepted even when the counts exceed every layer width");
        require(!ObservationConsistency::verify_field(m.narrow_representation ? "The observed activation variation occupies multiple directions."
                                                                              : "This is a narrow observed representation.", m),
                "narrow / multiple-direction claim checked against numerical ranks");

        // Second-layer width slot, unfillable for a one-layer network.
        const std::vector<std::string> second_width =
            BatchStatementDelexicalizer::delexicalize("Across 5 observations, the hidden widths are 8 and 4.");
        require(second_width.size() > 2 && second_width[second_width.size() - 2] == "<COUNT:W_2|d>",
                "hidden-width list delexicalized positionally");
        // One-layer copy of the sample-004 observations.
        NetworkObservations one_layer = m;
        one_layer.layers.resize(1);
        // Text relexicalized for the one-layer observations.
        std::string unfilled;
        require(!ObservationDelexicalizer::relexicalize(second_width, ShapeSlotLexicon::from(one_layer), unfilled),
                "second-layer slot not fillable for a one-layer network");
    }

    void verify_encoder() {
        std::cout << "channel encoder\n";
        ChannelSequenceFormatter formatter(config);
        const PerceiverIOEncoder perceiver(config);
        for (const TrainingSample* s : all_samples()) {
            const Eigen::MatrixXd features = formatter.channel_features(s->telemetry);
            require(features.rows() == static_cast<Eigen::Index>(s->telemetry.shape().channel_count()) &&
                        features.cols() == static_cast<Eigen::Index>(ChannelFeatureExtractor::kFeatureDim) &&
                        features.allFinite() && features.cwiseAbs().maxCoeff() <= 1.0 + 1e-9,
                    s->sample_id + " channel features C x F, bounded");
            const Eigen::MatrixXd z = perceiver.encode(formatter.format_sequence(features));
            require(z.rows() == static_cast<Eigen::Index>(config.num_latents) &&
                        z.cols() == static_cast<Eigen::Index>(config.d_model) && z.allFinite(),
                    s->sample_id + " Z is M x d regardless of shape");
        }

        const TrainingSample& s = sample("syn_math_train_004");
        ForwardTelemetryTensor scaled = s.telemetry;
        Eigen::MatrixXd permuted_inputs = scaled.inputs;
        permuted_inputs.col(0).swap(permuted_inputs.col(1));
        scaled.inputs = permuted_inputs;
        const Eigen::MatrixXd original = perceiver.encode(formatter.format_sequence(s.telemetry));
        const Eigen::MatrixXd swapped = perceiver.encode(formatter.format_sequence(scaled));
        require((original - swapped).cwiseAbs().maxCoeff() < 0.5, "swapping two input columns changes Z only through index features");
    }

    void verify_description() {
        std::cout << "narrator and consistency\n";
        std::size_t truth_pass = 0;
        std::size_t truth_total = 0;
        for (const TrainingSample* s : all_samples()) {
            const ObservationText narrated = ObservationNarrator::describe(s->measured);
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                const std::string reason = ObservationConsistency::violation(narrated.field(f), s->measured);
                require(!narrated.field(f).empty() && reason.empty(),
                        s->sample_id + " narrated " + ObservationText::field_names()[f] + " consistent " + reason);
                truth_pass += ObservationConsistency::verify_field(s->ground_truth_targets.observations.field(f), s->measured);
                ++truth_total;
            }
        }
        std::cout << "  ground-truth fields passing the consistency check: " << truth_pass << "/" << truth_total << "\n";

        const NetworkObservations& t4 = sample("syn_math_train_004").measured;
        require(!ObservationConsistency::verify_field("x99 drives everything.", t4), "out-of-range coordinate rejected");
        require(!ObservationConsistency::verify_field("The second layer is flat.", t4), "second layer rejected on one-layer network");
        require(!ObservationConsistency::verify_field("The sign of x2 gates the layer.", t4), "gating rejected on ungated network");
        require(!ObservationConsistency::verify_field("Three of the ten units are dead.", t4), "wrong count pair rejected");
        require(!ObservationConsistency::verify_field("a b c d a b c d a b c d", t4), "repetition loop rejected");
        require(ObservationConsistency::verify_field("No unit is dead.", t4), "negated dead claim accepted");
        require(!ObservationConsistency::verify_field("The prediction is flat.", t4), "flat rejected when the output varies");
        require(!ObservationConsistency::verify_field("The prediction rises.", sample("syn_math_train_001").measured),
                "singular prediction rejected on a four-output network");

        ObservationTokenizer tokenizer;
        std::size_t round_trips = 0;
        for (const TrainingSample& s : train) {
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                const std::string& text = s.ground_truth_targets.observations.field(f);
                round_trips += tokenizer.decode(tokenizer.encode(text, 512, true)) == text ? 1 : 0;
            }
        }
        require(round_trips == train.size() * ObservationText::kFieldCount, "tokenizer round-trips every ground-truth field");
        require(tokenizer.encode_known("zzzunseen", 8)[1] == ObservationTokenizer::kUnkToken, "unseen word maps to UNK");
    }

    static std::vector<std::size_t> coordinate_mentions(const std::string& text) {
        std::vector<std::size_t> out;
        for (const std::string& lexeme : LexemeSplitter::split(text)) {
            const std::string w = LexemeSplitter::lowercase(lexeme);
            if (w.size() > 1 && w[0] == 'x' && w.find_first_not_of("0123456789", 1) == std::string::npos) {
                out.push_back(static_cast<std::size_t>(std::stoul(w.substr(1))));
            }
        }
        return out;
    }

    void verify_slots() {
        std::cout << "slots (improvement 1)\n";
        std::size_t round_trips = 0;
        std::size_t slot_count = 0;
        std::size_t literal_coordinates = 0;
        std::size_t literal_numbers_ge2 = 0;
        for (const TrainingSample* s : all_samples()) {
            const ShapeSlotLexicon lex = ShapeSlotLexicon::from(s->measured);
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                const std::string& text = s->ground_truth_targets.observations.field(f);
                const std::vector<std::string> slotted = ObservationDelexicalizer::delexicalize(text, lex);
                std::string filled;
                const bool ok = ObservationDelexicalizer::relexicalize(slotted, lex, filled);
                round_trips += ok && filled == text ? 1 : 0;
                if (!(ok && filled == text)) {
                    std::cout << "  round trip differs " << s->sample_id << "." << ObservationText::field_names()[f] << "\n    "
                              << text << "\n    " << filled << "\n";
                }
                for (const std::string& lexeme : slotted) {
                    std::size_t value = 0;
                    slot_count += ObservationDelexicalizer::is_slot(lexeme) ? 1 : 0;
                    literal_coordinates += coordinate_mentions(lexeme).size();
                    literal_numbers_ge2 += NumberWords::parse(lexeme, value) && value >= 2 ? 1 : 0;
                }
            }
        }
        const std::size_t field_total = (train.size() + eval.size()) * ObservationText::kFieldCount;
        require(round_trips == field_total, "delexicalize -> relexicalize restores every ground-truth field exactly");
        require(slot_count > 0, "ground truth contains slot-bound quantities");
        std::cout << "  " << round_trips << "/" << field_total << " fields round-trip; " << slot_count
                  << " slot tokens; literal coordinates left " << literal_coordinates << "; literal numbers >= 2 left "
                  << literal_numbers_ge2 << "\n";

        const TrainingSample& t1 = sample("syn_math_train_001");
        std::cout << "  train_001.pathway_activity slots: "
                  << LexemeSplitter::join(DualObjectiveDistillationBridge::target_lexemes(t1, 0)) << "\n";

        const std::vector<std::string> generated = LexemeSplitter::split(
            "<COUNT:DEAD_L1|W> of the <DIM:W_1|w> units ( units <LIST:DEAD_L1> ) never fire , and the intensity rises "
            "with <COORD:RISE_1> and <COORD:RISE_2> .");
        const TrainingSample& e4 = sample("syn_math_eval_004");
        std::string e4_text;
        require(ObservationDelexicalizer::relexicalize(generated, ShapeSlotLexicon::from(e4.measured), e4_text),
                "slot text fills on eval_004");
        std::cout << "  filled on eval_004 (" << e4.telemetry.shape().to_string() << "): " << e4_text << "\n";
        require(ObservationConsistency::verify_field(e4_text, e4.measured), "filled slot text is consistent with eval_004");

        const TrainingSample& t5 = sample("syn_math_train_005");
        std::string t5_text;
        const bool t5_ok = ObservationDelexicalizer::relexicalize(generated, ShapeSlotLexicon::from(t5.measured), t5_text);
        std::cout << "  same slots on train_005 (" << t5.telemetry.shape().to_string() << "): "
                  << (t5_ok ? t5_text : std::string("unfillable (no such quantity)")) << "\n";
        require(!t5_ok, "slot naming an absent quantity (no dead units on train_005) is reported unfillable");

        std::size_t literal_out_of_range = 0;
        std::size_t slot_out_of_range = 0;
        std::size_t filled_pairs = 0;
        for (const TrainingSample& source : train) {
            const ShapeSlotLexicon source_lex = ShapeSlotLexicon::from(source.measured);
            for (const TrainingSample* target : all_samples()) {
                const ShapeSlotLexicon target_lex = ShapeSlotLexicon::from(target->measured);
                const std::size_t d_in = target->telemetry.shape().input_dim;
                for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                    const std::string& text = source.ground_truth_targets.observations.field(f);
                    for (std::size_t c : coordinate_mentions(text)) {
                        literal_out_of_range += c >= d_in ? 1 : 0;
                    }
                    std::vector<std::string> slotted = ObservationDelexicalizer::delexicalize(text, source_lex);
                    std::vector<std::string> slot_only;
                    for (const std::string& lexeme : slotted) {
                        slot_only.push_back(ObservationDelexicalizer::is_slot(lexeme) ? lexeme : "w");
                    }
                    std::string filled;
                    if (ObservationDelexicalizer::relexicalize(slot_only, target_lex, filled)) {
                        ++filled_pairs;
                        for (std::size_t c : coordinate_mentions(filled)) {
                            slot_out_of_range += c >= d_in ? 1 : 0;
                        }
                    }
                }
            }
        }
        std::cout << "  training text copied verbatim onto every network: " << literal_out_of_range
                  << " out-of-range coordinate mentions; slot-filled (" << filled_pairs << " fillable field pairs): "
                  << slot_out_of_range << "\n";
        require(literal_out_of_range > 0 && slot_out_of_range == 0, "slot filling never names a coordinate the network lacks");
    }

    void verify_decoder() {
        std::cout << "decoder (improvements 2, 3, 4)\n";
        DistillerConfig small;
        small.d_model = 16;
        small.decoder_hidden_dim = 12;
        small.num_latents = 4;
        small.max_field_tokens = 24;
        small.decoder_dropout = 0.0;
        small.decoder_output_weight_decay = 0.0;
        const std::vector<std::string> lexemes = LexemeSplitter::split("<COUNT:DEAD_L1|W> of the <DIM:W_1|w> units never fire .");
        DeterministicRng rng(77);
        Eigen::MatrixXd z(4, 16);
        for (Eigen::Index r = 0; r < z.rows(); ++r) {
            for (Eigen::Index c = 0; c < z.cols(); ++c) {
                z(r, c) = rng.normal(0.0, 1.0);
            }
        }
        Eigen::VectorXd d_hat(static_cast<Eigen::Index>(ObservationDescriptor::dim()));
        for (Eigen::Index i = 0; i < d_hat.size(); ++i) {
            d_hat(i) = rng.uniform();
        }

        AutoregressiveCausalDecoder decoder(small, 5);
        decoder.prepare_vocabulary({lexemes});
        const Eigen::MatrixXd alpha = decoder.attention_weights(z, d_hat, 0, lexemes);
        require(alpha.rows() == static_cast<Eigen::Index>(lexemes.size() + 1) && alpha.cols() == 4,
                "attention is T x M (one weight per latent per step)");
        require((alpha.rowwise().sum().array() - 1.0).abs().maxCoeff() < 1e-12, "attention rows sum to 1");
        require((alpha.rowwise().maxCoeff() - alpha.rowwise().minCoeff()).maxCoeff() > 1e-6,
                "attention weights differ across latents (not a mean)");

        Eigen::MatrixXd grad;
        const double loss = decoder.loss_and_latent_gradient(z, d_hat, 1, lexemes, grad);
        double worst = 0.0;
        for (Eigen::Index r = 0; r < z.rows(); ++r) {
            for (Eigen::Index c = 0; c < z.cols(); c += 3) {
                Eigen::MatrixXd plus = z;
                Eigen::MatrixXd minus = z;
                plus(r, c) += 1e-5;
                minus(r, c) -= 1e-5;
                const double numeric =
                    (decoder.compute_loss(plus, d_hat, 1, lexemes) - decoder.compute_loss(minus, d_hat, 1, lexemes)) / 2e-5;
                worst = std::max(worst, std::abs(numeric - grad(r, c)) / std::max(1e-6, std::abs(numeric) + std::abs(grad(r, c))));
            }
        }
        std::cout << "  loss " << loss << ", dLoss/dZ finite-difference worst relative error " << worst << "\n";
        require(worst < 1e-4, "cross-attention backprop into Z matches finite differences");
        double row_spread = 0.0;
        for (Eigen::Index r = 1; r < grad.rows(); ++r) {
            row_spread = std::max(row_spread, (grad.row(r) - grad.row(0)).norm());
        }
        std::cout << "  max |dZ_r - dZ_0| over latent rows " << row_spread << " (mean pooling gives 0)\n";
        require(row_spread > 1e-6, "each latent row receives its own gradient");

        Eigen::VectorXd other = d_hat;
        other(0) += 1.0;
        other(5) -= 0.7;
        const double with_dhat = decoder.compute_loss(z, d_hat, 2, lexemes);
        const double with_other = decoder.compute_loss(z, other, 2, lexemes);
        std::cout << "  loss with d_hat " << with_dhat << ", with altered d_hat " << with_other << "\n";
        require(std::abs(with_dhat - with_other) > 1e-9, "predicted descriptor conditions the decoder (W_aux d_hat)");
        require_throws([&]() { decoder.compute_loss(z, Eigen::VectorXd::Zero(3), 0, lexemes); },
                       "descriptor of the wrong size rejected");

        DistillerConfig dropped = small;
        dropped.decoder_dropout = 0.15;
        AutoregressiveCausalDecoder with_dropout(dropped, 5);
        with_dropout.prepare_vocabulary({lexemes});
        with_dropout.set_frozen(false);
        const double eval_loss = with_dropout.compute_loss(z, d_hat, 0, lexemes);
        Eigen::MatrixXd sink;
        const double train_loss = with_dropout.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        AutoregressiveCausalDecoder without_dropout(small, 5);
        without_dropout.prepare_vocabulary({lexemes});
        without_dropout.set_frozen(false);
        const double plain_eval = without_dropout.compute_loss(z, d_hat, 0, lexemes);
        const double plain_train = without_dropout.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        std::cout << "  dropout 0.15: eval-mode loss " << eval_loss << ", training-pass loss " << train_loss
                  << "; dropout 0: " << plain_eval << " / " << plain_train << "\n";
        require(std::abs(eval_loss - train_loss) > 1e-9, "dropout active in training steps");
        require(std::abs(plain_eval - plain_train) < 1e-12, "no dropout in eval mode");

        DistillerConfig decayed = small;
        decayed.decoder_output_weight_decay = 0.1;
        AutoregressiveCausalDecoder a(decayed, 9);
        AutoregressiveCausalDecoder b(small, 9);
        a.prepare_vocabulary({lexemes});
        b.prepare_vocabulary({lexemes});
        a.set_frozen(false);
        b.set_frozen(false);
        a.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        b.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        const double shrink = 1.0 - decayed.decoder_learning_rate * 0.1;
        const double decay_error = (a.output_weights() - shrink * b.output_weights()).cwiseAbs().maxCoeff();
        std::cout << "  ||W_out||: decay 0.1 " << a.output_weights().norm() << ", decay 0 " << b.output_weights().norm()
                  << ", max |W_decay - (1 - lr*0.1) W_plain| " << decay_error << "\n";
        require(decay_error < 1e-12 && a.output_weights().norm() < b.output_weights().norm(),
                "decoupled weight decay shrinks the vocabulary projection");
    }

    static Eigen::MatrixXd random_matrix(DeterministicRng& rng, Eigen::Index rows, Eigen::Index cols) {
        Eigen::MatrixXd m(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) {
                m(r, c) = rng.normal(0.0, 1.0);
            }
        }
        return m;
    }

    static Eigen::VectorXd random_descriptor(DeterministicRng& rng) {
        Eigen::VectorXd d(static_cast<Eigen::Index>(ObservationDescriptor::dim()));
        for (Eigen::Index i = 0; i < d.size(); ++i) {
            d(i) = rng.uniform();
        }
        return d;
    }

    void verify_device_decoder() {
        std::cout << "decoder on the OpenCL device (CLBlast GEMM + OpenCL C kernels)\n";
        DistillerConfig small;
        small.d_model = 32;
        small.decoder_hidden_dim = 24;
        small.num_latents = 4;
        small.max_field_tokens = 40;
        DistillerConfig small_cpu = small;
        small_cpu.use_gpu = false;
        AutoregressiveCausalDecoder gpu(small, 11);
        AutoregressiveCausalDecoder cpu(small_cpu, 11);
        const std::string accelerator = gpu.accelerator();
        std::cout << "  accelerator: " << accelerator << "\n";
        require(cpu.accelerator() == "cpu", "use_gpu = false keeps the decoder on the CPU");
        if (accelerator == "cpu") {
            std::cout << "  no OpenCL GPU with double precision; device checks skipped\n";
            return;
        }

        const std::vector<std::vector<std::string>> sentences = {
            LexemeSplitter::split("<COUNT:DEAD_L1|W> of the <DIM:W_1|w> first-layer units never fire on any sample ."),
            LexemeSplitter::split("The output tracks x3 closely , with a correlation of 0.97 across samples ."),
            LexemeSplitter::split("Hidden activity spans two directions ; the leading one carries most of the variance ."),
        };
        const std::vector<std::string> unseen = LexemeSplitter::split("A gate closes whenever x5 drops below zero .");
        gpu.prepare_vocabulary(sentences);
        cpu.prepare_vocabulary(sentences);
        gpu.set_frozen(false);
        cpu.set_frozen(false);
        DeterministicRng rng(404);
        double worst_loss = 0.0;
        double worst_dz = 0.0;
        for (std::size_t step = 0; step < 9; ++step) {
            const Eigen::MatrixXd z = random_matrix(rng, 4, 32);
            const Eigen::VectorXd d_hat = random_descriptor(rng);
            const std::vector<std::string>& lexemes = step == 6 ? unseen : sentences[step % sentences.size()];
            Eigen::MatrixXd dz_gpu = Eigen::MatrixXd::Zero(4, 32);
            Eigen::MatrixXd dz_cpu = Eigen::MatrixXd::Zero(4, 32);
            const double loss_gpu = gpu.compute_loss_and_step(z, d_hat, step % 4, lexemes, dz_gpu);
            const double loss_cpu = cpu.compute_loss_and_step(z, d_hat, step % 4, lexemes, dz_cpu);
            worst_loss = std::max(worst_loss, std::abs(loss_gpu - loss_cpu));
            worst_dz = std::max(worst_dz, (dz_gpu - dz_cpu).cwiseAbs().maxCoeff());
        }
        std::cout << "  9 training steps with dropout 0.15 and decay 0.1 (step 7 adds new tokens): max |loss gpu - cpu| "
                  << worst_loss << ", max |dZ gpu - cpu| " << worst_dz << "\n";
        require(worst_loss < 1e-9, "device training-step losses match the CPU");
        require(worst_dz < 1e-9, "device dLoss/dZ matches the CPU");
        require(gpu.tokenizer().vocab_size() == cpu.tokenizer().vocab_size(), "vocabulary grows identically");

        const Eigen::MatrixXd z = random_matrix(rng, 4, 32);
        const Eigen::VectorXd d_hat = random_descriptor(rng);
        const double eval_gpu = gpu.compute_loss(z, d_hat, 1, sentences[1]);
        const double eval_cpu = cpu.compute_loss(z, d_hat, 1, sentences[1]);
        std::cout << "  eval-mode loss gpu " << eval_gpu << ", cpu " << eval_cpu << "\n";
        require(std::abs(eval_gpu - eval_cpu) < 1e-9, "device eval-mode loss matches the CPU");

        AutoregressiveCausalDecoder after_step(small, 11);
        after_step.prepare_vocabulary(sentences);
        after_step.set_frozen(false);
        Eigen::MatrixXd sink = Eigen::MatrixXd::Zero(4, 32);
        after_step.compute_loss_and_step(z, d_hat, 0, sentences[0], sink);
        require_throws([&]() { AutoregressiveCausalDecoder copy(after_step); },
                       "copying while the device holds newer parameters is rejected");

        const double weight_gap = (gpu.output_weights() - cpu.output_weights()).cwiseAbs().maxCoeff();
        const Eigen::MatrixXd attention_gap =
            gpu.attention_weights(z, d_hat, 2, sentences[2]) - cpu.attention_weights(z, d_hat, 2, sentences[2]);
        const std::vector<int> generated_gpu = gpu.generate_field_tokens(z, d_hat, 0, 40);
        const std::vector<int> generated_cpu = cpu.generate_field_tokens(z, d_hat, 0, 40);
        std::cout << "  after synchronizing: max |W_out gpu - cpu| " << weight_gap << ", max |attention gpu - cpu| "
                  << attention_gap.cwiseAbs().maxCoeff() << ", generated " << generated_gpu.size() << " / "
                  << generated_cpu.size() << " tokens\n";
        require(weight_gap < 1e-9, "synchronized device parameters match the CPU");
        require(attention_gap.cwiseAbs().maxCoeff() < 1e-9 && generated_gpu == generated_cpu,
                "generation from synchronized parameters matches the CPU");
        AutoregressiveCausalDecoder copy(gpu);
        require(std::abs(copy.compute_loss(z, d_hat, 1, sentences[1]) - eval_cpu) < 1e-9,
                "a copy made after synchronizing carries the trained parameters");

        DistillerConfig full;
        DistillerConfig full_cpu = full;
        full_cpu.use_gpu = false;
        AutoregressiveCausalDecoder full_gpu(full, 3);
        AutoregressiveCausalDecoder full_host(full_cpu, 3);
        std::vector<std::vector<std::string>> fields;
        for (std::size_t s = 0; s < 8; ++s) {
            std::vector<std::string> lexemes;
            for (std::size_t i = 0; i < 60; ++i) {
                lexemes.push_back("w" + std::to_string((s * 211 + i * 29) % 1700));
            }
            fields.push_back(lexemes);
        }
        full_gpu.prepare_vocabulary(fields);
        full_host.prepare_vocabulary(fields);
        full_gpu.set_frozen(false);
        full_host.set_frozen(false);
        const Eigen::MatrixXd z_full = random_matrix(rng, 8, 256);
        const Eigen::VectorXd d_full = random_descriptor(rng);
        Eigen::MatrixXd dz_full = Eigen::MatrixXd::Zero(8, 256);
        const auto time_steps = [&](AutoregressiveCausalDecoder& decoder) {
            decoder.compute_loss_and_step(z_full, d_full, 0, fields[0], dz_full);
            const auto started = std::chrono::steady_clock::now();
            for (std::size_t step = 1; step <= 16; ++step) {
                decoder.compute_loss_and_step(z_full, d_full, step % 4, fields[step % fields.size()], dz_full);
            }
            decoder.synchronize_host();
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() / 16.0;
        };
        const double cpu_seconds = time_steps(full_host);
        const double gpu_seconds = time_steps(full_gpu);
        std::cout << "  d_model 256, hidden 256, 8 latents, vocab " << full_gpu.tokenizer().vocab_size()
                  << ", 61-token fields: training step cpu " << cpu_seconds * 1000.0 << " ms, gpu " << gpu_seconds * 1000.0
                  << " ms (" << cpu_seconds / gpu_seconds << "x)\n";
        require(gpu_seconds < cpu_seconds, "device training step is faster than the CPU at full size");
    }

    void verify_training() {
        std::cout << "training\n";
        const std::vector<TrainingSample> subset(train.begin(), train.begin() + std::min<std::size_t>(12, train.size()));
        DualObjectiveDistillationBridge bridge(config);
        bridge.prepare_vocabulary(subset);
        const std::vector<LossBreakdown> pre = bridge.run_pretraining_phase(subset, 20, 0.05);
        require(pre.back().aux1_observation_loss < pre.front().aux1_observation_loss, "step 1 lowers Aux1");
        require(pre.back().aux2_eigen_sign_loss < pre.front().aux2_eigen_sign_loss, "step 1 lowers Aux2");
        require(pre.back().token_cross_entropy == 0.0, "decoder frozen in step 1");

        const double ce_before = bridge.evaluate_sample(subset.front()).token_cross_entropy;
        const FinetuneOutcome memorize = bridge.run_finetuning_phase(subset, 40, 0.05, subset);
        const double ce_after = bridge.evaluate_sample(subset.front()).token_cross_entropy;
        const LossBreakdown first_epoch = memorize.history.front().train;
        const LossBreakdown last_epoch = memorize.history.back().train;
        require(last_epoch.token_cross_entropy < 0.5 * first_epoch.token_cross_entropy, "step 2 lowers token CE");
        require(ce_after < ce_before, "evaluate_sample CE falls");
        std::cout << "  token CE " << first_epoch.token_cross_entropy << " -> " << last_epoch.token_cross_entropy << " over "
                  << memorize.history.size() << " epochs\n";

        double train_f1 = 0.0;
        std::size_t filled_fields = 0;
        for (const TrainingSample& s : subset) {
            const DistillationResult r = bridge.distill_telemetry(s.telemetry);
            train_f1 += ObservationSimilarity::mean_f1(r.decoder_observations, s.ground_truth_targets.observations);
            for (bool ok : r.decoder_slots_filled) {
                filled_fields += ok ? 1 : 0;
            }
        }
        train_f1 /= static_cast<double>(subset.size());
        std::cout << "  decoder word F1 on training records " << train_f1 << ", slot-filled fields " << filled_fields << "/"
                  << subset.size() * ObservationText::kFieldCount << "\n";
        require(train_f1 > 0.6, "decoder reproduces training observations");

        const FinetuneOutcome outcome = bridge.run_finetuning_phase(subset, 30, 0.05, eval);
        double lowest_total = 1e300;
        std::size_t lowest_epoch = 0;
        std::cout << "  held-out token CE per epoch:";
        for (const EpochRecord& r : outcome.history) {
            std::cout << " " << r.eval.token_cross_entropy << (r.best_so_far ? "*" : "");
            if (r.eval.total_loss < lowest_total) {
                lowest_total = r.eval.total_loss;
                lowest_epoch = r.epoch;
            }
        }
        std::cout << "\n  ran " << outcome.history.size() << "/30, stopped_early " << outcome.stopped_early << ", best epoch "
                  << outcome.best_epoch << "\n";
        bool trailing_without_decrease = true;
        if (outcome.stopped_early) {
            double best_before = 1e300;
            const std::size_t n = outcome.history.size();
            const std::size_t p = config.early_stopping_patience;
            for (std::size_t i = 0; i + p < n; ++i) {
                best_before = std::min(best_before, outcome.history[i].eval.token_cross_entropy);
            }
            for (std::size_t i = n - p; i < n; ++i) {
                trailing_without_decrease = trailing_without_decrease && outcome.history[i].eval.token_cross_entropy >= best_before;
            }
        }
        require(outcome.stopped_early ? trailing_without_decrease && outcome.history.size() < 30 : outcome.history.size() == 30,
                "early stopping fires only after 5 epochs without a held-out CE decrease");
        require(outcome.best_epoch == lowest_epoch, "best epoch is the lowest held-out total");
        const double restored_total = bridge.evaluate_dataset(eval).total_loss;
        std::cout << "  held-out total after restore " << restored_total << ", recorded at best epoch " << lowest_total << "\n";
        require(std::abs(restored_total - lowest_total) < 1e-9, "best-epoch weights restored");

        const std::filesystem::path artifact_root = std::filesystem::temp_directory_path() / "forward_analysis_test_artifacts";
        std::filesystem::remove_all(artifact_root);
        TrainingRunArtifactWriter artifacts_probe(artifact_root, "run_test");
        artifacts_probe.record_epoch("finetune", 1, outcome.history.back().train, outcome.history.back().eval);
        for (const TrainingSample& s : eval) {
            const DistillationResult r = bridge.distill_telemetry(s.telemetry);
            require(r.channel_count == s.telemetry.shape().channel_count(), s.sample_id + " channel count reported");
            require(r.predicted_eigen_signs.size() == 4 &&
                        r.predicted_observation_descriptor.size() == static_cast<Eigen::Index>(ObservationDescriptor::dim()),
                    s.sample_id + " aux predictions sized");
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                require(ObservationConsistency::verify_field(r.observations.field(f), r.measured),
                        s.sample_id + " final " + ObservationText::field_names()[f] + " consistent");
                require(r.decoder_token_counts[f] >= 2 && r.decoder_token_counts[f] <= config.max_field_tokens,
                        s.sample_id + " decoder budget respected");
            }
            artifacts_probe.record_sample("eval", s, r);
            std::cout << "  " << s.sample_id << " word F1 decoder "
                      << ObservationSimilarity::mean_f1(r.decoder_observations, s.ground_truth_targets.observations)
                      << ", output " << ObservationSimilarity::mean_f1(r.observations, s.ground_truth_targets.observations)
                      << "\n";
        }

        TrainingRunSettings settings;
        settings.finetune_epochs = 40;
        artifacts_probe.finish(settings, config, train, eval);
        const std::filesystem::path dir = artifacts_probe.directory();
        for (const char* name : {"run.json", "loss_history.csv", "samples.csv", "fields.csv", "samples/syn_math_eval_001.json"}) {
            require(std::filesystem::exists(dir / name), std::string("artifact ") + name + " written");
        }
        std::ifstream run_file(dir / "run.json");
        const nlohmann::json run = nlohmann::json::parse(run_file);
        require(run.at("metrics").at("eval").at("samples") == eval.size() &&
                    run.at("metrics").at("eval").at("total_fields") == eval.size() * ObservationText::kFieldCount,
                "run.json aggregates the eval split");
        std::ifstream fields_file(dir / "fields.csv");
        std::size_t field_lines = 0;
        for (std::string line; std::getline(fields_file, line);) {
            ++field_lines;
        }
        require(field_lines == 1 + eval.size() * ObservationText::kFieldCount,
                "fields.csv has a header and one row per record x field");
        run_file.close();
        fields_file.close();
        std::filesystem::remove_all(artifact_root);
    }
};

int main() {
    ForwardAnalysisTestSuite suite;
    try {
        suite.verify_math();
        suite.verify_text_helpers();
        suite.verify_builder();
        suite.verify_importer();
        suite.verify_extractor();
        suite.verify_corrected_dataset();
        suite.verify_encoder();
        suite.verify_description();
        suite.verify_slots();
        suite.verify_decoder();
        suite.verify_device_decoder();
        suite.verify_training();
    } catch (const std::exception& e) {
        std::cout << "unexpected exception: " << e.what() << "\n";
        ++suite.failed;
    }
    std::cout << suite.passed << " passed, " << suite.failed << " failed\n";
    return suite.failed == 0 ? 0 : 1;
}

