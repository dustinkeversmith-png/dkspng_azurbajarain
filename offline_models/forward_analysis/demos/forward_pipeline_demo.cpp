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
