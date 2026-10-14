#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace forward_analysis {

// How a training run was launched, recorded alongside its artifacts.
struct TrainingRunSettings {
    // JSONL file the model trained on.
    std::string train_path;
    // JSONL file of held-out records.
    std::string eval_path;
    // Seed passed to DualObjectiveDistillationBridge.
    std::uint64_t seed = 20261013ULL;
    // Step 1 epochs (auxiliary losses only, decoder frozen).
    std::size_t pretrain_epochs = 0;
    // Step 2 upper bound on epochs (CE + Aux1 + Aux2, everything trains).
    std::size_t finetune_epochs = 0;
    // Step 2 epochs actually run before early stopping (or finetune_epochs when it never triggered).
    std::size_t finetune_epochs_run = 0;
    // Step 2 epoch whose weights were restored (lowest held-out total loss).
    std::size_t best_finetune_epoch = 0;
    // True when step 2 stopped early on held-out token cross-entropy.
    bool stopped_early = false;
    // SGD step size for the adapter, Perceiver and heads.
    double encoder_learning_rate = 0.0;
    // Wall-clock seconds spent in steps 1 and 2.
    double training_seconds = 0.0;
    // Decoder vocabulary size after prepare_vocabulary.
    std::size_t vocabulary_size = 0;
};

}  // namespace forward_analysis
