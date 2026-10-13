#pragma once

#include "stevejob/models/evaluation_score.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <string>
#include <vector>

namespace stevejob::models {

struct JournalConfig {
    std::size_t capacity = 5;
    std::string task = "regression";
    std::string description =
        "Sliding window of tournament trials. loss is the scored error. "
        "A checkpoint write requires the loss to fall by more than 0.0001.";
};

struct TrialNote {
    std::string status;
    double loss = 0.0;
    double delta = 0.0;
    std::string checksum;
    std::string slice_id;
    std::string winner;
    std::string winner_step;
    bool record_winner = false;
    bool record_step = false;
    bool record_slice = false;
    const ModelEvaluation* score = nullptr;
};

struct CheckpointDocument {
    std::string status = "seeded";
    std::vector<stevejob::storage::MetricField> fields;
    bool include_layers = true;
    bool include_directives = true;
};

}  // namespace stevejob::models
