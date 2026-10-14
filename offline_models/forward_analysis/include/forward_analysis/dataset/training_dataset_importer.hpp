#pragma once

#include "forward_analysis/analysis/observation_extractor.hpp"
#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/dataset/json_tensor_reader.hpp"
#include "forward_analysis/dataset/training_sample.hpp"
#include "forward_analysis/telemetry/telemetry_builder.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace forward_analysis {

// Loads JSONL records of any shape: each line carries its own input width, hidden layer count and
// widths, output width and batch size.
class TrainingDatasetImporter {
public:
    explicit TrainingDatasetImporter(const DistillerConfig& config = DistillerConfig{})
        : config_(config) {
        config_.validate();
    }

    std::vector<TrainingSample> load_jsonl(const std::string& path) const {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Cannot open training dataset: " + path);
        }
        return read_jsonl(in, path);
    }

    std::vector<TrainingSample> read_jsonl(std::istream& in, const std::string& source = "<stream>") const {
        std::vector<TrainingSample> samples;
        std::string line;
        std::size_t line_number = 0;
        while (std::getline(in, line)) {
            ++line_number;
            if (line.find_first_not_of(" \t\r") == std::string::npos) {
                continue;
            }
            try {
                samples.push_back(parse_sample(nlohmann::json::parse(line)));
            } catch (const std::exception& error) {
                throw std::invalid_argument(source + ":" + std::to_string(line_number) + ": " + error.what());
            }
        }
        if (samples.empty()) {
            throw std::invalid_argument(source + " contains no samples");
        }
        return samples;
    }

    TrainingSample parse_sample(const nlohmann::json& record) const {
        TrainingSample sample;
        sample.sample_id = record.at("sample_id").get<std::string>();

        const nlohmann::json& telemetry = record.at("telemetry");
        sample.telemetry = TelemetryBuilder::build(
            JsonTensorReader::matrix(telemetry.at("inputs"), "telemetry.inputs"),
            JsonTensorReader::matrix(telemetry.at("targets"), "telemetry.targets"),
            JsonTensorReader::matrix(telemetry.at("predictions"), "telemetry.predictions"),
            JsonTensorReader::matrix_list(telemetry.at("layer_activations"), "telemetry.layer_activations"),
            config_);
        sample.measured = ObservationExtractor::extract(sample.telemetry, config_);

        const nlohmann::json& targets = record.at("ground_truth_targets");
        GroundTruthTargets& gt = sample.ground_truth_targets;
        gt.observations = ObservationText::from_json(targets.at("target_text"));
        gt.eigen_signs = targets.contains("eigen_signs")
                             ? JsonTensorReader::vector(targets.at("eigen_signs"), "ground_truth_targets.eigen_signs")
                             : sample.measured.eigen_signs;
        return sample;
    }

private:
    // Thresholds used when deriving masks and measured observations from imported telemetry.
    DistillerConfig config_;
};

}  // namespace forward_analysis
