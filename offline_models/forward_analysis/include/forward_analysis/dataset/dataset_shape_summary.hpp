#pragma once

#include "forward_analysis/dataset/training_sample.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

// Range of every shape dimension across a loaded batch of samples.
struct DatasetShapeSummary {
    // Number of samples summarized.
    std::size_t sample_count = 0;
    // Smallest and largest batch size (rows) seen.
    std::size_t min_rows = std::numeric_limits<std::size_t>::max();
    std::size_t max_rows = 0;
    // Smallest and largest input width seen.
    std::size_t min_input_dim = std::numeric_limits<std::size_t>::max();
    std::size_t max_input_dim = 0;
    // Smallest and largest hidden layer count seen.
    std::size_t min_layers = std::numeric_limits<std::size_t>::max();
    std::size_t max_layers = 0;
    // Smallest and largest single hidden layer width seen.
    std::size_t min_hidden_width = std::numeric_limits<std::size_t>::max();
    std::size_t max_hidden_width = 0;
    // Smallest and largest output width seen.
    std::size_t min_output_dim = std::numeric_limits<std::size_t>::max();
    std::size_t max_output_dim = 0;
    // Smallest and largest channel-token count (encoder sequence length) seen.
    std::size_t min_channels = std::numeric_limits<std::size_t>::max();
    std::size_t max_channels = 0;

    static DatasetShapeSummary from(const std::vector<TrainingSample>& samples) {
        DatasetShapeSummary s;
        for (const TrainingSample& sample : samples) {
            const TelemetryShape shape = sample.telemetry.shape();
            s.sample_count += 1;
            widen(s.min_rows, s.max_rows, shape.num_samples);
            widen(s.min_input_dim, s.max_input_dim, shape.input_dim);
            widen(s.min_layers, s.max_layers, shape.hidden_widths.size());
            for (std::size_t w : shape.hidden_widths) {
                widen(s.min_hidden_width, s.max_hidden_width, w);
            }
            widen(s.min_output_dim, s.max_output_dim, shape.output_dim);
            widen(s.min_channels, s.max_channels, shape.channel_count());
        }
        return s;
    }

    nlohmann::json to_json() const {
        return {{"sample_count", sample_count},
                {"rows", {min_rows, max_rows}},
                {"input_dim", {min_input_dim, max_input_dim}},
                {"hidden_layers", {min_layers, max_layers}},
                {"hidden_width", {min_hidden_width, max_hidden_width}},
                {"output_dim", {min_output_dim, max_output_dim}},
                {"channel_tokens", {min_channels, max_channels}}};
    }

    std::string to_string() const {
        std::ostringstream out;
        out << sample_count << " samples; rows " << min_rows << "-" << max_rows << ", inputs " << min_input_dim
            << "-" << max_input_dim << ", hidden layers " << min_layers << "-" << max_layers << " of width "
            << min_hidden_width << "-" << max_hidden_width << ", outputs " << min_output_dim << "-" << max_output_dim
            << ", channel tokens " << min_channels << "-" << max_channels;
        return out.str();
    }

private:
    static void widen(std::size_t& low, std::size_t& high, std::size_t value) {
        low = std::min(low, value);
        high = std::max(high, value);
    }
};

}  // namespace forward_analysis
