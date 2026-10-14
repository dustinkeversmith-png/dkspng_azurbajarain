#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

struct TelemetryShape {
    // Number of samples (rows) in the batch.
    std::size_t num_samples = 0;
    // Number of input coordinates fed to the observed network.
    std::size_t input_dim = 0;
    // Width of each hidden layer, in forward order.
    std::vector<std::size_t> hidden_widths;
    // Number of network outputs (and targets).
    std::size_t output_dim = 0;

    std::size_t total_hidden_units() const noexcept {
        std::size_t total = 0;
        for (std::size_t w : hidden_widths) {
            total += w;
        }
        return total;
    }

    // One channel per input coordinate, hidden unit, and per output its target, prediction and residual.
    std::size_t channel_count() const noexcept {
        return input_dim + total_hidden_units() + 3 * output_dim;
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "N=" << num_samples << " in=" << input_dim << " hidden=[";
        for (std::size_t l = 0; l < hidden_widths.size(); ++l) {
            out << (l > 0 ? "," : "") << hidden_widths[l];
        }
        out << "] out=" << output_dim;
        return out.str();
    }
};

}  // namespace forward_analysis
