#pragma once

#include "stevejob/models/neural_model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace stevejob::models {

struct ModelDirective {
    std::string task;
    std::string model_type;
    std::string rules;
};

class ModelTemplate {
public:
    virtual ~ModelTemplate() = default;

    virtual NeuralModel& network() = 0;
    virtual const NeuralModel& network() const = 0;
    virtual std::vector<double> forward(const std::vector<double>& input) const = 0;
    virtual RegressionScore evaluate(int sample_count = 50) const = 0;
    virtual ModelDirective get_directives() const = 0;
    virtual RegressionScore score_network(const NeuralModel& model, int sample_count = 50) const = 0;

    std::vector<SliceRef> get_slice_roster(std::size_t max_block = stevejob::core::kSliceBlock) const {
        return network().get_slice_roster(max_block);
    }

    std::map<std::string, double> derive_training_telemetry() const {
        double count = 0.0;
        double sum = 0.0;
        double square = 0.0;
        double peak = 0.0;
        double sparse = 0.0;
        for (const LayerSpec& layer : network().layers) {
            for (const std::vector<double>& row : layer.weights) {
                for (double value : row) {
                    const double magnitude = std::abs(value);
                    sum += magnitude;
                    square += value * value;
                    peak = std::max(peak, magnitude);
                    if (magnitude < 0.05) {
                        sparse += 1.0;
                    }
                    count += 1.0;
                }
            }
        }
        const double mean = count == 0.0 ? 0.0 : sum / count;
        std::map<std::string, double> telemetry;
        telemetry["sparsity"] = count == 0.0 ? 0.0 : sparse / count;
        telemetry["variance"] = count == 0.0 ? 0.0 : square / count;
        telemetry["condition"] = peak / std::max(mean, 1e-6);
        return telemetry;
    }

    std::string compute_slice_gradient(const std::string& slice_id) const {
        const SliceRef slice = network().find_slice(slice_id);
        const std::string token = network().slice_token(slice);
        const std::vector<int> left = codes_of(token, "A:");
        const std::vector<int> right = codes_of(token, "B:");
        const double baseline = evaluate().loss;
        std::ostringstream hint;
        hint << "A:[";
        write_signs(hint, token, "A:", left, slice, baseline);
        hint << "] B:[";
        write_signs(hint, token, "B:", right, slice, baseline);
        hint << "]";
        return hint.str();
    }

protected:
    static std::vector<double> sample_input(int index, int sample_count, std::size_t in_dim) {
        constexpr double kPi = 3.141592653589793;
        const double count = static_cast<double>(sample_count);
        std::vector<double> input(in_dim, 0.0);
        for (std::size_t axis = 0; axis < in_dim; ++axis) {
            if (axis == 0) {
                input[axis] = (static_cast<double>(index) / count) * kPi;
            } else if (axis == 1) {
                input[axis] = (static_cast<double>(sample_count - index) / count) * kPi;
            } else {
                const int phase = (index + static_cast<int>(axis) * 3) % sample_count;
                input[axis] = (static_cast<double>(phase) / count) * kPi;
            }
        }
        return input;
    }

private:
    static std::vector<int> codes_of(const std::string& token, const char* key) {
        const std::size_t start = token.find(key);
        const std::size_t from = start + std::char_traits<char>::length(key);
        const std::size_t end = token.find('|', from);
        std::istringstream input(token.substr(from, end == std::string::npos ? std::string::npos : end - from));
        std::vector<int> codes;
        int value = 0;
        while (input >> value) {
            codes.push_back(value);
        }
        return codes;
    }

    static std::string with_code(const std::string& token, const char* key, std::size_t index, int value) {
        const std::vector<int> codes = codes_of(token, key);
        std::ostringstream replaced;
        for (std::size_t cursor = 0; cursor < codes.size(); ++cursor) {
            if (cursor > 0) {
                replaced << ' ';
            }
            replaced << (cursor == index ? value : codes[cursor]);
        }
        const std::size_t start = token.find(key) + std::char_traits<char>::length(key);
        const std::size_t end = token.find('|', start);
        return token.substr(0, start) + replaced.str() + (end == std::string::npos ? "" : token.substr(end));
    }

    void write_signs(std::ostringstream& hint,
                     const std::string& token,
                     const char* key,
                     const std::vector<int>& codes,
                     const SliceRef& slice,
                     double baseline) const {
        for (std::size_t index = 0; index < codes.size(); ++index) {
            if (index > 0) {
                hint << ", ";
            }
            const int step = codes[index] >= 99 ? -1 : 1;
            const int probed = std::max(-99, std::min(99, codes[index] + step));
            NeuralModel trial = network();
            const std::string patched = with_code(token, key, index, probed);
            trial.patch_slice(slice.slice_id,
                              stevejob::messaging::TotrEngine::decode_matrix(patched, slice.rows, slice.cols),
                              stevejob::messaging::TotrEngine::decode_vector(network().slice_bias(slice), slice.cols));
            const double probed_loss = score_network(trial).loss;
            const double delta = probed_loss - baseline;
            const double signed_delta = step > 0 ? delta : -delta;
            const double magnitude = std::abs(signed_delta);
            const double steep = std::max(0.02, 0.01 * std::abs(baseline));
            if (magnitude < 1e-8) {
                hint << '0';
            } else if (signed_delta > 0.0) {
                hint << (magnitude >= steep ? "+2" : "+1");
            } else {
                hint << (magnitude >= steep ? "-2" : "-1");
            }
        }
    }
};

}  // namespace stevejob::models
