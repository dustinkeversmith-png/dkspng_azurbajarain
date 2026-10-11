#pragma once

#include "stevejob/core/config.hpp"
#include "stevejob/messaging/compressor.hpp"
#include "stevejob/messaging/totr_engine.hpp"
#include "stevejob/models/layer_spec.hpp"
#include "stevejob/models/loss_function.hpp"
#include "stevejob/models/regression_mlp_model.hpp"
#include "stevejob/models/slice_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

class NeuralModel {
public:
    std::string name;
    std::string architecture = "mlp";
    std::vector<LayerSpec> layers;
    LossType loss_type = LossType::MeanSquaredError;
    std::size_t slice_long = stevejob::core::kSliceBlock;
    std::size_t slice_short = stevejob::core::kSliceBlock;

    NeuralModel() = default;
    explicit NeuralModel(std::string model_name) : name(std::move(model_name)) {}

    NeuralModel& add_layer(const std::string& layer_name,
                           std::size_t in_dim,
                           std::size_t out_dim,
                           ActivationType activation = ActivationType::ReLU) {
        if (!layers.empty() && layers.back().out_dim != in_dim) {
            throw std::invalid_argument("Layer input dimension does not match previous output dimension.");
        }
        layers.emplace_back(layer_name, in_dim, out_dim, activation);
        return *this;
    }

    NeuralModel& set_loss(LossType type) {
        loss_type = type;
        return *this;
    }

    void initialize_default_weights() {
        for (LayerSpec& layer : layers) {
            layer.initialize_weights();
        }
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        std::vector<double> current = input;
        for (const LayerSpec& layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    RegressionScore evaluate_mse(int sample_count = stevejob::core::kEvalSamples) const {
        if (layers.empty()) {
            throw std::runtime_error("Cannot evaluate empty model.");
        }
        if (loss_type != LossType::MeanSquaredError) {
            throw std::invalid_argument("only mean squared error is scored");
        }
        if (sample_count < 1 || sample_count > stevejob::core::kEvalSamples) {
            throw std::invalid_argument("regression evaluation uses 1 to 50 samples");
        }
        constexpr double kPi = 3.141592653589793;
        const std::size_t in_dim = layers.front().in_dim;
        const std::size_t out_dim = layers.back().out_dim;
        const double count = static_cast<double>(sample_count);
        double squared = 0.0;
        double max_residual = 0.0;
        double residual_sum = 0.0;
        std::vector<double> residuals(static_cast<std::size_t>(sample_count));
        for (int index = 0; index < sample_count; ++index) {
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
            const std::vector<double> predicted = forward(input);
            double sample_square = 0.0;
            double sample_peak = 0.0;
            for (std::size_t coord = 0; coord < out_dim; ++coord) {
                double target = 0.0;
                for (std::size_t axis = 0; axis < in_dim; ++axis) {
                    target += (axis % 2 == 0) ? std::sin(input[axis]) : std::cos(input[axis]);
                }
                const double residual = std::abs(predicted[coord] - target);
                sample_square += residual * residual;
                sample_peak = std::max(sample_peak, residual);
            }
            residuals[static_cast<std::size_t>(index)] = sample_peak;
            squared += sample_square / static_cast<double>(out_dim);
            residual_sum += sample_peak;
            max_residual = std::max(max_residual, sample_peak);
        }
        const double mean_residual = residual_sum / count;
        double moment = 0.0;
        for (double residual : residuals) {
            const double delta = residual - mean_residual;
            moment += delta * delta;
        }
        RegressionScore score;
        score.loss = squared / count;
        score.max_residual = max_residual;
        score.std_residual = std::sqrt(moment / count);
        return score;
    }

    std::vector<SliceRef> get_slice_roster(std::size_t max_block = stevejob::core::kSliceBlock) const {
        std::vector<SliceRef> roster;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const LayerSpec& layer = layers[index];
            if (fits_slice(layer.in_dim, layer.out_dim)) {
                roster.push_back(SliceRef{layer.name, layer.name, index, 0, 0, layer.in_dim, layer.out_dim});
                continue;
            }
            for (std::size_t row = 0; row < layer.in_dim; row += max_block) {
                for (std::size_t col = 0; col < layer.out_dim; col += max_block) {
                    const std::size_t row_count = std::min(max_block, layer.in_dim - row);
                    const std::size_t col_count = std::min(max_block, layer.out_dim - col);
                    const std::string slice_id = layer.name + "@r" + std::to_string(row / max_block) + "c" +
                                                 std::to_string(col / max_block);
                    roster.push_back(SliceRef{layer.name, slice_id, index, row, col, row_count, col_count});
                }
            }
        }
        return roster;
    }

    void use_architecture_slices() {
        if (architecture == "bottleneck_ae") {
            slice_long = 8;
            slice_short = 4;
        }
    }

    SliceRef find_slice(const std::string& slice_id) const {
        for (const SliceRef& slice : get_slice_roster()) {
            if (slice.slice_id == slice_id) {
                return slice;
            }
        }
        throw std::invalid_argument("Slice identifier not found: " + slice_id);
    }

    void patch_slice(const std::string& slice_id,
                     const std::vector<std::vector<double>>& patch_w,
                     const std::vector<double>& patch_b) {
        const SliceRef slice = find_slice(slice_id);
        layers[slice.layer_index].patch_block(slice, patch_w, patch_b);
    }

    std::string slice_token(const SliceRef& slice) const {
        const std::string token = stevejob::messaging::TotrEngine::encode_matrix(layers[slice.layer_index].extract_block(slice), 1);
        return display_token(token);
    }

    std::string slice_bias(const SliceRef& slice) const {
        return display_bias(layers[slice.layer_index].biases, slice.col_offset, slice.cols);
    }

    stevejob::messaging::ModelData export_document() const {
        stevejob::messaging::ModelData document;
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const LayerSpec& layer = layers[index];
            stevejob::messaging::LayerRecord record;
            record.name = stored_name(layer, index == 0);
            record.weights = layer.weights;
            record.biases = layer.biases;
            document.layers.push_back(std::move(record));
        }
        return document;
    }

    static NeuralModel import_document(const stevejob::messaging::ModelData& document) {
        NeuralModel model("checkpoint");
        if (document.layers.empty()) {
            return model;
        }
        model.loss_type = stored_loss(document.layers.front().name);
        model.architecture = stored_architecture(document.layers.front().name);
        model.use_architecture_slices();
        for (std::size_t index = 0; index < document.layers.size(); ++index) {
            const stevejob::messaging::LayerRecord& record = document.layers[index];
            const std::size_t rows = record.weights.size();
            const std::size_t cols = record.biases.size();
            const ActivationType activation = stored_activation(record.name, index + 1 == document.layers.size());
            model.add_layer(visible_name(record.name), rows, cols, activation);
            model.layers.back().weights = record.weights;
            model.layers.back().biases = record.biases;
        }
        return model;
    }

private:
    bool fits_slice(std::size_t rows, std::size_t cols) const {
        const std::size_t long_side = std::max(rows, cols);
        const std::size_t short_side = std::min(rows, cols);
        return long_side <= slice_long && short_side <= slice_short;
    }

    static std::string trim_number(double value, int precision) {
        std::ostringstream out;
        out.setf(std::ios::fixed);
        out << std::setprecision(precision) << value;
        std::string text = out.str();
        const std::size_t dot = text.find('.');
        if (dot != std::string::npos) {
            while (!text.empty() && text.back() == '0') {
                text.pop_back();
            }
            if (!text.empty() && text.back() == '.') {
                text.pop_back();
            }
        }
        if (text.empty() || text == "-" || text == "-0") {
            return "0";
        }
        return text;
    }

    static std::string display_token(const std::string& token) {
        const std::size_t start = token.find("g:");
        const std::size_t end = token.find('|', start == std::string::npos ? 0 : start);
        if (start == std::string::npos || end == std::string::npos) {
            return token;
        }
        const double gamma = std::stod(token.substr(start + 2, end - (start + 2)));
        return token.substr(0, start + 2) + trim_number(gamma, 4) + token.substr(end);
    }

    static std::string display_bias(const std::vector<double>& values, std::size_t begin, std::size_t count) {
        std::ostringstream out;
        for (std::size_t index = 0; index < count; ++index) {
            if (index > 0) {
                out << ' ';
            }
            out << trim_number(values[begin + index], 2);
        }
        return out.str();
    }

    std::string stored_name(const LayerSpec& layer, bool first) const {
        std::string stored = layer.name;
        stored.push_back('\x1f');
        stored.push_back(static_cast<char>('0' + static_cast<int>(layer.activation)));
        if (first) {
            stored.push_back('\x1f');
            stored.push_back(static_cast<char>('0' + static_cast<int>(loss_type)));
            stored.push_back('\x1f');
            stored += architecture;
        }
        return stored;
    }

    static std::string visible_name(const std::string& stored) {
        const std::size_t mark = stored.find('\x1f');
        return mark == std::string::npos ? stored : stored.substr(0, mark);
    }

    static ActivationType stored_activation(const std::string& stored, bool last) {
        const std::size_t mark = stored.find('\x1f');
        if (mark == std::string::npos || mark + 1 >= stored.size()) {
            return last ? ActivationType::Linear : ActivationType::ReLU;
        }
        return static_cast<ActivationType>(stored[mark + 1] - '0');
    }

    static LossType stored_loss(const std::string& stored) {
        const std::size_t mark = stored.find('\x1f');
        if (mark == std::string::npos) {
            return LossType::MeanSquaredError;
        }
        const std::size_t loss_mark = stored.find('\x1f', mark + 1);
        if (loss_mark == std::string::npos || loss_mark + 1 >= stored.size()) {
            return LossType::MeanSquaredError;
        }
        return static_cast<LossType>(stored[loss_mark + 1] - '0');
    }

    static std::string stored_architecture(const std::string& stored) {
        const std::size_t mark = stored.find('\x1f');
        if (mark == std::string::npos) {
            return "mlp";
        }
        const std::size_t loss_mark = stored.find('\x1f', mark + 1);
        if (loss_mark == std::string::npos) {
            return "mlp";
        }
        const std::size_t arch_mark = stored.find('\x1f', loss_mark + 1);
        if (arch_mark == std::string::npos || arch_mark + 1 >= stored.size()) {
            return "mlp";
        }
        return stored.substr(arch_mark + 1);
    }
};

}  // namespace stevejob::models
