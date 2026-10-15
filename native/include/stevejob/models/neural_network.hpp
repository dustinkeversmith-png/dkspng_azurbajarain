#pragma once

#include "stevejob/models/eval_scheme.hpp"
#include "stevejob/models/layer_spec.hpp"
#include "stevejob/models/slice_ref.hpp"
#include "stevejob/models/slice_scheme.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

class NeuralNetwork {
public:
    std::string name;
    std::string architecture = "mlp";
    std::vector<LayerSpec> layers;
    std::vector<std::vector<double>> outputs;
    SliceScheme slices;
    EvalScheme evaluation;

    NeuralNetwork() = default;
    explicit NeuralNetwork(std::string model_name) : name(std::move(model_name)) {}

    NeuralNetwork& add_layer(const std::string& layer_name,
                           std::size_t in_dim,
                           std::size_t out_dim,
                           const IActivation& activation = Activations::relu()) {
        if (!layers.empty() && layers.back().out_dim != in_dim) {
            throw std::invalid_argument("Layer input dimension does not match previous output dimension.");
        }
        layers.emplace_back(layer_name, in_dim, out_dim, activation);
        return *this;
    }

    static NeuralNetwork from_shapes(const std::vector<std::vector<int>>& shapes) {
        NeuralNetwork model("stack");
        for (std::size_t index = 0; index < shapes.size(); ++index) {
            if (shapes[index].size() != 2 || shapes[index][0] < 1 || shapes[index][1] < 1) {
                throw std::invalid_argument("each layer shape is [rows, cols]");
            }
            const auto rows = static_cast<std::size_t>(shapes[index][0]);
            const auto cols = static_cast<std::size_t>(shapes[index][1]);
            const IActivation& activation =
                index + 1 == shapes.size() ? Activations::linear() : Activations::relu();
            model.add_layer("layer" + std::to_string(index + 1), rows, cols, activation);
        }
        model.initialize_default_weights();
        return model;
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

    std::vector<SliceRef> get_slice_roster() const {
        const std::size_t max_block = slices.block;
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

    std::string stored_layer_name(std::size_t index) const { return stored_name(layers.at(index), index == 0); }

    static NeuralNetwork from_stored_layers(const std::vector<std::string>& names,
                                          const std::vector<std::vector<std::vector<double>>>& weights,
                                          const std::vector<std::vector<double>>& biases,
                                          const std::vector<std::vector<double>>& stored_outputs) {
        NeuralNetwork model("checkpoint");
        model.outputs = stored_outputs;
        if (names.empty()) {
            return model;
        }
        model.architecture = stored_architecture(names.front());
        for (std::size_t index = 0; index < names.size(); ++index) {
            const std::size_t rows = weights[index].size();
            const std::size_t cols = biases[index].size();
            const IActivation& activation = stored_activation(names[index], index + 1 == names.size());
            model.add_layer(visible_name(names[index]), rows, cols, activation);
            model.layers.back().weights = weights[index];
            model.layers.back().biases = biases[index];
        }
        return model;
    }

private:
    bool fits_slice(std::size_t rows, std::size_t cols) const {
        const std::size_t long_side = std::max(rows, cols);
        const std::size_t short_side = std::min(rows, cols);
        return long_side <= slices.long_side && short_side <= slices.short_side;
    }

    std::string stored_name(const LayerSpec& layer, bool first) const {
        std::string stored = layer.name;
        stored.push_back('\x1f');
        stored.push_back(Activations::code(*layer.activation));
        if (first) {
            stored.push_back('\x1f');
            stored.push_back('0');
            stored.push_back('\x1f');
            stored += architecture;
        }
        return stored;
    }

    static std::string visible_name(const std::string& stored) {
        const std::size_t mark = stored.find('\x1f');
        return mark == std::string::npos ? stored : stored.substr(0, mark);
    }

    static const IActivation& stored_activation(const std::string& stored, bool last) {
        const std::size_t mark = stored.find('\x1f');
        if (mark == std::string::npos || mark + 1 >= stored.size()) {
            return last ? Activations::linear() : Activations::relu();
        }
        return Activations::from_code(stored[mark + 1], last);
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

