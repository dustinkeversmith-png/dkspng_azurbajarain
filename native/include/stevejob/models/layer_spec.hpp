#pragma once

#include "stevejob/core/helpers.hpp"
#include "stevejob/models/activation.hpp"
#include "stevejob/models/slice_ref.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::models {

struct LayerSpec {
    std::string name;
    std::size_t in_dim = 0;
    std::size_t out_dim = 0;
    ActivationType activation = ActivationType::Linear;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    LayerSpec(std::string layer_name, std::size_t in_d, std::size_t out_d, ActivationType act)
        : name(std::move(layer_name)), in_dim(in_d), out_dim(out_d), activation(act) {
        weights.assign(in_dim, std::vector<double>(out_dim, 0.0));
        biases.assign(out_dim, 0.1);
    }

    void initialize_weights() {
        for (std::size_t row = 0; row < in_dim; ++row) {
            for (std::size_t col = 0; col < out_dim; ++col) {
                weights[row][col] = 0.02 * static_cast<double>(static_cast<int>(row) - static_cast<int>(col));
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        if (input.size() != in_dim) {
            throw std::invalid_argument("Input dimension mismatch in layer " + name);
        }
        std::vector<double> output(out_dim, 0.0);
        for (std::size_t col = 0; col < out_dim; ++col) {
            double sum = biases[col];
            for (std::size_t row = 0; row < in_dim; ++row) {
                stevejob::core::require_finite(input[row]);
                sum += input[row] * weights[row][col];
            }
            stevejob::core::require_finite(sum);
            output[col] = apply_activation(sum, activation);
        }
        return output;
    }

    std::vector<std::vector<double>> extract_block(const SliceRef& slice) const {
        std::vector<std::vector<double>> block(slice.rows, std::vector<double>(slice.cols, 0.0));
        for (std::size_t row = 0; row < slice.rows; ++row) {
            for (std::size_t col = 0; col < slice.cols; ++col) {
                block[row][col] = weights[slice.row_offset + row][slice.col_offset + col];
            }
        }
        return block;
    }

    void patch_block(const SliceRef& slice,
                     const std::vector<std::vector<double>>& patch_w,
                     const std::vector<double>& patch_b) {
        if (patch_w.size() != slice.rows || patch_b.size() != slice.cols) {
            throw std::invalid_argument(name + " patch does not match the slice");
        }
        for (std::size_t row = 0; row < slice.rows; ++row) {
            if (patch_w[row].size() != slice.cols) {
                throw std::invalid_argument(name + " patch row is ragged");
            }
            for (std::size_t col = 0; col < slice.cols; ++col) {
                weights[slice.row_offset + row][slice.col_offset + col] = patch_w[row][col];
            }
        }
        if (slice.row_offset == 0) {
            for (std::size_t col = 0; col < slice.cols; ++col) {
                biases[slice.col_offset + col] = patch_b[col];
            }
        }
    }
};

}  // namespace stevejob::models
