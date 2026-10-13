#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "stevejob/core/hash.hpp"
#include "stevejob/core/helpers.hpp"
#include "stevejob/messaging/container_scheme.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace stevejob::messaging {

class SJC1Compressor : public IContainerScheme {
public:
    static const SJC1Compressor& instance() {
        static const SJC1Compressor scheme;
        return scheme;
    }

    EncodingDescriptor get_descriptor() const override {
        return EncodingDescriptor{
            "sjc1",
            "SJC1 version 1. Magic, flags, body size, checksum, then an optional RLE payload.",
            "Each weight, bias, and output cell is an IEEE-754 little-endian float64.",
            "The first stored layer name carries activation, loss, and architecture. Later names carry activation."};
    }

    std::string pack(const stevejob::models::NeuralNetwork& model) const override {
        stevejob::core::require_little_endian("SJC1 packing requires a little-endian host");
        const std::string body = pack_body(model);
        if (body.size() > 0xffffffffu) {
            throw std::invalid_argument("SJC1 body exceeds 4 GiB");
        }
        const std::string encoded = rle_encode(body);
        const bool compressed = encoded.size() < body.size();
        const std::string& payload = compressed ? encoded : body;
        const std::uint64_t checksum = stevejob::core::fnv1a64_bytes(
            reinterpret_cast<const unsigned char*>(body.data()), body.size());

        std::string out;
        out.reserve(18 + payload.size());
        out.append("SJC1", 4);
        out.push_back(static_cast<char>(1));
        out.push_back(static_cast<char>(compressed ? 1 : 0));
        stevejob::core::append_u32(out, static_cast<std::uint32_t>(body.size()));
        stevejob::core::append_u64(out, checksum);
        out.append(payload);
        return out;
    }

    stevejob::models::NeuralNetwork unpack(const std::string& bytes) const override {
        const Opened opened = open(bytes);
        return to_model(opened.wire);
    }

    static py::dict document(const std::string& bytes) {
        const Opened opened = open(bytes);
        py::dict layers;
        for (const WireLayer& layer : opened.wire.layers) {
            py::dict entry;
            entry["weights"] = layer.weights;
            entry["biases"] = layer.biases;
            layers[py::str(layer.name)] = entry;
        }
        py::dict result;
        result["layers"] = layers;
        result["outputs"] = opened.wire.outputs;
        result["compressed"] = opened.compressed;
        result["container_checksum"] = stevejob::core::to_hex(opened.checksum);
        return result;
    }

private:
    struct WireLayer {
        std::string name;
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
    };

    struct WireModel {
        std::vector<WireLayer> layers;
        std::vector<std::vector<double>> outputs;
    };

    struct Opened {
        WireModel wire;
        bool compressed = false;
        std::uint64_t checksum = 0;
    };

    static stevejob::models::NeuralNetwork to_model(const WireModel& wire) {
        std::vector<std::string> names;
        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>> biases;
        names.reserve(wire.layers.size());
        for (const WireLayer& layer : wire.layers) {
            names.push_back(layer.name);
            weights.push_back(layer.weights);
            biases.push_back(layer.biases);
        }
        return stevejob::models::NeuralNetwork::from_stored_layers(names, weights, biases, wire.outputs);
    }

    static Opened open(const std::string& bytes) {
        stevejob::core::require_little_endian("SJC1 packing requires a little-endian host");
        if (bytes.size() < 18 || bytes.compare(0, 4, "SJC1") != 0) {
            throw std::invalid_argument("payload is not an SJC1 checkpoint");
        }
        const auto version = static_cast<unsigned char>(bytes[4]);
        const auto flags = static_cast<unsigned char>(bytes[5]);
        if (version != 1) {
            throw std::invalid_argument("unsupported SJC1 version");
        }
        if ((flags & ~std::uint8_t{1}) != 0) {
            throw std::invalid_argument("unsupported SJC1 flags");
        }
        std::size_t cursor = 6;
        const std::uint32_t body_size = stevejob::core::read_u32(bytes, cursor);
        const std::uint64_t checksum = stevejob::core::read_u64(bytes, cursor);
        const std::string payload = bytes.substr(cursor);
        const bool compressed = (flags & 1) != 0;
        const std::string body = compressed ? rle_decode(payload, body_size) : payload;
        if (!compressed && body.size() != body_size) {
            throw std::invalid_argument("SJC1 body size does not match the header");
        }
        const std::uint64_t actual = stevejob::core::fnv1a64_bytes(
            reinterpret_cast<const unsigned char*>(body.data()), body.size());
        if (actual != checksum) {
            throw std::invalid_argument("SJC1 checksum does not match");
        }
        Opened opened;
        opened.wire = unpack_body(body);
        opened.compressed = compressed;
        opened.checksum = checksum;
        return opened;
    }

    static bool same_block(const std::string& raw, std::size_t left, std::size_t right) {
        for (std::size_t offset = 0; offset < 8; ++offset) {
            if (raw[left + offset] != raw[right + offset]) {
                return false;
            }
        }
        return true;
    }

    static std::string rle_encode(const std::string& raw) {
        std::string out;
        std::size_t index = 0;
        const std::size_t size = raw.size();
        while (index < size) {
            if (index + 16 <= size && same_block(raw, index, index + 8)) {
                std::size_t blocks = 2;
                while (index + (blocks + 1) * 8 <= size &&
                       same_block(raw, index, index + blocks * 8) && blocks < 255) {
                    ++blocks;
                }
                out.push_back(static_cast<char>(2));
                out.push_back(static_cast<char>(blocks));
                out.append(raw, index, 8);
                index += blocks * 8;
                continue;
            }
            const char value = raw[index];
            std::size_t run = 1;
            while (index + run < size && raw[index + run] == value && run < 255) {
                ++run;
            }
            if (run >= 3) {
                out.push_back(static_cast<char>(1));
                out.push_back(static_cast<char>(run));
                out.push_back(value);
                index += run;
                continue;
            }
            const std::size_t literal_begin = index;
            index += run;
            while (index < size && index - literal_begin < 256) {
                if (index + 16 <= size && same_block(raw, index, index + 8)) {
                    break;
                }
                std::size_t ahead = 1;
                while (index + ahead < size && raw[index + ahead] == raw[index] && ahead < 255) {
                    ++ahead;
                }
                if (ahead >= 3) {
                    break;
                }
                index += ahead;
            }
            const std::size_t literal_len = index - literal_begin;
            out.push_back(static_cast<char>(0));
            out.push_back(static_cast<char>(static_cast<unsigned char>(literal_len - 1)));
            out.append(raw, literal_begin, literal_len);
        }
        return out;
    }

    static std::string rle_decode(const std::string& packed, std::size_t expected) {
        std::string out;
        out.reserve(expected);
        std::size_t index = 0;
        while (index < packed.size()) {
            const auto op = static_cast<unsigned char>(packed[index++]);
            if (op == 0) {
                if (index >= packed.size()) {
                    throw std::invalid_argument("SJC1 compressed body ended early");
                }
                const std::size_t literal_len =
                    static_cast<std::size_t>(static_cast<unsigned char>(packed[index++])) + 1;
                if (index + literal_len > packed.size()) {
                    throw std::invalid_argument("SJC1 compressed body ended early");
                }
                out.append(packed, index, literal_len);
                index += literal_len;
            } else if (op == 1) {
                if (index + 2 > packed.size()) {
                    throw std::invalid_argument("SJC1 compressed body ended early");
                }
                const std::size_t run = static_cast<unsigned char>(packed[index++]);
                const char value = packed[index++];
                if (run < 3) {
                    throw std::invalid_argument("SJC1 compressed body is invalid");
                }
                out.append(run, value);
            } else if (op == 2) {
                if (index + 9 > packed.size()) {
                    throw std::invalid_argument("SJC1 compressed body ended early");
                }
                const std::size_t blocks = static_cast<unsigned char>(packed[index++]);
                if (blocks < 2) {
                    throw std::invalid_argument("SJC1 compressed body is invalid");
                }
                const std::string block = packed.substr(index, 8);
                index += 8;
                for (std::size_t copy = 0; copy < blocks; ++copy) {
                    out.append(block);
                }
            } else {
                throw std::invalid_argument("SJC1 compressed body is invalid");
            }
        }
        if (out.size() != expected) {
            throw std::invalid_argument("SJC1 body size does not match the header");
        }
        return out;
    }

    static std::string pack_body(const stevejob::models::NeuralNetwork& model) {
        if (model.layers.size() > 32) {
            throw std::invalid_argument("too many layers to pack");
        }
        std::string body;
        stevejob::core::append_u16(body, static_cast<std::uint16_t>(model.layers.size()));
        for (std::size_t index = 0; index < model.layers.size(); ++index) {
            const stevejob::models::LayerSpec& layer = model.layers[index];
            const std::string name = model.stored_layer_name(index);
            if (name.empty() || name.size() > 64) {
                throw std::invalid_argument("layer name must be 1 to 64 bytes");
            }
            if (layer.weights.size() > 65535) {
                throw std::invalid_argument("too many weight rows");
            }
            body.push_back(static_cast<char>(name.size()));
            body.append(name);
            const auto rows = static_cast<std::uint16_t>(layer.weights.size());
            std::uint16_t cols = 0;
            if (!layer.weights.empty()) {
                if (layer.weights[0].size() > 65535) {
                    throw std::invalid_argument("too many weight columns");
                }
                cols = static_cast<std::uint16_t>(layer.weights[0].size());
            }
            for (const std::vector<double>& row : layer.weights) {
                if (row.size() != cols) {
                    throw std::invalid_argument("weight rows must be rectangular");
                }
            }
            if (layer.biases.size() > 65535) {
                throw std::invalid_argument("too many biases");
            }
            stevejob::core::append_u16(body, rows);
            stevejob::core::append_u16(body, cols);
            stevejob::core::append_u16(body, static_cast<std::uint16_t>(layer.biases.size()));
        }
        if (model.outputs.size() > 65535) {
            throw std::invalid_argument("too many output rows");
        }
        std::uint16_t output_cols = 0;
        if (!model.outputs.empty()) {
            if (model.outputs[0].size() > 65535) {
                throw std::invalid_argument("too many output columns");
            }
            output_cols = static_cast<std::uint16_t>(model.outputs[0].size());
        }
        for (const std::vector<double>& row : model.outputs) {
            if (row.size() != output_cols) {
                throw std::invalid_argument("output rows must be rectangular");
            }
        }
        stevejob::core::append_u16(body, static_cast<std::uint16_t>(model.outputs.size()));
        stevejob::core::append_u16(body, output_cols);
        for (const stevejob::models::LayerSpec& layer : model.layers) {
            for (const std::vector<double>& row : layer.weights) {
                for (double value : row) {
                    stevejob::core::append_f64(body, value);
                }
            }
            for (double value : layer.biases) {
                stevejob::core::append_f64(body, value);
            }
        }
        for (const std::vector<double>& row : model.outputs) {
            for (double value : row) {
                stevejob::core::append_f64(body, value);
            }
        }
        return body;
    }

    static WireModel unpack_body(const std::string& body) {
        std::size_t cursor = 0;
        const std::uint16_t layer_count = stevejob::core::read_u16(body, cursor);
        struct Layout {
            std::string name;
            std::uint16_t rows;
            std::uint16_t cols;
            std::uint16_t biases;
        };
        std::vector<Layout> layouts;
        layouts.reserve(layer_count);
        for (std::uint16_t layer = 0; layer < layer_count; ++layer) {
            if (cursor >= body.size()) {
                throw std::invalid_argument("SJC1 body ended early");
            }
            const auto name_len = static_cast<unsigned char>(body[cursor++]);
            if (name_len == 0 || cursor + name_len > body.size()) {
                throw std::invalid_argument("SJC1 layer name is invalid");
            }
            Layout layout;
            layout.name = body.substr(cursor, name_len);
            cursor += name_len;
            layout.rows = stevejob::core::read_u16(body, cursor);
            layout.cols = stevejob::core::read_u16(body, cursor);
            layout.biases = stevejob::core::read_u16(body, cursor);
            layouts.push_back(layout);
        }
        const std::uint16_t output_rows = stevejob::core::read_u16(body, cursor);
        const std::uint16_t output_cols = stevejob::core::read_u16(body, cursor);

        WireModel model;
        model.layers.reserve(layouts.size());
        for (const Layout& layout : layouts) {
            WireLayer record;
            record.name = layout.name;
            record.weights.resize(layout.rows);
            for (std::uint16_t row = 0; row < layout.rows; ++row) {
                record.weights[row].resize(layout.cols);
                for (std::uint16_t col = 0; col < layout.cols; ++col) {
                    record.weights[row][col] = stevejob::core::read_f64(body, cursor);
                }
            }
            record.biases.resize(layout.biases);
            for (std::uint16_t index = 0; index < layout.biases; ++index) {
                record.biases[index] = stevejob::core::read_f64(body, cursor);
            }
            model.layers.push_back(std::move(record));
        }
        model.outputs.resize(output_rows);
        for (std::uint16_t row = 0; row < output_rows; ++row) {
            model.outputs[row].resize(output_cols);
            for (std::uint16_t col = 0; col < output_cols; ++col) {
                model.outputs[row][col] = stevejob::core::read_f64(body, cursor);
            }
        }
        if (cursor != body.size()) {
            throw std::invalid_argument("SJC1 body has trailing bytes");
        }
        return model;
    }
};

}  // namespace stevejob::messaging
