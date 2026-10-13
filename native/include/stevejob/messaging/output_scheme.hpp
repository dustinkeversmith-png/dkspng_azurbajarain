#pragma once

#include "stevejob/messaging/scheme_descriptor.hpp"
#include "stevejob/messaging/totr_engine.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::messaging {

class IOutputScheme {
public:
    virtual ~IOutputScheme() = default;
    virtual EncodingDescriptor get_descriptor() const = 0;

    virtual std::string get_output_description() const {
        const EncodingDescriptor info = get_descriptor();
        return info.scheme_name + ". Format: " + info.format_description + ". Equation: " + info.equation +
               ". Rules: " + info.actionable_rules;
    }

    virtual std::string serialize_output(const std::vector<std::vector<double>>& outputs) const = 0;
    virtual std::vector<std::vector<double>> deserialize_output(const std::string& token, std::size_t rows, std::size_t cols) const = 0;

protected:
    static std::string pack_rows(const std::vector<std::vector<double>>& outputs) {
        std::ostringstream out;
        out.setf(std::ios::fixed);
        out.precision(4);
        for (std::size_t row = 0; row < outputs.size(); ++row) {
            if (row > 0) {
                out << ';';
            }
            for (std::size_t col = 0; col < outputs[row].size(); ++col) {
                if (col > 0) {
                    out << ' ';
                }
                out << outputs[row][col];
            }
        }
        return out.str();
    }

    static std::vector<std::vector<double>> unpack_rows(const std::string& token, std::size_t rows, std::size_t cols) {
        std::vector<std::vector<double>> outputs;
        std::string pending = token;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t split = pending.find(';');
            const std::string piece = pending.substr(0, split);
            outputs.push_back(TotrEngine::decode_vector(piece, cols));
            if (split == std::string::npos) {
                pending.clear();
            } else {
                pending = pending.substr(split + 1);
            }
        }
        if (outputs.size() != rows) {
            throw std::invalid_argument("output token has the wrong row count");
        }
        return outputs;
    }
};

}  // namespace stevejob::messaging
