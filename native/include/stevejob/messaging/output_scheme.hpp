#pragma once

#include "stevejob/messaging/scheme_descriptor.hpp"

#include <cstddef>
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
};

}  // namespace stevejob::messaging

