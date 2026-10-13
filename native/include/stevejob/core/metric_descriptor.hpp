#pragma once
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

namespace stevejob::core {

enum class FieldType { Scalar, Text, Token, Vector };

struct DescribedField {
    std::string key;
    FieldType type = FieldType::Scalar;
    std::variant<double, std::string, std::vector<double>> value;
    std::string tag;                // E.g., "telemetry", "eval_metric", "gradient_probe"
    std::string format_description; // Explains syntax/format
    std::string actionable_rule;    // Explains what the LLM should do with this value

    std::string to_string() const {
        if (std::holds_alternative<double>(value)) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.4f", std::get<double>(value));
            return buf;
        }
        if (std::holds_alternative<std::string>(value)) {
            return std::get<std::string>(value);
        }
        return "[vector]";
    }
};

} // namespace stevejob::core
