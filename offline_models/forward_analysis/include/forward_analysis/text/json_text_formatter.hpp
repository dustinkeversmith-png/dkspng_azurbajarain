#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

struct JsonTextFormatter {
    static std::string escape_json(const std::string& input) {
        std::ostringstream out;
        for (char c : input) {
            switch (c) {
                case '"': out << "\\\""; break;
                case '\\': out << "\\\\"; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out << c; break;
            }
        }
        return out.str();
    }

    static std::string format_double(double value, int precision = 2) {
        std::ostringstream out;
        out.setf(std::ios::fixed);
        out << std::setprecision(precision) << value;
        return out.str();
    }

    static std::string format_double_array(const Eigen::VectorXd& values, int precision = 2) {
        std::ostringstream out;
        out << "[";
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            out << (i > 0 ? ", " : "") << format_double(values(i), precision);
        }
        out << "]";
        return out.str();
    }

    static std::string format_index_array(const std::vector<std::size_t>& values) {
        return "[" + join_indices(values, ", ") + "]";
    }

    static std::string join_indices(const std::vector<std::size_t>& values, const std::string& separator) {
        std::ostringstream out;
        for (std::size_t i = 0; i < values.size(); ++i) {
            out << (i > 0 ? separator : "") << values[i];
        }
        return out.str();
    }

    static std::string single_string_field(const std::string& key, const std::string& value) {
        return "{\"" + escape_json(key) + "\": \"" + escape_json(value) + "\"}";
    }
};

}  // namespace forward_analysis
