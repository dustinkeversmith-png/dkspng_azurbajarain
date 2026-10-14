#pragma once

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace forward_analysis {

struct JsonTensorReader {
    static Eigen::MatrixXd matrix(const nlohmann::json& node, const std::string& name) {
        if (!node.is_array() || node.empty() || !node.front().is_array() || node.front().empty()) {
            throw std::invalid_argument(name + " must be a non-empty array of non-empty rows");
        }
        const Eigen::Index rows = static_cast<Eigen::Index>(node.size());
        const Eigen::Index cols = static_cast<Eigen::Index>(node.front().size());
        Eigen::MatrixXd out(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            const nlohmann::json& row = node[static_cast<std::size_t>(r)];
            if (!row.is_array() || static_cast<Eigen::Index>(row.size()) != cols) {
                throw std::invalid_argument(name + " row " + std::to_string(r) + " must have " +
                                            std::to_string(cols) + " values");
            }
            for (Eigen::Index c = 0; c < cols; ++c) {
                out(r, c) = number(row[static_cast<std::size_t>(c)], name);
            }
        }
        return out;
    }

    static Eigen::VectorXd vector(const nlohmann::json& node, const std::string& name) {
        if (!node.is_array() || node.empty()) {
            throw std::invalid_argument(name + " must be a non-empty array");
        }
        Eigen::VectorXd out(static_cast<Eigen::Index>(node.size()));
        for (Eigen::Index i = 0; i < out.size(); ++i) {
            out(i) = number(node[static_cast<std::size_t>(i)], name);
        }
        return out;
    }

    static std::vector<Eigen::MatrixXd> matrix_list(const nlohmann::json& node, const std::string& name) {
        if (!node.is_array() || node.empty()) {
            throw std::invalid_argument(name + " must be a non-empty array of matrices");
        }
        std::vector<Eigen::MatrixXd> out;
        out.reserve(node.size());
        for (std::size_t l = 0; l < node.size(); ++l) {
            out.push_back(matrix(node[l], name + "[" + std::to_string(l) + "]"));
        }
        return out;
    }

private:
    static double number(const nlohmann::json& value, const std::string& name) {
        if (!value.is_number()) {
            throw std::invalid_argument(name + " contains a non-numeric value");
        }
        return value.get<double>();
    }
};

}  // namespace forward_analysis
