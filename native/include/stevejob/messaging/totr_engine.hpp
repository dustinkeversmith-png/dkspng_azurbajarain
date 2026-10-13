#pragma once

#include "stevejob/core/helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::messaging {

// Rank-1 int8 tokens for the agent. Stored checkpoints stay SJC1.
class TotrEngine {
public:
    static std::string encode_matrix(const std::vector<std::vector<double>>& matrix, int rank = 1) {
        if (rank != 1) {
            throw std::invalid_argument("TOTR rank must be 1");
        }
        if (matrix.empty() || matrix[0].empty()) {
            throw std::invalid_argument("TOTR matrix is empty");
        }
        const std::size_t rows = matrix.size();
        const std::size_t cols = matrix[0].size();
        for (const std::vector<double>& row : matrix) {
            if (row.size() != cols) {
                throw std::invalid_argument("TOTR matrix must be rectangular");
            }
            for (double value : row) {
                stevejob::core::require_finite(value);
            }
        }

        std::vector<std::vector<double>> probes;
        probes.push_back(std::vector<double>(cols, 1.0));
        for (std::size_t col = 0; col < cols; ++col) {
            std::vector<double> basis(cols, 0.0);
            basis[col] = 1.0;
            probes.push_back(std::move(basis));
        }
        for (std::vector<double>& probe : probes) {
            std::string token;
            if (encode_from_probe(matrix, probe, token)) {
                return token;
            }
        }
        return pack(0.0, std::vector<int>(rows, 0), std::vector<int>(cols, 0));
    }

    static std::vector<std::vector<double>> decode_matrix(const std::string& encoded,
                                                           std::size_t rows,
                                                           std::size_t cols) {
        const double gamma = field_number(encoded, "g:");
        const std::vector<int> left = field_ints(encoded, "A:");
        const std::vector<int> right = field_ints(encoded, "B:");
        if (left.size() != rows || right.size() != cols) {
            throw std::invalid_argument("TOTR matrix has the wrong shape");
        }
        std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
        for (std::size_t row = 0; row < rows; ++row) {
            const double left_value = static_cast<double>(left[row]) * gamma;
            for (std::size_t col = 0; col < cols; ++col) {
                matrix[row][col] = left_value * (static_cast<double>(right[col]) * gamma);
            }
        }
        return matrix;
    }

    static std::string encode_vector(const std::vector<double>& values) {
        std::ostringstream out;
        out.setf(std::ios::fmtflags(0), std::ios::floatfield);
        out.precision(17);
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (index > 0) {
                out << ' ';
            }
            stevejob::core::require_finite(values[index]);
            out << values[index];
        }
        return out.str();
    }

    static std::vector<double> decode_vector(const std::string& text, std::size_t count) {
        std::vector<double> values;
        std::istringstream input(text);
        double value = 0.0;
        while (input >> value) {
            stevejob::core::require_finite(value);
            values.push_back(value);
        }
        if (values.size() != count) {
            throw std::invalid_argument("TOTR vector has the wrong length");
        }
        return values;
    }

private:
    static bool encode_from_probe(const std::vector<std::vector<double>>& matrix,
                                  std::vector<double> right,
                                  std::string& token) {
        const std::size_t rows = matrix.size();
        const std::size_t cols = right.size();
        std::vector<double> left(rows, 0.0);
        if (!normalize(right)) {
            return false;
        }
        for (int iteration = 0; iteration < 10; ++iteration) {
            multiply(matrix, right, left);
            if (!normalize(left)) {
                return false;
            }
            multiply_transpose(matrix, left, right);
            if (!normalize(right)) {
                return false;
            }
        }
        multiply(matrix, right, left);
        const double sigma = norm(left);
        if (sigma == 0.0 || !normalize(left)) {
            return false;
        }
        const double scale = std::sqrt(sigma);
        double peak = 0.0;
        std::vector<double> left_factor(rows);
        std::vector<double> right_factor(cols);
        for (std::size_t row = 0; row < rows; ++row) {
            left_factor[row] = left[row] * scale;
            peak = std::max(peak, std::abs(left_factor[row]));
        }
        for (std::size_t col = 0; col < cols; ++col) {
            right_factor[col] = right[col] * scale;
            peak = std::max(peak, std::abs(right_factor[col]));
        }
        if (peak == 0.0) {
            return false;
        }
        const double gamma = peak / 99.0;
        std::vector<int> left_codes(rows);
        std::vector<int> right_codes(cols);
        for (std::size_t row = 0; row < rows; ++row) {
            left_codes[row] = quantize(left_factor[row] / gamma);
        }
        for (std::size_t col = 0; col < cols; ++col) {
            right_codes[col] = quantize(right_factor[col] / gamma);
        }
        token = pack(gamma, left_codes, right_codes);
        return true;
    }

    static int quantize(double value) {
        const double rounded = std::round(value);
        if (rounded > 99.0) {
            return 99;
        }
        if (rounded < -99.0) {
            return -99;
        }
        return static_cast<int>(rounded);
    }

    static double norm(const std::vector<double>& values) {
        double total = 0.0;
        for (double value : values) {
            total += value * value;
        }
        return std::sqrt(total);
    }

    static bool normalize(std::vector<double>& values) {
        const double length = norm(values);
        if (length == 0.0) {
            return false;
        }
        for (double& value : values) {
            value /= length;
        }
        return true;
    }

    static void multiply(const std::vector<std::vector<double>>& matrix,
                         const std::vector<double>& right,
                         std::vector<double>& left) {
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            double total = 0.0;
            for (std::size_t col = 0; col < right.size(); ++col) {
                total += matrix[row][col] * right[col];
            }
            left[row] = total;
        }
    }

    static void multiply_transpose(const std::vector<std::vector<double>>& matrix,
                                   const std::vector<double>& left,
                                   std::vector<double>& right) {
        std::fill(right.begin(), right.end(), 0.0);
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            for (std::size_t col = 0; col < right.size(); ++col) {
                right[col] += matrix[row][col] * left[row];
            }
        }
    }

    static std::string pack(double gamma, const std::vector<int>& left, const std::vector<int>& right) {
        std::ostringstream out;
        out.setf(std::ios::fmtflags(0), std::ios::floatfield);
        out.precision(17);
        out << "r:1|g:" << gamma << "|A:";
        for (std::size_t index = 0; index < left.size(); ++index) {
            if (index > 0) {
                out << ' ';
            }
            out << left[index];
        }
        out << "|B:";
        for (std::size_t index = 0; index < right.size(); ++index) {
            if (index > 0) {
                out << ' ';
            }
            out << right[index];
        }
        return out.str();
    }

    static std::string field(const std::string& encoded, const char* key) {
        const std::string token = std::string("|") + key;
        std::size_t start = encoded.find(token);
        if (start == std::string::npos) {
            if (encoded.rfind(key, 0) == 0) {
                start = 0;
            } else {
                throw std::invalid_argument("TOTR token is missing a field");
            }
        } else {
            start += 1;
        }
        start += std::char_traits<char>::length(key);
        const std::size_t end = encoded.find('|', start);
        return encoded.substr(start, end == std::string::npos ? std::string::npos : end - start);
    }

    static double field_number(const std::string& encoded, const char* key) {
        return std::stod(field(encoded, key));
    }

public:
    static int clamp_int(int code, int bound = 99) {
        if (code < -bound) {
            return -bound;
        }
        if (code > bound) {
            return bound;
        }
        return code;
    }

    static int integer_distance(const std::string& candidate, const std::string& active) {
        int distance = 0;
        for (const char* key : {"A:", "B:"}) {
            const std::vector<int> left = field_ints(candidate, key);
            const std::vector<int> right = field_ints(active, key);
            if (left.empty() || right.empty() || left.size() != right.size()) {
                const std::size_t span = std::max(left.size(), right.size());
                return static_cast<int>(std::max<std::size_t>(span, 2));
            }
            for (std::size_t index = 0; index < left.size(); ++index) {
                if (left[index] != right[index]) {
                    ++distance;
                }
            }
        }
        return distance;
    }

    static std::vector<int> field_ints(const std::string& encoded, const char* key) {
        std::string text = field(encoded, key);
        for (char& character : text) {
            if (character == ',') {
                character = ' ';
            }
        }
        std::istringstream input(text);
        std::vector<int> values;
        int value = 0;
        while (input >> value) {
            values.push_back(clamp_int(value));
        }
        if (values.empty()) {
            throw std::invalid_argument("TOTR token is missing a factor");
        }
        return values;
    }
};

}  // namespace stevejob::messaging
