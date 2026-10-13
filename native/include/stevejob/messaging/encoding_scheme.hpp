#pragma once

#include "stevejob/messaging/scheme_descriptor.hpp"
#include "stevejob/messaging/totr_engine.hpp"
#include "stevejob/storage/json_text.hpp"

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stevejob::messaging {

struct AcceptedValue {
    std::string token;
    std::vector<std::pair<std::string, std::string>> notes;
};

class IEncodingScheme {
public:
    virtual ~IEncodingScheme() = default;
    virtual EncodingDescriptor get_descriptor() const = 0;

    virtual AcceptedValue accept(const storage::JsonText& value, std::size_t rows, std::size_t cols) const {
        (void)rows;
        (void)cols;
        if (value.kind == storage::JsonText::Kind::String && !value.text.empty()) {
            return AcceptedValue{value.text, {}};
        }
        throw std::invalid_argument(get_descriptor().scheme_name + " completion value is not a token");
    }

    virtual std::string constraint_line(int bound) const {
        (void)bound;
        return "";
    }

    virtual int token_distance(const std::string& candidate, const std::string& active) const {
        (void)candidate;
        (void)active;
        throw std::invalid_argument(get_descriptor().scheme_name + " has no integer distance");
    }

    virtual std::string step_label(const std::string& token, const std::string& active) const {
        (void)token;
        (void)active;
        return "";
    }

    virtual bool raised(const std::string& updated, const std::string& active, std::size_t count) const {
        (void)updated;
        (void)active;
        (void)count;
        return false;
    }

    virtual std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const = 0;
    virtual std::vector<std::vector<double>> deserialize_matrix(const std::string& token, std::size_t rows, std::size_t cols) const = 0;
    virtual std::string serialize_vector(const std::vector<double>& vec, std::size_t start, std::size_t count) const = 0;
    virtual std::vector<double> deserialize_vector(const std::string& text, std::size_t count) const = 0;
};

class TotrRank1Scheme : public IEncodingScheme {
public:
    static const TotrRank1Scheme& instance() {
        static const TotrRank1Scheme scheme;
        return scheme;
    }

    EncodingDescriptor get_descriptor() const override {
        return EncodingDescriptor{
            "totr_rank1",
            "r:1|g:<gamma>|A:<row ints>|B:<col ints>",
            "W[row][col] = (A[row] * gamma) * (B[col] * gamma).",
            "Codes are clamped to [-99, 99] before scoring."};
    }

    std::string constraint_line(int bound) const override {
        const int edge = bound == 0 ? 25 : bound;
        return "Bound: Ints must stay in [-" + std::to_string(edge) + ", " + std::to_string(edge) +
               "]. Never write < -99 or > 99.\n";
    }

    std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const override {
        return TotrEngine::encode_matrix(matrix, 1);
    }

    std::vector<std::vector<double>> deserialize_matrix(const std::string& token, std::size_t rows, std::size_t cols) const override {
        return TotrEngine::decode_matrix(token, rows, cols);
    }

    std::string serialize_vector(const std::vector<double>& vec, std::size_t start, std::size_t count) const override {
        return TotrEngine::encode_vector(window(vec, start, count));
    }

    std::vector<double> deserialize_vector(const std::string& text, std::size_t count) const override {
        return TotrEngine::decode_vector(text, count);
    }

    AcceptedValue accept(const storage::JsonText& value, std::size_t rows, std::size_t cols) const override {
        if (value.kind == storage::JsonText::Kind::String) {
            const std::string token = value.text;
            if (word_count(field_body(token, "A:")) != rows || word_count(field_body(token, "B:")) != cols) {
                throw std::invalid_argument("totr_rank1 token has the wrong factor counts");
            }
            return AcceptedValue{token, {{"gamma", field_body(token, "g:")}}};
        }
        if (value.kind != storage::JsonText::Kind::Object) {
            throw std::invalid_argument("totr_rank1 completion value is not a token");
        }
        const storage::JsonText* left = find_any(value, {"A", "a"});
        const storage::JsonText* right = find_any(value, {"B", "b_vec"});
        const storage::JsonText* gamma_value = find_any(value, {"g", "gamma"});
        std::vector<long long> left_codes;
        std::vector<long long> right_codes;
        std::vector<double> gamma_numbers;
        if (left == nullptr || right == nullptr || gamma_value == nullptr || !as_ints(*left, left_codes) ||
            !as_ints(*right, right_codes) || !as_numbers(*gamma_value, gamma_numbers) || gamma_numbers.size() != 1 ||
            left_codes.size() != rows || right_codes.size() != cols) {
            throw std::invalid_argument("totr_rank1 factors have the wrong shape");
        }
        const double gamma = gamma_numbers[0];
        if (!(gamma > 0.0) || gamma > 5.0) {
            throw std::invalid_argument("totr_rank1 gamma is outside the readable range");
        }
        const std::string token = pack_rank1(gamma, left_codes, right_codes);
        return AcceptedValue{token, {{"gamma", field_body(token, "g:")}}};
    }

    int token_distance(const std::string& candidate, const std::string& active) const override {
        return TotrEngine::integer_distance(candidate, active);
    }

    std::string step_label(const std::string& token, const std::string& active) const override {
        if (token.empty()) {
            return "";
        }
        const std::size_t pos = token.find("g:");
        std::string gamma = "?";
        if (pos != std::string::npos) {
            const std::size_t begin = pos + 2;
            const std::size_t end = token.find('|', begin);
            gamma = token.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        }
        int distance = 0;
        try {
            distance = token_distance(token, active);
        } catch (const std::invalid_argument&) {
            distance = 0;
        }
        return "g=" + gamma + " d=" + std::to_string(distance);
    }

private:
    static const storage::JsonText* find_any(const storage::JsonText& node, std::initializer_list<const char*> keys) {
        if (node.kind != storage::JsonText::Kind::Object) {
            return nullptr;
        }
        for (const char* key : keys) {
            for (const std::pair<std::string, storage::JsonText>& entry : node.object) {
                if (entry.first == key) {
                    return &entry.second;
                }
            }
        }
        return nullptr;
    }

    static bool as_numbers(const storage::JsonText& value, std::vector<double>& out) {
        out.clear();
        if (value.kind == storage::JsonText::Kind::Number) {
            out.push_back(value.number);
            return true;
        }
        if (value.kind != storage::JsonText::Kind::Array || value.array.empty()) {
            return false;
        }
        for (const storage::JsonText& item : value.array) {
            if (item.kind != storage::JsonText::Kind::Number) {
                return false;
            }
            out.push_back(item.number);
        }
        return true;
    }

    static bool as_ints(const storage::JsonText& value, std::vector<long long>& out) {
        std::vector<double> numbers;
        if (!as_numbers(value, numbers)) {
            return false;
        }
        out.clear();
        if (value.kind == storage::JsonText::Kind::Number && value.number_is_int) {
            out.push_back(value.integer);
            return true;
        }
        if (value.kind == storage::JsonText::Kind::Array) {
            for (const storage::JsonText& item : value.array) {
                if (item.kind == storage::JsonText::Kind::Number && item.number_is_int) {
                    out.push_back(item.integer);
                } else if (item.kind == storage::JsonText::Kind::Number) {
                    out.push_back(std::llround(item.number));
                } else {
                    return false;
                }
            }
            return out.size() == numbers.size();
        }
        out.push_back(std::llround(numbers[0]));
        return true;
    }

    static std::string field_body(const std::string& token, const char* key) {
        const std::size_t pos = token.find(key);
        if (pos == std::string::npos) {
            return "";
        }
        const std::size_t begin = pos + std::char_traits<char>::length(key);
        const std::size_t end = token.find('|', begin);
        const std::string body = token.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::size_t first = body.find_first_not_of(' ');
        if (first == std::string::npos) {
            return "";
        }
        const std::size_t last = body.find_last_not_of(' ');
        return body.substr(first, last - first + 1);
    }

    static std::size_t word_count(const std::string& text) {
        std::size_t count = 0;
        std::istringstream in(text);
        std::string word;
        while (in >> word) {
            ++count;
        }
        return count;
    }

    static std::string pack_rank1(double gamma, const std::vector<long long>& left, const std::vector<long long>& right) {
        char gamma_text[64];
        std::snprintf(gamma_text, sizeof(gamma_text), "%.6g", gamma);
        std::ostringstream out;
        out << "r:1|g:" << gamma_text << "|A:";
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

private:
    static std::vector<double> window(const std::vector<double>& vec, std::size_t start, std::size_t count) {
        if (start + count > vec.size()) {
            throw std::invalid_argument("vector window is outside the source");
        }
        return std::vector<double>(vec.begin() + static_cast<std::ptrdiff_t>(start),
                                   vec.begin() + static_cast<std::ptrdiff_t>(start + count));
    }
};

class BiasFloatScheme : public IEncodingScheme {
public:
    static const BiasFloatScheme& instance() {
        static const BiasFloatScheme scheme;
        return scheme;
    }

    EncodingDescriptor get_descriptor() const override {
        return EncodingDescriptor{
            "bias_float",
            "space-separated finite floats, one per output coordinate",
            "bias[col] = the parsed float.",
            "Each value must be finite."};
    }

    std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const override {
        if (matrix.size() != 1) {
            throw std::invalid_argument("bias_float matrix encoding expects one row");
        }
        return serialize_vector(matrix[0], 0, matrix[0].size());
    }

    std::vector<std::vector<double>> deserialize_matrix(const std::string& token, std::size_t rows, std::size_t cols) const override {
        if (rows != 1) {
            throw std::invalid_argument("bias_float matrix decoding expects one row");
        }
        return {deserialize_vector(token, cols)};
    }

    std::string serialize_vector(const std::vector<double>& vec, std::size_t start, std::size_t count) const override {
        std::ostringstream out;
        out.setf(std::ios::fixed);
        out.precision(2);
        for (std::size_t index = 0; index < count; ++index) {
            if (index > 0) {
                out << ' ';
            }
            out << vec.at(start + index);
        }
        return out.str();
    }

    std::vector<double> deserialize_vector(const std::string& text, std::size_t count) const override {
        return TotrEngine::decode_vector(text, count);
    }

    bool raised(const std::string& updated, const std::string& active, std::size_t count) const override {
        return values_raised(*this, updated, active, count);
    }

    static bool values_raised(const IEncodingScheme& scheme, const std::string& updated, const std::string& active, std::size_t count) {
        try {
            const std::vector<double> next = scheme.deserialize_vector(updated, count);
            const std::vector<double> previous = scheme.deserialize_vector(active, count);
            for (std::size_t index = 0; index < count; ++index) {
                if (next.at(index) >= previous.at(index) + 0.15) {
                    return true;
                }
            }
        } catch (const std::invalid_argument&) {
            return false;
        }
        return false;
    }
};

class BiasDeltaScheme : public IEncodingScheme {
public:
    static const BiasDeltaScheme& instance() {
        static const BiasDeltaScheme scheme;
        return scheme;
    }

    EncodingDescriptor get_descriptor() const override {
        return EncodingDescriptor{
            "bias_delta",
            "q:0.01|D:<ints>",
            "bias[col] = D[col] * 0.01.",
            "Codes are clamped to [-99, 99]. A token without D: is read as space-separated floats."};
    }

    std::string serialize_matrix(const std::vector<std::vector<double>>& matrix) const override {
        if (matrix.size() != 1) {
            throw std::invalid_argument("bias_delta matrix encoding expects one row");
        }
        return serialize_vector(matrix[0], 0, matrix[0].size());
    }

    std::vector<std::vector<double>> deserialize_matrix(const std::string& token, std::size_t rows, std::size_t cols) const override {
        if (rows != 1) {
            throw std::invalid_argument("bias_delta matrix decoding expects one row");
        }
        return {deserialize_vector(token, cols)};
    }

    std::string serialize_vector(const std::vector<double>& vec, std::size_t start, std::size_t count) const override {
        std::ostringstream out;
        out << "q:0.01|D:";
        for (std::size_t index = 0; index < count; ++index) {
            if (index > 0) {
                out << ' ';
            }
            const int code = TotrEngine::clamp_int(static_cast<int>(std::lround(vec.at(start + index) / 0.01)));
            out << code;
        }
        return out.str();
    }

    AcceptedValue accept(const storage::JsonText& value, std::size_t rows, std::size_t cols) const override {
        (void)rows;
        (void)cols;
        if (value.kind == storage::JsonText::Kind::String && !value.text.empty()) {
            return AcceptedValue{value.text, {}};
        }
        if (value.kind == storage::JsonText::Kind::Number || value.kind == storage::JsonText::Kind::Array) {
            std::vector<double> numbers;
            if (value.kind == storage::JsonText::Kind::Number) {
                numbers.push_back(value.number);
            } else {
                for (const storage::JsonText& item : value.array) {
                    if (item.kind != storage::JsonText::Kind::Number) {
                        throw std::invalid_argument("bias_delta completion value is not numeric");
                    }
                    numbers.push_back(item.number);
                }
            }
            if (numbers.empty()) {
                throw std::invalid_argument("bias_delta completion value is empty");
            }
            std::ostringstream out;
            for (std::size_t index = 0; index < numbers.size(); ++index) {
                if (index > 0) {
                    out << ' ';
                }
                char text[64];
                std::snprintf(text, sizeof(text), "%.6g", numbers[index]);
                out << text;
            }
            return AcceptedValue{out.str(), {}};
        }
        throw std::invalid_argument("bias_delta completion value is not a token");
    }

    std::vector<double> deserialize_vector(const std::string& text, std::size_t count) const override {
        if (text.find("D:") == std::string::npos) {
            return BiasFloatScheme::instance().deserialize_vector(text, count);
        }
        const std::vector<int> codes = TotrEngine::field_ints(text, "D:");
        if (codes.size() != count) {
            throw std::invalid_argument("bias_delta vector has the wrong length");
        }
        std::vector<double> values;
        values.reserve(codes.size());
        for (int code : codes) {
            values.push_back(static_cast<double>(code) * 0.01);
        }
        return values;
    }

    bool raised(const std::string& updated, const std::string& active, std::size_t count) const override {
        return BiasFloatScheme::values_raised(*this, updated, active, count);
    }
};

}  // namespace stevejob::messaging
