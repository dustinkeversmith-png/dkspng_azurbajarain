#pragma once

#include <array>
#include <cctype>
#include <cstddef>
#include <string>

namespace forward_analysis {

// English cardinal words for 0..99, both directions.
struct NumberWords {
    static std::string word(std::size_t n) {
        if (n < 20) {
            return units()[n];
        }
        if (n < 100) {
            const std::string ten = tens()[n / 10];
            return n % 10 == 0 ? ten : ten + "-" + units()[n % 10];
        }
        return std::to_string(n);
    }

    static std::string capitalized(std::size_t n) {
        std::string out = word(n);
        out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
        return out;
    }

    // Parses "twenty-four", "Eleven", "7"; returns false for anything else.
    static bool parse(const std::string& text, std::size_t& value) {
        std::string lower;
        for (char c : text) {
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (!lower.empty() && std::isdigit(static_cast<unsigned char>(lower[0]))) {
            for (char c : lower) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    return false;
                }
            }
            value = static_cast<std::size_t>(std::stoul(lower));
            return true;
        }
        for (std::size_t n = 0; n < 100; ++n) {
            if (word(n) == lower) {
                value = n;
                return true;
            }
        }
        return false;
    }

private:
    static const std::array<std::string, 20>& units() {
        static const std::array<std::string, 20> words = {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
        return words;
    }

    static const std::array<std::string, 10>& tens() {
        static const std::array<std::string, 10> words = {
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
        return words;
    }
};

}  // namespace forward_analysis
