#pragma once

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Splits prose into word and punctuation lexemes and joins them back with natural spacing.
// A "<...>" span (slot token such as "<DIM:W_1|w>") is always one lexeme.
struct LexemeSplitter {
    static bool is_punctuation(char c) noexcept {
        return c == ',' || c == '.' || c == ';' || c == ':' || c == '(' || c == ')' || c == '/';
    }

    static std::vector<std::string> split(const std::string& text) {
        std::vector<std::string> lexemes;
        std::string current;
        bool in_slot = false;
        const auto flush = [&]() {
            if (!current.empty()) {
                lexemes.push_back(current);
                current.clear();
            }
        };
        for (char c : text) {
            if (in_slot) {
                current.push_back(c);
                if (c == '>') {
                    flush();
                    in_slot = false;
                }
            } else if (c == '<') {
                flush();
                current.push_back(c);
                in_slot = true;
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                flush();
            } else if (is_punctuation(c)) {
                flush();
                lexemes.emplace_back(1, c);
            } else {
                current.push_back(c);
            }
        }
        flush();
        return lexemes;
    }

    // A "." between a lexeme ending in a digit and one starting with a digit is a decimal point ("1.825",
    // "x3-0.5·x7") and is joined without spaces.
    static std::string join(const std::vector<std::string>& lexemes) {
        std::string out;
        bool suppress_next_space = true;
        for (std::size_t i = 0; i < lexemes.size(); ++i) {
            const std::string& lex = lexemes[i];
            const bool closes = lex == "," || lex == "." || lex == ";" || lex == ":" || lex == ")" || lex == "/";
            if (!suppress_next_space && !closes) {
                out += ' ';
            }
            out += lex;
            suppress_next_space = (lex == "(" || lex == "/") || is_decimal_point(lexemes, i);
        }
        return out;
    }

    static bool is_decimal_point(const std::vector<std::string>& lexemes, std::size_t i) {
        return lexemes[i] == "." && i > 0 && i + 1 < lexemes.size() && !lexemes[i - 1].empty() &&
               !lexemes[i + 1].empty() && std::isdigit(static_cast<unsigned char>(lexemes[i - 1].back())) &&
               std::isdigit(static_cast<unsigned char>(lexemes[i + 1].front()));
    }

    static std::string lowercase(const std::string& text) {
        std::string out = text;
        for (char& c : out) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return out;
    }
};

}  // namespace forward_analysis
