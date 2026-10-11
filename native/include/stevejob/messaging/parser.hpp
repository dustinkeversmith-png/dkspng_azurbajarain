#pragma once

#include "stevejob/core/config.hpp"
#include "stevejob/core/helpers.hpp"

#include <cctype>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace stevejob::messaging {

struct Parsed {
    std::vector<double> samples;
    std::string idempotency_key;
};

class Parser {
public:
    explicit Parser(const std::string& text) : in_(text) {}

    Parsed parse() {
        skip_bom();
        skip_ws();
        if (i_ >= in_.size()) {
            fail("payload has no samples");
        }
        Parsed out;
        const char kind = in_[i_];
        if (kind == '[') {
            out.samples = parse_number_array();
        } else if (kind == '{') {
            parse_object(out, true);
            if (out.samples.empty()) {
                fail("object payload is missing samples");
            }
        } else {
            out.samples = parse_loose();
        }
        skip_ws();
        if (i_ != in_.size()) {
            fail("trailing data in payload");
        }
        if (out.samples.empty()) {
            fail("payload has no samples");
        }
        if (out.samples.size() > stevejob::config::kMaxSamples) {
            fail("payload exceeds 5000000 samples");
        }
        return out;
    }

private:
    const std::string& in_;
    std::size_t i_ = 0;

    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument(message);
    }

    void skip_bom() {
        if (in_.size() >= 3 &&
            static_cast<unsigned char>(in_[0]) == 0xEF &&
            static_cast<unsigned char>(in_[1]) == 0xBB &&
            static_cast<unsigned char>(in_[2]) == 0xBF) {
            i_ = 3;
        }
    }

    void skip_ws() {
        while (i_ < in_.size() && std::isspace(static_cast<unsigned char>(in_[i_]))) {
            ++i_;
        }
    }

    bool consume(char expected) {
        skip_ws();
        if (i_ < in_.size() && in_[i_] == expected) {
            ++i_;
            return true;
        }
        return false;
    }

    char peek() {
        skip_ws();
        if (i_ >= in_.size()) {
            fail("unexpected end of payload");
        }
        return in_[i_];
    }

    bool consume_literal(const char* literal) {
        skip_ws();
        const std::size_t n = std::strlen(literal);
        if (i_ + n <= in_.size() && in_.compare(i_, n, literal) == 0) {
            const std::size_t next = i_ + n;
            if (next == in_.size() || !std::isalnum(static_cast<unsigned char>(in_[next]))) {
                i_ = next;
                return true;
            }
        }
        return false;
    }

    double parse_number() {
        skip_ws();
        const std::size_t start = i_;
        if (i_ < in_.size() && in_[i_] == '-') {
            ++i_;
        }
        if (i_ >= in_.size() || !std::isdigit(static_cast<unsigned char>(in_[i_]))) {
            fail("expected a number");
        }
        while (i_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[i_]))) {
            ++i_;
        }
        if (i_ < in_.size() && in_[i_] == '.') {
            ++i_;
            if (i_ >= in_.size() || !std::isdigit(static_cast<unsigned char>(in_[i_]))) {
                fail("expected digits after decimal point");
            }
            while (i_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[i_]))) {
                ++i_;
            }
        }
        if (i_ < in_.size() && (in_[i_] == 'e' || in_[i_] == 'E')) {
            ++i_;
            if (i_ < in_.size() && (in_[i_] == '+' || in_[i_] == '-')) {
                ++i_;
            }
            if (i_ >= in_.size() || !std::isdigit(static_cast<unsigned char>(in_[i_]))) {
                fail("expected digits in exponent");
            }
            while (i_ < in_.size() && std::isdigit(static_cast<unsigned char>(in_[i_]))) {
                ++i_;
            }
        }
        return stevejob::core::parse_chars_double(in_.data() + start, in_.data() + i_);
    }

    char32_t read_u_escape() {
        if (i_ + 4 > in_.size()) {
            fail("bad unicode escape");
        }
        char32_t code_point = 0;
        for (int nibble = 0; nibble < 4; ++nibble) {
            code_point = (code_point << 4) |
                         static_cast<char32_t>(stevejob::core::hex_nibble(in_[i_++]));
        }
        return code_point;
    }

    std::string parse_string() {
        if (!consume('"')) {
            fail("expected a string");
        }
        std::string out;
        while (i_ < in_.size()) {
            const unsigned char c = static_cast<unsigned char>(in_[i_++]);
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (i_ >= in_.size()) {
                    fail("bad escape");
                }
                const char escaped = in_[i_++];
                switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(escaped);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    char32_t code_point = read_u_escape();
                    if (code_point >= 0xD800 && code_point <= 0xDBFF) {
                        if (!(i_ + 2 <= in_.size() && in_[i_] == '\\' && in_[i_ + 1] == 'u')) {
                            fail("lone unicode surrogate");
                        }
                        i_ += 2;
                        const char32_t low = read_u_escape();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            fail("invalid unicode surrogate pair");
                        }
                        code_point = 0x10000 + (((code_point - 0xD800) << 10) | (low - 0xDC00));
                    } else if (code_point >= 0xDC00 && code_point <= 0xDFFF) {
                        fail("lone unicode surrogate");
                    }
                    stevejob::core::append_utf8(out, code_point);
                    break;
                }
                default:
                    fail("bad escape");
                }
            } else if (c < 0x20) {
                fail("raw control character in string");
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
        fail("unterminated string");
    }

    std::vector<double> parse_number_array() {
        if (!consume('[')) {
            fail("expected an array");
        }
        std::vector<double> samples;
        if (consume(']')) {
            return samples;
        }
        while (true) {
            samples.push_back(parse_number());
            if (samples.size() > stevejob::config::kMaxSamples) {
                fail("payload exceeds 5000000 samples");
            }
            if (consume(']')) {
                return samples;
            }
            if (!consume(',')) {
                fail("expected a comma in samples array");
            }
        }
    }

    void skip_value() {
        const char c = peek();
        if (c == '"') {
            parse_string();
            return;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            parse_number();
            return;
        }
        if (c == '[') {
            skip_array();
            return;
        }
        if (c == '{') {
            Parsed ignored;
            parse_object(ignored, false);
            return;
        }
        if (consume_literal("true") || consume_literal("false") || consume_literal("null")) {
            return;
        }
        fail("unsupported JSON value");
    }

    void skip_array() {
        if (!consume('[')) {
            fail("expected an array");
        }
        if (consume(']')) {
            return;
        }
        while (true) {
            skip_value();
            if (consume(']')) {
                return;
            }
            if (!consume(',')) {
                fail("expected a comma");
            }
        }
    }

    void parse_object(Parsed& out, bool capture) {
        if (!consume('{')) {
            fail("expected an object");
        }
        if (consume('}')) {
            return;
        }
        while (true) {
            const std::string key = parse_string();
            if (!consume(':')) {
                fail("expected a colon");
            }
            if (capture && key == "samples") {
                if (!out.samples.empty()) {
                    fail("duplicate samples field");
                }
                out.samples = parse_number_array();
            } else if (capture && key == "idempotency_key") {
                if (!out.idempotency_key.empty()) {
                    fail("duplicate idempotency_key field");
                }
                out.idempotency_key = parse_string();
            } else {
                skip_value();
            }
            if (consume('}')) {
                return;
            }
            if (!consume(',')) {
                fail("expected a comma in object");
            }
        }
    }

    std::vector<double> parse_loose() {
        std::vector<double> samples;
        while (true) {
            skip_ws();
            if (i_ >= in_.size()) {
                break;
            }
            samples.push_back(parse_number());
            if (samples.size() > stevejob::config::kMaxSamples) {
                fail("payload exceeds 5000000 samples");
            }
            skip_ws();
            if (i_ < in_.size() && in_[i_] == ',') {
                ++i_;
                continue;
            }
            if (i_ < in_.size()) {
                const unsigned char next = static_cast<unsigned char>(in_[i_]);
                if (!(std::isdigit(next) || in_[i_] == '-')) {
                    fail("loose payloads must be numbers separated by commas or whitespace");
                }
            }
        }
        return samples;
    }
};

}  // namespace stevejob::messaging
