#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stevejob::storage {

struct JsonText {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    bool number_is_int = false;
    long long integer = 0;
    std::string text;
    std::vector<JsonText> array;
    std::vector<std::pair<std::string, JsonText>> object;

    static JsonText parse(const std::string& raw) {
        Parser parser(raw);
        JsonText value = parser.parse_value();
        parser.skip();
        if (parser.index != raw.size()) {
            throw std::invalid_argument("blob payload is not JSON");
        }
        return value;
    }

private:
    struct Parser {
        const std::string& raw;
        std::size_t index = 0;

        explicit Parser(const std::string& body) : raw(body) {}

        void skip() {
            while (index < raw.size() && std::isspace(static_cast<unsigned char>(raw[index]))) {
                ++index;
            }
        }

        char peek() {
            skip();
            if (index >= raw.size()) {
                throw std::invalid_argument("blob payload is not JSON");
            }
            return raw[index];
        }

        JsonText parse_value() {
            const char token = peek();
            if (token == '{') {
                return parse_object();
            }
            if (token == '[') {
                return parse_array();
            }
            if (token == '"') {
                JsonText value;
                value.kind = Kind::String;
                value.text = parse_string();
                return value;
            }
            if (token == 't' || token == 'f') {
                return parse_bool();
            }
            if (token == 'n') {
                expect("null");
                return JsonText();
            }
            if (token == '-' || (token >= '0' && token <= '9')) {
                return parse_number();
            }
            throw std::invalid_argument("blob payload is not JSON");
        }

        JsonText parse_object() {
            expect_char('{');
            JsonText value;
            value.kind = Kind::Object;
            skip();
            if (peek() == '}') {
                ++index;
                return value;
            }
            while (true) {
                if (peek() != '"') {
                    throw std::invalid_argument("blob payload is not JSON");
                }
                std::string key = parse_string();
                expect_char(':');
                value.object.emplace_back(std::move(key), parse_value());
                skip();
                if (peek() == ',') {
                    ++index;
                    continue;
                }
                expect_char('}');
                break;
            }
            return value;
        }

        JsonText parse_array() {
            expect_char('[');
            JsonText value;
            value.kind = Kind::Array;
            skip();
            if (peek() == ']') {
                ++index;
                return value;
            }
            while (true) {
                value.array.push_back(parse_value());
                skip();
                if (peek() == ',') {
                    ++index;
                    continue;
                }
                expect_char(']');
                break;
            }
            return value;
        }

        JsonText parse_bool() {
            JsonText value;
            value.kind = Kind::Bool;
            if (raw.compare(index, 4, "true") == 0) {
                index += 4;
                value.boolean = true;
                return value;
            }
            if (raw.compare(index, 5, "false") == 0) {
                index += 5;
                return value;
            }
            throw std::invalid_argument("blob payload is not JSON");
        }

        JsonText parse_number() {
            const std::size_t start = index;
            if (raw[index] == '-') {
                ++index;
            }
            if (index >= raw.size() || raw[index] < '0' || raw[index] > '9') {
                throw std::invalid_argument("blob payload is not JSON");
            }
            while (index < raw.size() && raw[index] >= '0' && raw[index] <= '9') {
                ++index;
            }
            bool integral = true;
            if (index < raw.size() && raw[index] == '.') {
                integral = false;
                ++index;
                while (index < raw.size() && raw[index] >= '0' && raw[index] <= '9') {
                    ++index;
                }
            }
            if (index < raw.size() && (raw[index] == 'e' || raw[index] == 'E')) {
                integral = false;
                ++index;
                if (index < raw.size() && (raw[index] == '+' || raw[index] == '-')) {
                    ++index;
                }
                while (index < raw.size() && raw[index] >= '0' && raw[index] <= '9') {
                    ++index;
                }
            }
            const std::string token = raw.substr(start, index - start);
            JsonText value;
            value.kind = Kind::Number;
            value.number_is_int = integral;
            if (integral) {
                try {
                    value.integer = std::stoll(token);
                    value.number = static_cast<double>(value.integer);
                } catch (const std::exception&) {
                    value.number_is_int = false;
                    value.number = std::stod(token);
                }
            } else {
                value.number = std::stod(token);
            }
            if (!std::isfinite(value.number)) {
                throw std::invalid_argument("blob payload is not JSON");
            }
            return value;
        }

        std::string parse_string() {
            expect_char('"');
            std::string text;
            while (index < raw.size()) {
                const char token = raw[index++];
                if (token == '"') {
                    return text;
                }
                if (token != '\\') {
                    text.push_back(token);
                    continue;
                }
                if (index >= raw.size()) {
                    throw std::invalid_argument("blob payload is not JSON");
                }
                const char escaped = raw[index++];
                switch (escaped) {
                    case '"':
                    case '\\':
                    case '/':
                        text.push_back(escaped);
                        break;
                    case 'b':
                        text.push_back('\b');
                        break;
                    case 'f':
                        text.push_back('\f');
                        break;
                    case 'n':
                        text.push_back('\n');
                        break;
                    case 'r':
                        text.push_back('\r');
                        break;
                    case 't':
                        text.push_back('\t');
                        break;
                    case 'u':
                        append_codepoint(text, read_codepoint());
                        break;
                    default:
                        throw std::invalid_argument("blob payload is not JSON");
                }
            }
            throw std::invalid_argument("blob payload is not JSON");
        }

        int read_hex4() {
            if (index + 4 > raw.size()) {
                throw std::invalid_argument("blob payload is not JSON");
            }
            int code = 0;
            for (int nibble = 0; nibble < 4; ++nibble) {
                const char token = raw[index++];
                code <<= 4;
                if (token >= '0' && token <= '9') {
                    code += token - '0';
                } else if (token >= 'a' && token <= 'f') {
                    code += token - 'a' + 10;
                } else if (token >= 'A' && token <= 'F') {
                    code += token - 'A' + 10;
                } else {
                    throw std::invalid_argument("blob payload is not JSON");
                }
            }
            return code;
        }

        int read_codepoint() {
            const int code = read_hex4();
            if (code >= 0xD800 && code <= 0xDBFF) {
                if (index + 2 <= raw.size() && raw[index] == '\\' && raw[index + 1] == 'u') {
                    index += 2;
                    const int low = read_hex4();
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        return 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                }
                throw std::invalid_argument("blob payload is not JSON");
            }
            return code;
        }

        void expect(const char* literal) {
            const std::size_t length = std::char_traits<char>::length(literal);
            if (raw.compare(index, length, literal) != 0) {
                throw std::invalid_argument("blob payload is not JSON");
            }
            index += length;
        }

        void expect_char(char token) {
            if (peek() != token) {
                throw std::invalid_argument("blob payload is not JSON");
            }
            ++index;
        }
    };

    static void append_codepoint(std::string& text, int code) {
        if (code < 0 || code > 0x10FFFF) {
            throw std::invalid_argument("blob payload is not JSON");
        }
        if (code < 0x80) {
            text.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            text.push_back(static_cast<char>(0xC0 | (code >> 6)));
            text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            text.push_back(static_cast<char>(0xE0 | (code >> 12)));
            text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            text.push_back(static_cast<char>(0xF0 | (code >> 18)));
            text.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }
};

}  // namespace stevejob::storage
