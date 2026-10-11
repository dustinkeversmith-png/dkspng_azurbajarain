#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

namespace stevejob::core {

inline void require_finite(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("tensor contains a non-finite number");
    }
}

inline bool little_endian_host() {
    const unsigned char probe_bytes[] = {1, 0};
    const std::uint16_t probe = 1;
    return std::memcmp(&probe, probe_bytes, sizeof(probe)) == 0;
}

inline void require_little_endian(const char* message) {
    if (!little_endian_host()) {
        throw std::runtime_error(message);
    }
}

inline void append_u16(std::string& out, std::uint16_t value) {
    out.push_back(static_cast<char>(value & 0xff));
    out.push_back(static_cast<char>((value >> 8) & 0xff));
}

inline void append_u32(std::string& out, std::uint32_t value) {
    out.push_back(static_cast<char>(value & 0xff));
    out.push_back(static_cast<char>((value >> 8) & 0xff));
    out.push_back(static_cast<char>((value >> 16) & 0xff));
    out.push_back(static_cast<char>((value >> 24) & 0xff));
}

inline void append_u64(std::string& out, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<char>((value >> shift) & 0xff));
    }
}

inline void append_f64(std::string& out, double value) {
    require_finite(value);
    unsigned char bytes[sizeof(double)];
    std::memcpy(bytes, &value, sizeof(double));
    out.append(reinterpret_cast<const char*>(bytes), sizeof(double));
}

inline std::uint16_t read_u16(const std::string& in, std::size_t& cursor) {
    if (cursor + 2 > in.size()) {
        throw std::invalid_argument("SJC1 body ended early");
    }
    const auto first = static_cast<unsigned char>(in[cursor]);
    const auto second = static_cast<unsigned char>(in[cursor + 1]);
    cursor += 2;
    return static_cast<std::uint16_t>(first | (second << 8));
}

inline std::uint32_t read_u32(const std::string& in, std::size_t& cursor) {
    if (cursor + 4 > in.size()) {
        throw std::invalid_argument("SJC1 body ended early");
    }
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(in[cursor++])) << shift;
    }
    return value;
}

inline std::uint64_t read_u64(const std::string& in, std::size_t& cursor) {
    if (cursor + 8 > in.size()) {
        throw std::invalid_argument("SJC1 body ended early");
    }
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) {
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(in[cursor++])) << shift;
    }
    return value;
}

inline double read_f64(const std::string& in, std::size_t& cursor) {
    if (cursor + sizeof(double) > in.size()) {
        throw std::invalid_argument("SJC1 body ended early");
    }
    double value = 0.0;
    std::memcpy(&value, in.data() + cursor, sizeof(double));
    cursor += sizeof(double);
    require_finite(value);
    return value;
}

inline double parse_chars_double(const char* first, const char* last) {
    double value = 0;
    const auto parsed = std::from_chars(first, last, value);
    if (parsed.ec != std::errc() || parsed.ptr != last || !std::isfinite(value)) {
        throw std::invalid_argument("invalid number");
    }
    return value;
}

inline int hex_nibble(char digit) {
    if (digit >= '0' && digit <= '9') {
        return digit - '0';
    }
    if (digit >= 'a' && digit <= 'f') {
        return digit - 'a' + 10;
    }
    if (digit >= 'A' && digit <= 'F') {
        return digit - 'A' + 10;
    }
    throw std::invalid_argument("bad unicode escape");
}

inline void append_utf8(std::string& out, char32_t code_point) {
    if (code_point > 0x10FFFF || (code_point >= 0xD800 && code_point <= 0xDFFF)) {
        throw std::invalid_argument("invalid unicode escape");
    }
    if (code_point < 0x80) {
        out.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
}

class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed == 0 ? 0x9E3779B97F4A7C15ULL : seed) {}

    std::uint64_t next() {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        return state_ * 0x2545F4914F6CDD1DULL;
    }

    double uniform() {
        return (static_cast<double>(next() >> 11) + 1.0) / 9007199254740992.0;
    }

    void normal_pair(double& first, double& second) {
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle = 6.28318530717958647692 * uniform();
        first = radius * std::cos(angle);
        second = radius * std::sin(angle);
    }

private:
    std::uint64_t state_;
};

}  // namespace stevejob::core
