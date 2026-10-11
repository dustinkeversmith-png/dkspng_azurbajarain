#pragma once

#include "stevejob/core/helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace stevejob::core {

inline std::uint64_t fnv1a64_bytes(const unsigned char* data, std::size_t count) {
    std::uint64_t hash = 14695981039346656037ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= data[index];
        hash *= prime;
    }
    return hash;
}

inline std::uint64_t fnv1a64(const double* data, std::size_t count) {
    require_little_endian("native_engine checksum requires a little-endian host");
    if (count == 0 || data == nullptr) {
        return 14695981039346656037ull;
    }
    return fnv1a64_bytes(reinterpret_cast<const unsigned char*>(data), count * sizeof(double));
}

inline std::string to_hex(std::uint64_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text(16, '0');
    for (int index = 15; index >= 0; --index) {
        text[static_cast<std::size_t>(index)] = digits[value & 0xF];
        value >>= 4;
    }
    return text;
}

}  // namespace stevejob::core
