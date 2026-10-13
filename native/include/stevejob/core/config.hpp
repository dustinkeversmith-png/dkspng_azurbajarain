#pragma once

#include <cstddef>

namespace stevejob::config {

inline constexpr int kMaxIterations = 256;
inline constexpr std::size_t kMaxSamples = 5000000;
inline constexpr std::size_t kMaxPayloadBytes = 64 * 1024 * 1024;
inline constexpr std::size_t kMaxCacheEntries = 4096;
inline constexpr std::size_t kParallelThreshold = 65536;

}  // namespace stevejob::config
