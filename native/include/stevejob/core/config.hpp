#pragma once

#include <cstddef>

namespace stevejob::config {

struct LearningConfig {
    int max_iterations{256};
    std::size_t max_samples{5000000};
    std::size_t max_payload_bytes{64u * 1024u * 1024u};
    std::size_t max_cache_entries{4096};
    std::size_t parallel_threshold{65536};
    int max_candidates{3};
    int max_parameters{22};
    int max_residuals{8};
    int max_anomalies{4};
    std::size_t max_blob_bytes{16384};
    double max_learning_rate{0.05};
    double default_learning_rate{0.01};
    double default_weight_decay{0.0};
    double min_dampening{0.1};
    double max_dampening{1.0};
    std::size_t max_input_rows{8};
    std::size_t max_input_cols{8};
    int int_bound{25};
    std::size_t visible_history_count{3};
    double min_commit_drop{1e-4};
    double divergence_threshold{100.0};
};

inline constexpr LearningConfig kDefaultConfig{};
inline constexpr int kMaxIterations = kDefaultConfig.max_iterations;
inline constexpr std::size_t kMaxSamples = kDefaultConfig.max_samples;
inline constexpr std::size_t kMaxPayloadBytes = kDefaultConfig.max_payload_bytes;
inline constexpr std::size_t kMaxCacheEntries = kDefaultConfig.max_cache_entries;
inline constexpr std::size_t kParallelThreshold = kDefaultConfig.parallel_threshold;

}  // namespace stevejob::config

