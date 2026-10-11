#pragma once

#include "stevejob/core/config.hpp"
#include "stevejob/messaging/parser.hpp"
#include "stevejob/results/native_result.hpp"

#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace stevejob::core {

class IdempotencyCache {
public:
    bool get(const std::string& key, stevejob::results::NativeResult& out) {
        if (key.empty()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) {
            return false;
        }
        out = found->second;
        out.cache_hit = true;
        return true;
    }

    void put(const std::string& key, const stevejob::results::NativeResult& value) {
        if (key.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.find(key) != entries_.end()) {
            return;
        }
        if (order_.size() >= stevejob::config::kMaxCacheEntries) {
            entries_.erase(order_.front());
            order_.pop_front();
        }
        entries_.emplace(key, value);
        order_.push_back(key);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
        order_.clear();
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, stevejob::results::NativeResult> entries_;
    std::deque<std::string> order_;
};

inline IdempotencyCache& process_cache() {
    static IdempotencyCache cache;
    return cache;
}

inline stevejob::results::NativeResult crunch(const std::string& input_json,
                                              int iterations,
                                              const std::string& idempotency_key,
                                              IdempotencyCache& cache) {
    if (input_json.size() > stevejob::config::kMaxPayloadBytes) {
        throw std::invalid_argument("payload exceeds 64 MiB");
    }
    if (iterations < 1 || iterations > stevejob::config::kMaxIterations) {
        throw std::invalid_argument("iterations must be from 1 to 256");
    }

    stevejob::messaging::Parsed parsed = stevejob::messaging::Parser(input_json).parse();
    const std::string& key = idempotency_key.empty() ? parsed.idempotency_key : idempotency_key;
    stevejob::results::NativeResult cached;
    if (cache.get(key, cached)) {
        return cached;
    }

    const auto started = std::chrono::steady_clock::now();
    std::vector<double> current = std::move(parsed.samples);
    std::vector<double> next(current.size());
    const int threads = stevejob::results::choose_threads(current.size());
    for (int pass = 0; pass < iterations; ++pass) {
        stevejob::results::smooth_pass(current, next, threads);
        current.swap(next);
    }
    const auto finished = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count();
    stevejob::results::NativeResult result =
        stevejob::results::reduce_samples(current, iterations, threads, elapsed_ns);
    cache.put(key, result);
    return result;
}

}  // namespace stevejob::core
