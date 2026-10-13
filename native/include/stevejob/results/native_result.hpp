#pragma once

#include "stevejob/core/config.hpp"
#include "stevejob/core/hash.hpp"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace stevejob::results {

struct NativeResult {
    bool cache_hit = false;
    int iterations = 0;
    int threads = 1;
    std::uint64_t count = 0;
    std::uint64_t checksum = 0;
    std::int64_t elapsed_ns = 0;
    std::map<std::string, double> metrics;

    double metric(const std::string& key, double fallback = 0.0) const {
        const auto found = metrics.find(key);
        return found == metrics.end() ? fallback : found->second;
    }

    py::dict to_python_dict() const {
        py::dict out;
        out["cache_hit"] = cache_hit;
        out["iterations"] = iterations;
        out["threads"] = threads;
        out["count"] = count;
        out["checksum"] = stevejob::core::to_hex(checksum);
        out["elapsed_ns"] = elapsed_ns;
        py::dict numbers;
        for (const auto& item : metrics) {
            numbers[py::str(item.first)] = item.second;
            out[py::str(item.first)] = item.second;
        }
        out["metrics"] = numbers;
        return out;
    }
};

class SampleProcessor {
public:
    int choose_threads(std::size_t count) const {
        if (count < stevejob::config::kParallelThreshold) {
            return 1;
        }
        unsigned workers = std::thread::hardware_concurrency();
        if (workers == 0) {
            workers = 2;
        }
        if (workers > 8) {
            workers = 8;
        }
        return static_cast<int>(workers);
    }

    void smooth_pass(std::vector<double>& current, std::vector<double>& next, int threads) const {
        const std::size_t count = current.size();
        if (threads <= 1 || count < stevejob::config::kParallelThreshold) {
            for (std::size_t index = 0; index < count; ++index) {
                const std::size_t neighbor = (index + 1 == count) ? 0 : index + 1;
                next[index] = 0.5 * current[index] + 0.5 * current[neighbor];
            }
            return;
        }

        const std::size_t chunk =
            (count + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(threads));
        for (int worker = 0; worker < threads; ++worker) {
            const std::size_t begin = static_cast<std::size_t>(worker) * chunk;
            if (begin >= count) {
                break;
            }
            const std::size_t end = std::min(count, begin + chunk);
            workers.emplace_back([&current, &next, begin, end, count]() {
                for (std::size_t index = begin; index < end; ++index) {
                    const std::size_t neighbor = (index + 1 == count) ? 0 : index + 1;
                    next[index] = 0.5 * current[index] + 0.5 * current[neighbor];
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    NativeResult reduce_samples(const std::vector<double>& samples,
                                int iterations,
                                int threads,
                                std::int64_t elapsed_ns) const {
        const double count = static_cast<double>(samples.size());
        double sum = 0;
        double sum_squares = 0;
        double min_value = samples[0];
        double max_value = samples[0];
        for (double sample : samples) {
            sum += sample;
            sum_squares += sample * sample;
            if (sample < min_value) {
                min_value = sample;
            }
            if (sample > max_value) {
                max_value = sample;
            }
        }
        const double mean = sum / count;
        double moment = 0;
        for (double sample : samples) {
            const double delta = sample - mean;
            moment += delta * delta;
        }

        NativeResult result;
        result.iterations = iterations;
        result.threads = threads;
        result.count = samples.size();
        result.checksum = stevejob::core::fnv1a64(samples.data(), samples.size());
        result.elapsed_ns = elapsed_ns;
        result.metrics["sum"] = sum;
        result.metrics["mean"] = mean;
        result.metrics["variance"] = moment / count;
        result.metrics["min"] = min_value;
        result.metrics["max"] = max_value;
        result.metrics["rms"] = std::sqrt(sum_squares / count);
        return result;
    }
};

}  // namespace stevejob::results
