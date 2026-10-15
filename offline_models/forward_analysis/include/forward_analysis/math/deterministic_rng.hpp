#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace forward_analysis {

struct DeterministicRng {
    // splitmix64 internal state; advancing it yields the next pseudo-random 64-bit value.
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;

    explicit DeterministicRng(std::uint64_t seed = 0x9E3779B97F4A7C15ULL) noexcept
        : state(seed == 0 ? 0x9E3779B97F4A7C15ULL : seed) {}

    // Advances the generator as if next_u64() had been called draws times (each call adds one golden increment).
    void skip(std::uint64_t draws) noexcept { state += draws * 0x9E3779B97F4A7C15ULL; }

    std::uint64_t next_u64() noexcept {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    double uniform(double low = 0.0, double high = 1.0) noexcept {
        constexpr double inv = 1.0 / static_cast<double>(1ULL << 53);
        const double unit = static_cast<double>(next_u64() >> 11) * inv;
        return low + unit * (high - low);
    }

    double normal(double mean = 0.0, double stddev = 1.0) noexcept {
        const double u1 = std::max(1e-12, uniform(0.0, 1.0));
        const double u2 = uniform(0.0, 1.0);
        constexpr double two_pi = 6.28318530717958647692;
        const double z0 = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
        return mean + z0 * stddev;
    }

    Eigen::MatrixXd xavier_matrix(std::size_t rows, std::size_t cols) {
        const Eigen::Index r_count = static_cast<Eigen::Index>(rows);
        const Eigen::Index c_count = static_cast<Eigen::Index>(cols);
        Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(r_count, c_count);
        if (rows == 0 || cols == 0) {
            return mat;
        }
        const double limit = std::sqrt(6.0 / static_cast<double>(rows + cols));
        for (Eigen::Index r = 0; r < r_count; ++r) {
            for (Eigen::Index c = 0; c < c_count; ++c) {
                mat(r, c) = uniform(-limit, limit);
            }
        }
        return mat;
    }

    Eigen::VectorXd uniform_vector(std::size_t dim, double low = -0.1, double high = 0.1) {
        Eigen::VectorXd vec(static_cast<Eigen::Index>(dim));
        for (Eigen::Index i = 0; i < vec.size(); ++i) {
            vec(i) = uniform(low, high);
        }
        return vec;
    }
};

}  // namespace forward_analysis
