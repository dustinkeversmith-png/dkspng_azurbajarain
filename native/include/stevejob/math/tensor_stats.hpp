#pragma once

#include "stevejob/core/helpers.hpp"

#include <cstddef>
#include <vector>

namespace stevejob::math {

class TensorStats {
public:
    static TensorStats from_matrix(const std::vector<std::vector<double>>& matrix) {
        TensorStats stats;
        stats.compute(matrix);
        return stats;
    }

    void compute(const std::vector<std::vector<double>>& matrix) {
        double sum = 0.0;
        double min_val = 0.0;
        double max_val = 0.0;
        std::size_t count = 0;
        for (const std::vector<double>& row : matrix) {
            for (double value : row) {
                stevejob::core::require_finite(value);
                if (count == 0) {
                    min_val = value;
                    max_val = value;
                } else {
                    if (value < min_val) {
                        min_val = value;
                    }
                    if (value > max_val) {
                        max_val = value;
                    }
                }
                sum += value;
                ++count;
            }
        }

        count_ = count;
        if (count == 0) {
            mean_ = 0.0;
            variance_ = 0.0;
            min_val_ = 0.0;
            max_val_ = 0.0;
            return;
        }
        mean_ = sum / static_cast<double>(count);
        min_val_ = min_val;
        max_val_ = max_val;

        double moment = 0.0;
        for (const std::vector<double>& row : matrix) {
            for (double value : row) {
                const double delta = value - mean_;
                moment += delta * delta;
            }
        }
        variance_ = moment / static_cast<double>(count);
    }

    double mean() const { return mean_; }
    double variance() const { return variance_; }
    double min_val() const { return min_val_; }
    double max_val() const { return max_val_; }
    std::size_t count() const { return count_; }

private:
    double mean_ = 0.0;
    double variance_ = 0.0;
    double max_val_ = 0.0;
    double min_val_ = 0.0;
    std::size_t count_ = 0;
};

}  // namespace stevejob::math
