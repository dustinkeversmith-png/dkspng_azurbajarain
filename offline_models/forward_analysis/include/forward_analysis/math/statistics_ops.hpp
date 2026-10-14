#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace forward_analysis {

struct StatisticsOps {
    static double standard_deviation(const Eigen::VectorXd& values) {
        if (values.size() < 2) {
            return 0.0;
        }
        return std::sqrt((values.array() - values.mean()).square().mean());
    }

    // Pearson correlation; 0 when either side has no spread.
    static double pearson(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        if (a.size() != b.size() || a.size() < 2) {
            return 0.0;
        }
        const Eigen::ArrayXd ac = a.array() - a.mean();
        const Eigen::ArrayXd bc = b.array() - b.mean();
        const double denom = std::sqrt(ac.square().sum() * bc.square().sum());
        return denom > 1e-12 ? (ac * bc).sum() / denom : 0.0;
    }

    // 0-based ranks with ties sharing their average rank.
    static Eigen::VectorXd ranks(const Eigen::VectorXd& values) {
        const Eigen::Index n = values.size();
        std::vector<Eigen::Index> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), Eigen::Index{0});
        std::stable_sort(order.begin(), order.end(),
                         [&values](Eigen::Index l, Eigen::Index r) { return values(l) < values(r); });
        Eigen::VectorXd out(n);
        Eigen::Index start = 0;
        while (start < n) {
            Eigen::Index end = start;
            while (end + 1 < n && values(order[static_cast<std::size_t>(end + 1)]) ==
                                      values(order[static_cast<std::size_t>(start)])) {
                ++end;
            }
            const double shared = 0.5 * static_cast<double>(start + end);
            for (Eigen::Index k = start; k <= end; ++k) {
                out(order[static_cast<std::size_t>(k)]) = shared;
            }
            start = end + 1;
        }
        return out;
    }

    static double spearman(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return pearson(ranks(a), ranks(b));
    }

    // True when every pair of samples is ordered the same way in both vectors.
    static bool same_order(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (Eigen::Index i = 0; i < a.size(); ++i) {
            for (Eigen::Index j = i + 1; j < a.size(); ++j) {
                const double da = a(i) - a(j);
                const double db = b(i) - b(j);
                if ((da > 0.0 && db <= 0.0) || (da < 0.0 && db >= 0.0) || (da == 0.0 && db != 0.0)) {
                    return false;
                }
            }
        }
        return true;
    }
};

}  // namespace forward_analysis
