#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace burger {

struct DataPoint {
    double x = 0.0;
    double t = 0.0;
    double u_true = 0.0;
    std::vector<double> encoded_input;
};

class BurgerDatasetGenerator {
public:
    static constexpr double kPi = 3.14159265358979323846;

    // Cole-Hopf analytical evaluation for initial condition u(x,0) = -sin(pi*x)
    static double cole_hopf_solution(double x, double t, double nu = 0.01 / kPi, int quadrature_nodes = 256) {
        if (t <= 1e-12) {
            return -std::sin(kPi * x);
        }
        double num = 0.0;
        double den = 0.0;
        const double eta_min = -3.0;
        const double eta_max = 3.0;
        const double d_eta = (eta_max - eta_min) / static_cast<double>(quadrature_nodes);

        for (int i = 0; i <= quadrature_nodes; ++i) {
            const double eta = eta_min + static_cast<double>(i) * d_eta;
            const double diff = x - eta;
            const double phi = (diff * diff) / (4.0 * nu * t) + (1.0 - std::cos(kPi * eta)) / (2.0 * kPi * nu);
            const double weight = (i == 0 || i == quadrature_nodes) ? 0.5 : 1.0;
            const double exp_term = std::exp(-phi);
            den += weight * exp_term * d_eta;
            num += weight * (diff / (2.0 * nu * t)) * exp_term * d_eta;
        }
        return -2.0 * nu * (num / std::max(1e-15, den));
    }

    static std::vector<double> encode_coordinates(double x, double t, int r_enc = 8) {
        std::vector<double> out;
        out.reserve(static_cast<std::size_t>(r_enc));
        const int octaves = std::max(1, r_enc / 4);
        for (int b = 0; b < octaves; ++b) {
            const double freq = std::pow(2.0, b) * kPi;
            out.push_back(std::sin(freq * x));
            out.push_back(std::cos(freq * x));
            out.push_back(std::sin(freq * t));
            out.push_back(std::cos(freq * t));
        }
        if (static_cast<int>(out.size()) > r_enc) {
            out.resize(static_cast<std::size_t>(r_enc));
        }
        while (static_cast<int>(out.size()) < r_enc) {
            out.push_back(0.0);
        }
        return out;
    }

    static void generate_splits(std::vector<DataPoint>& train_interp,
                                std::vector<DataPoint>& test_extrap,
                                int n_train = 2048,
                                int n_test = 1024,
                                int r_enc = 8,
                                std::uint32_t seed = 2026) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist_x(-1.0, 1.0);
        std::uniform_real_distribution<double> dist_t_train(0.0, 0.75);
        std::uniform_real_distribution<double> dist_t_extrap(0.75, 1.0);

        train_interp.clear();
        train_interp.reserve(static_cast<std::size_t>(n_train));
        for (int i = 0; i < n_train; ++i) {
            const double x = dist_x(rng);
            const double t = dist_t_train(rng);
            const double u = cole_hopf_solution(x, t);
            train_interp.push_back(DataPoint{x, t, u, encode_coordinates(x, t, r_enc)});
        }

        test_extrap.clear();
        test_extrap.reserve(static_cast<std::size_t>(n_test));
        for (int i = 0; i < n_test; ++i) {
            const double x = dist_x(rng);
            const double t = dist_t_extrap(rng);
            const double u = cole_hopf_solution(x, t);
            test_extrap.push_back(DataPoint{x, t, u, encode_coordinates(x, t, r_enc)});
        }
    }
};

}  // namespace burger
