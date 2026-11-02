#pragma once

#include "burger_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace burger {

struct Layer {
    std::string name;
    int in_dim = 0;
    int out_dim = 0;
    std::vector<std::vector<double>> weights;  // in_dim x out_dim
    std::vector<double> biases;

    Layer(std::string n, int in_d, int out_d) : name(std::move(n)), in_dim(in_d), out_dim(out_d) {
        weights.assign(static_cast<std::size_t>(in_dim), std::vector<double>(static_cast<std::size_t>(out_dim), 0.0));
        biases.assign(static_cast<std::size_t>(out_dim), 0.0);
    }

    void init_xavier(std::mt19937& rng) {
        const double limit = std::sqrt(6.0 / static_cast<double>(in_dim + out_dim));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (int r = 0; r < in_dim; ++r) {
            for (int c = 0; c < out_dim; ++c) {
                weights[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = dist(rng);
            }
        }
    }
};

class BurgerBaseNetwork {
public:
    std::vector<Layer> layers;

    // Base tier default topology: [8 -> 68 -> 68 -> 68 -> 1] (~10,133 parameters)
    explicit BurgerBaseNetwork(int input_dim = 8, int width = 68, std::uint32_t seed = 42) {
        layers.emplace_back("l1", input_dim, width);
        layers.emplace_back("l2", width, width);
        layers.emplace_back("l3", width, width);
        layers.emplace_back("head", width, 1);
        std::mt19937 rng(seed);
        for (auto& layer : layers) {
            layer.init_xavier(rng);
        }
    }

    static double silu(double x) {
        return x / (1.0 + std::exp(-std::clamp(x, -50.0, 50.0)));
    }

    static double silu_derivative(double x) {
        const double s = 1.0 / (1.0 + std::exp(-std::clamp(x, -50.0, 50.0)));
        return s * (1.0 + x * (1.0 - s));
    }

    double forward(const std::vector<double>& input) const {
        std::vector<double> curr = input;
        for (std::size_t l = 0; l < layers.size(); ++l) {
            const Layer& lay = layers[l];
            std::vector<double> next(static_cast<std::size_t>(lay.out_dim), 0.0);
            for (int c = 0; c < lay.out_dim; ++c) {
                double val = lay.biases[static_cast<std::size_t>(c)];
                for (int r = 0; r < lay.in_dim; ++r) {
                    val += curr[static_cast<std::size_t>(r)] * lay.weights[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                }
                next[static_cast<std::size_t>(c)] = (l + 1 == layers.size()) ? val : silu(val);
            }
            curr = std::move(next);
        }
        return curr.empty() ? 0.0 : curr[0];
    }

    double compute_mse(const std::vector<std::vector<double>>& inputs, const std::vector<double>& targets) const {
        double total_err = 0.0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const double d = forward(inputs[i]) - targets[i];
            total_err += d * d;
        }
        return total_err / static_cast<double>(std::max<std::size_t>(1, inputs.size()));
    }

    // Analytical backprop for supervised MSE (used by Adam/SGD baseline).
    void backward_mse(const std::vector<double>& input,
                      double target,
                      std::vector<std::vector<std::vector<double>>>& grad_w,
                      std::vector<std::vector<double>>& grad_b) const {
        const std::size_t L = layers.size();
        std::vector<std::vector<double>> pre(L);
        std::vector<std::vector<double>> act(L);
        std::vector<double> curr = input;
        for (std::size_t l = 0; l < L; ++l) {
            const Layer& lay = layers[l];
            pre[l].assign(static_cast<std::size_t>(lay.out_dim), 0.0);
            act[l].assign(static_cast<std::size_t>(lay.out_dim), 0.0);
            for (int c = 0; c < lay.out_dim; ++c) {
                double val = lay.biases[static_cast<std::size_t>(c)];
                for (int r = 0; r < lay.in_dim; ++r) {
                    val += curr[static_cast<std::size_t>(r)] *
                           lay.weights[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                }
                pre[l][static_cast<std::size_t>(c)] = val;
                act[l][static_cast<std::size_t>(c)] = (l + 1 == L) ? val : silu(val);
            }
            curr = act[l];
        }

        std::vector<double> delta = act.back();
        delta[0] = 2.0 * (act.back()[0] - target);  // d(MSE)/dy for single sample contribution later averaged

        for (int l = static_cast<int>(L) - 1; l >= 0; --l) {
            const Layer& lay = layers[static_cast<std::size_t>(l)];
            const std::vector<double>& prev_act =
                (l == 0) ? input : act[static_cast<std::size_t>(l - 1)];
            if (l + 1 != static_cast<int>(L)) {
                for (int c = 0; c < lay.out_dim; ++c) {
                    delta[static_cast<std::size_t>(c)] *= silu_derivative(pre[static_cast<std::size_t>(l)][static_cast<std::size_t>(c)]);
                }
            }
            for (int c = 0; c < lay.out_dim; ++c) {
                grad_b[static_cast<std::size_t>(l)][static_cast<std::size_t>(c)] += delta[static_cast<std::size_t>(c)];
                for (int r = 0; r < lay.in_dim; ++r) {
                    grad_w[static_cast<std::size_t>(l)][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] +=
                        prev_act[static_cast<std::size_t>(r)] * delta[static_cast<std::size_t>(c)];
                }
            }
            if (l == 0) {
                break;
            }
            std::vector<double> prev_delta(static_cast<std::size_t>(lay.in_dim), 0.0);
            for (int r = 0; r < lay.in_dim; ++r) {
                double sum = 0.0;
                for (int c = 0; c < lay.out_dim; ++c) {
                    sum += lay.weights[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] *
                           delta[static_cast<std::size_t>(c)];
                }
                prev_delta[static_cast<std::size_t>(r)] = sum;
            }
            delta = std::move(prev_delta);
        }
    }

    double compute_pde_residual(double x, double t, int r_enc = 8, double nu = 0.01 / BurgerDatasetGenerator::kPi) const {
        const double h = 1e-4;
        auto enc = [&](double px, double pt) { return BurgerDatasetGenerator::encode_coordinates(px, pt, r_enc); };
        const double u = forward(enc(x, t));
        const double u_t = (forward(enc(x, t + h)) - forward(enc(x, t - h))) / (2.0 * h);
        const double u_x = (forward(enc(x + h, t)) - forward(enc(x - h, t))) / (2.0 * h);
        const double u_xx = (forward(enc(x + h, t)) - 2.0 * u + forward(enc(x - h, t))) / (h * h);
        const double res = u_t + u * u_x - nu * u_xx;
        return res * res;
    }

    int total_parameters() const {
        int total = 0;
        for (const auto& layer : layers) {
            total += layer.in_dim * layer.out_dim + layer.out_dim;
        }
        return total;
    }
};

}  // namespace burger
