#pragma once

#include "burger_dataset.hpp"
#include "burger_network.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace burger {

struct ThresholdHit {
    double threshold = 0.0;
    int epoch = -1;          // 1-indexed epoch when train MSE first crossed below threshold; -1 = never
    double wall_seconds = -1.0;
};

struct BenchmarkArtifact {
    std::string experiment_name = "burger_base_adam_baseline";
    int parameter_count = 0;
    double train_mse_initial = 0.0;
    double train_mse_final = 0.0;
    double train_mse_best = 0.0;
    double extrap_mse_final = 0.0;
    double extrap_mse_at_best = 0.0;
    double mean_pde_residual = 0.0;
    double elapsed_wall_seconds = 0.0;
    double mean_epoch_seconds = 0.0;
    int epochs_completed = 0;
    int best_epoch = 0;
    bool converged = false;
    std::string stop_reason = "max_epochs";
    std::vector<double> loss_history;
    std::vector<double> epoch_seconds;
    std::vector<double> extrap_history;  // sampled each epoch (may be empty if disabled)
    std::vector<ThresholdHit> threshold_hits;

    std::string to_json() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(8);
        ss << "{\n"
           << "  \"experiment_name\": \"" << experiment_name << "\",\n"
           << "  \"parameter_count\": " << parameter_count << ",\n"
           << "  \"train_mse_initial\": " << train_mse_initial << ",\n"
           << "  \"train_mse_final\": " << train_mse_final << ",\n"
           << "  \"train_mse_best\": " << train_mse_best << ",\n"
           << "  \"extrap_mse_final\": " << extrap_mse_final << ",\n"
           << "  \"extrap_mse_at_best\": " << extrap_mse_at_best << ",\n"
           << "  \"mean_pde_residual\": " << mean_pde_residual << ",\n"
           << "  \"elapsed_wall_seconds\": " << elapsed_wall_seconds << ",\n"
           << "  \"mean_epoch_seconds\": " << mean_epoch_seconds << ",\n"
           << "  \"epochs_completed\": " << epochs_completed << ",\n"
           << "  \"best_epoch\": " << best_epoch << ",\n"
           << "  \"converged\": " << (converged ? "true" : "false") << ",\n"
           << "  \"stop_reason\": \"" << stop_reason << "\",\n"
           << "  \"threshold_hits\": [";
        for (std::size_t i = 0; i < threshold_hits.size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << "{\"threshold\": " << threshold_hits[i].threshold
               << ", \"epoch\": " << threshold_hits[i].epoch
               << ", \"wall_seconds\": " << threshold_hits[i].wall_seconds << "}";
        }
        ss << "],\n  \"loss_history\": [";
        for (std::size_t i = 0; i < loss_history.size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << loss_history[i];
        }
        ss << "],\n  \"epoch_seconds\": [";
        for (std::size_t i = 0; i < epoch_seconds.size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << epoch_seconds[i];
        }
        ss << "]\n}\n";
        return ss.str();
    }

    std::string trajectory_csv() const {
        std::ostringstream ss;
        ss << "epoch,train_mse,epoch_seconds,cumulative_seconds";
        if (!extrap_history.empty()) {
            ss << ",extrap_mse";
        }
        ss << "\n";
        ss << std::fixed << std::setprecision(8);
        double cum = 0.0;
        for (std::size_t i = 0; i < loss_history.size(); ++i) {
            const double ep_s = i < epoch_seconds.size() ? epoch_seconds[i] : 0.0;
            cum += ep_s;
            ss << (i + 1) << "," << loss_history[i] << "," << ep_s << "," << cum;
            if (!extrap_history.empty()) {
                ss << "," << (i < extrap_history.size() ? extrap_history[i] : -1.0);
            }
            ss << "\n";
        }
        return ss.str();
    }
};

class BurgerBenchmarkRunner {
public:
    static BenchmarkArtifact run_standard_training(BurgerBaseNetwork& net,
                                                   const std::vector<DataPoint>& train_set,
                                                   const std::vector<DataPoint>& test_set,
                                                   int epochs = 1000,
                                                   double learning_rate = 0.001,
                                                   int batch_size = 64,
                                                   std::uint32_t seed = 7,
                                                   int patience = 50,
                                                   double min_delta = 1e-4,
                                                   bool track_extrap_each_epoch = true) {
        BenchmarkArtifact artifact;
        artifact.parameter_count = net.total_parameters();

        std::vector<std::vector<double>> x_train;
        std::vector<double> y_train;
        std::vector<std::vector<double>> x_test;
        std::vector<double> y_test;
        x_train.reserve(train_set.size());
        y_train.reserve(train_set.size());
        for (const auto& d : train_set) {
            x_train.push_back(d.encoded_input);
            y_train.push_back(d.u_true);
        }
        for (const auto& d : test_set) {
            x_test.push_back(d.encoded_input);
            y_test.push_back(d.u_true);
        }

        const std::vector<double> thresholds = {1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01};
        artifact.threshold_hits.reserve(thresholds.size());
        for (double thr : thresholds) {
            artifact.threshold_hits.push_back(ThresholdHit{thr, -1, -1.0});
        }

        artifact.train_mse_initial = net.compute_mse(x_train, y_train);
        artifact.train_mse_best = artifact.train_mse_initial;
        artifact.best_epoch = 0;

        std::vector<std::vector<std::vector<double>>> m_w(net.layers.size());
        std::vector<std::vector<std::vector<double>>> v_w(net.layers.size());
        std::vector<std::vector<double>> m_b(net.layers.size());
        std::vector<std::vector<double>> v_b(net.layers.size());
        for (std::size_t l = 0; l < net.layers.size(); ++l) {
            const Layer& layer = net.layers[l];
            m_w[l].assign(static_cast<std::size_t>(layer.in_dim),
                          std::vector<double>(static_cast<std::size_t>(layer.out_dim), 0.0));
            v_w[l].assign(static_cast<std::size_t>(layer.in_dim),
                          std::vector<double>(static_cast<std::size_t>(layer.out_dim), 0.0));
            m_b[l].assign(static_cast<std::size_t>(layer.out_dim), 0.0);
            v_b[l].assign(static_cast<std::size_t>(layer.out_dim), 0.0);
        }

        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double eps = 1e-8;
        std::mt19937 rng(seed);
        std::vector<std::size_t> order(x_train.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }

        auto t0 = std::chrono::steady_clock::now();
        int step = 0;
        int stall = 0;
        BurgerBaseNetwork best_net = net;

        for (int ep = 0; ep < epochs; ++ep) {
            auto ep0 = std::chrono::steady_clock::now();
            std::shuffle(order.begin(), order.end(), rng);
            for (std::size_t start = 0; start < order.size(); start += static_cast<std::size_t>(batch_size)) {
                const std::size_t end = std::min(order.size(), start + static_cast<std::size_t>(batch_size));
                const double inv_bs = 1.0 / static_cast<double>(end - start);

                std::vector<std::vector<std::vector<double>>> gw = m_w;
                std::vector<std::vector<double>> gb = m_b;
                for (auto& row : gw) {
                    for (auto& col : row) {
                        std::fill(col.begin(), col.end(), 0.0);
                    }
                }
                for (auto& b : gb) {
                    std::fill(b.begin(), b.end(), 0.0);
                }

                for (std::size_t ii = start; ii < end; ++ii) {
                    const std::size_t idx = order[ii];
                    net.backward_mse(x_train[idx], y_train[idx], gw, gb);
                }
                for (std::size_t l = 0; l < net.layers.size(); ++l) {
                    for (auto& col : gw[l]) {
                        for (double& g : col) {
                            g *= inv_bs;
                        }
                    }
                    for (double& g : gb[l]) {
                        g *= inv_bs;
                    }
                }

                ++step;
                const double bc1 = 1.0 - std::pow(beta1, step);
                const double bc2 = 1.0 - std::pow(beta2, step);
                for (std::size_t l = 0; l < net.layers.size(); ++l) {
                    Layer& layer = net.layers[l];
                    for (int r = 0; r < layer.in_dim; ++r) {
                        for (int c = 0; c < layer.out_dim; ++c) {
                            const double g = gw[l][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            double& mw = m_w[l][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            double& vw = v_w[l][static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
                            mw = beta1 * mw + (1.0 - beta1) * g;
                            vw = beta2 * vw + (1.0 - beta2) * g * g;
                            const double mhat = mw / bc1;
                            const double vhat = vw / bc2;
                            layer.weights[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] -=
                                learning_rate * mhat / (std::sqrt(vhat) + eps);
                        }
                    }
                    for (int c = 0; c < layer.out_dim; ++c) {
                        const double g = gb[l][static_cast<std::size_t>(c)];
                        double& mb = m_b[l][static_cast<std::size_t>(c)];
                        double& vb = v_b[l][static_cast<std::size_t>(c)];
                        mb = beta1 * mb + (1.0 - beta1) * g;
                        vb = beta2 * vb + (1.0 - beta2) * g * g;
                        const double mhat = mb / bc1;
                        const double vhat = vb / bc2;
                        layer.biases[static_cast<std::size_t>(c)] -= learning_rate * mhat / (std::sqrt(vhat) + eps);
                    }
                }
            }

            auto ep1 = std::chrono::steady_clock::now();
            const double ep_s = std::chrono::duration<double>(ep1 - ep0).count();
            artifact.epoch_seconds.push_back(ep_s);

            const double curr_loss = net.compute_mse(x_train, y_train);
            artifact.loss_history.push_back(curr_loss);
            if (track_extrap_each_epoch) {
                artifact.extrap_history.push_back(net.compute_mse(x_test, y_test));
            }

            const double wall_so_far = std::chrono::duration<double>(ep1 - t0).count();
            for (ThresholdHit& hit : artifact.threshold_hits) {
                if (hit.epoch < 0 && curr_loss <= hit.threshold) {
                    hit.epoch = ep + 1;
                    hit.wall_seconds = wall_so_far;
                }
            }

            if (curr_loss + min_delta < artifact.train_mse_best) {
                artifact.train_mse_best = curr_loss;
                artifact.best_epoch = ep + 1;
                best_net = net;
                stall = 0;
            } else {
                ++stall;
            }

            if (curr_loss <= 0.01) {
                artifact.converged = true;
                artifact.stop_reason = "reached_train_mse_0.01";
                artifact.epochs_completed = ep + 1;
                break;
            }
            if (stall >= patience) {
                artifact.converged = stall >= patience && artifact.train_mse_best < artifact.train_mse_initial * 0.5;
                artifact.stop_reason = "early_stop_patience";
                artifact.epochs_completed = ep + 1;
                break;
            }
            artifact.epochs_completed = ep + 1;
        }

        if (artifact.stop_reason == "max_epochs" && artifact.epochs_completed >= epochs) {
            artifact.stop_reason = "max_epochs";
            artifact.converged = artifact.train_mse_best <= 0.05;
        }

        net = best_net;
        auto t1 = std::chrono::steady_clock::now();
        artifact.elapsed_wall_seconds = std::chrono::duration<double>(t1 - t0).count();
        double ep_sum = 0.0;
        for (double s : artifact.epoch_seconds) {
            ep_sum += s;
        }
        artifact.mean_epoch_seconds =
            artifact.epoch_seconds.empty() ? 0.0 : ep_sum / static_cast<double>(artifact.epoch_seconds.size());

        artifact.train_mse_final = net.compute_mse(x_train, y_train);
        artifact.extrap_mse_final = net.compute_mse(x_test, y_test);
        artifact.extrap_mse_at_best = artifact.extrap_mse_final;
        if (!artifact.extrap_history.empty() && artifact.best_epoch > 0 &&
            static_cast<std::size_t>(artifact.best_epoch) <= artifact.extrap_history.size()) {
            artifact.extrap_mse_at_best = artifact.extrap_history[static_cast<std::size_t>(artifact.best_epoch - 1)];
        }

        double res_accum = 0.0;
        const int grid_points = 20;
        for (int i = 0; i < grid_points; ++i) {
            const double px = -0.8 + 1.6 * (static_cast<double>(i) / static_cast<double>(grid_points));
            const double pt = 0.1 + 0.6 * (static_cast<double>(i) / static_cast<double>(grid_points));
            res_accum += net.compute_pde_residual(px, pt);
        }
        artifact.mean_pde_residual = res_accum / static_cast<double>(grid_points);
        return artifact;
    }

    static void write_artifacts(const BenchmarkArtifact& artifact,
                                const std::string& json_path,
                                const std::string& csv_path) {
        {
            std::ofstream out(json_path);
            out << artifact.to_json();
        }
        {
            std::ofstream out(csv_path);
            out << artifact.trajectory_csv();
        }
    }
};

}  // namespace burger
