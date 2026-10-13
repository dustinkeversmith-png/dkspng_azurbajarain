#pragma once
#include <algorithm>
#include <cmath>
#include <string>

namespace stevejob::models {

struct IActivation {
    virtual ~IActivation() = default;
    virtual std::string name() const = 0;
    virtual double apply(double x) const = 0;
    virtual double derivative(double x) const = 0; // f'(x)
};

struct LinearActivation : public IActivation {
    std::string name() const override { return "Linear"; }
    double apply(double x) const override { return x; }
    double derivative(double /*x*/) const override { return 1.0; }
};

struct ReLUActivation : public IActivation {
    std::string name() const override { return "ReLU"; }
    double apply(double x) const override { return std::max(0.0, x); }
    double derivative(double x) const override { return x > 0.0 ? 1.0 : 0.0; }
};

struct TanhActivation : public IActivation {
    std::string name() const override { return "Tanh"; }
    double apply(double x) const override { return std::tanh(x); }
    double derivative(double x) const override {
        double t = std::tanh(x);
        return 1.0 - t * t;
    }
};

struct SigmoidActivation : public IActivation {
    std::string name() const override { return "Sigmoid"; }
    double apply(double x) const override { return 1.0 / (1.0 + std::exp(-x)); }
    double derivative(double x) const override {
        double s = apply(x);
        return s * (1.0 - s);
    }
};

// Process-lifetime instances. Checkpoint digits stay 0=Linear, 1=ReLU, 2=Tanh, 3=Sigmoid.
struct Activations {
    static const IActivation& linear() {
        static const LinearActivation value;
        return value;
    }
    static const IActivation& relu() {
        static const ReLUActivation value;
        return value;
    }
    static const IActivation& tanh() {
        static const TanhActivation value;
        return value;
    }
    static const IActivation& sigmoid() {
        static const SigmoidActivation value;
        return value;
    }

    static const IActivation& canonical(const IActivation& activation) {
        const std::string label = activation.name();
        if (label == "ReLU") {
            return relu();
        }
        if (label == "Tanh") {
            return tanh();
        }
        if (label == "Sigmoid") {
            return sigmoid();
        }
        return linear();
    }

    static char code(const IActivation& activation) {
        const std::string label = activation.name();
        if (label == "ReLU") {
            return '1';
        }
        if (label == "Tanh") {
            return '2';
        }
        if (label == "Sigmoid") {
            return '3';
        }
        return '0';
    }

    static const IActivation& from_code(char digit, bool last_layer) {
        switch (digit) {
            case '0':
                return linear();
            case '1':
                return relu();
            case '2':
                return tanh();
            case '3':
                return sigmoid();
            default:
                return last_layer ? linear() : relu();
        }
    }
};

} // namespace stevejob::models
