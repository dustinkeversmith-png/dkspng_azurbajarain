#pragma once

#include "forward_analysis/config/distiller_config.hpp"
#include "forward_analysis/encoder/channel_feature_extractor.hpp"
#include "forward_analysis/encoder/linear_input_adapter.hpp"
#include "forward_analysis/telemetry/forward_telemetry_tensor.hpp"

#include <cstdint>

namespace forward_analysis {

// Channel tokens (C x F) projected by the linear adapter into the encoder's input sequence (C x d_model).
class ChannelSequenceFormatter {
public:
    explicit ChannelSequenceFormatter(const DistillerConfig& config = DistillerConfig{},
                                      std::uint64_t seed = 1337ULL)
        : config_(config),
          adapter_(config.d_model, ChannelFeatureExtractor::kFeatureDim, seed) {
        config_.validate();
    }

    Eigen::MatrixXd channel_features(const ForwardTelemetryTensor& telemetry) const {
        return ChannelFeatureExtractor::extract(telemetry, config_);
    }

    Eigen::MatrixXd format_sequence(const Eigen::MatrixXd& features) {
        return adapter_.project(features);
    }

    Eigen::MatrixXd format_sequence(const ForwardTelemetryTensor& telemetry) {
        return format_sequence(channel_features(telemetry));
    }

    void update_adapter(const Eigen::MatrixXd& features, const Eigen::MatrixXd& d_sequence, double learning_rate) {
        adapter_.update_from_sequence_grad(features, d_sequence, learning_rate);
    }

    const LinearInputAdapter& adapter() const noexcept { return adapter_; }

private:
    // Thresholds used by the feature extractor and the model width of the adapter.
    DistillerConfig config_;
    // Learned projection from channel features (F) to d_model.
    LinearInputAdapter adapter_;
};

}  // namespace forward_analysis
