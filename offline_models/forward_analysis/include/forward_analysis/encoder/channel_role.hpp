#pragma once

#include <cstddef>

namespace forward_analysis {

// What a single telemetry column is: one token is produced per input coordinate, hidden unit,
// and per output for its target, prediction and residual.
enum class ChannelRole : std::size_t {
    Input = 0,
    Hidden = 1,
    Target = 2,
    Prediction = 3,
    Residual = 4,
};

struct ChannelRoleNames {
    // Number of distinct roles (width of the role one-hot block).
    static constexpr std::size_t kCount = 5;

    static const char* name(ChannelRole role) noexcept {
        switch (role) {
            case ChannelRole::Input: return "input";
            case ChannelRole::Hidden: return "hidden";
            case ChannelRole::Target: return "target";
            case ChannelRole::Prediction: return "prediction";
            case ChannelRole::Residual: return "residual";
        }
        return "unknown";
    }
};

}  // namespace forward_analysis
