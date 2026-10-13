#pragma once

#include "stevejob/messaging/scheme_descriptor.hpp"
#include "stevejob/models/neural_network.hpp"

#include <string>

namespace stevejob::messaging {

class IContainerScheme {
public:
    virtual ~IContainerScheme() = default;
    virtual EncodingDescriptor get_descriptor() const = 0;
    virtual std::string pack(const stevejob::models::NeuralNetwork& model) const = 0;
    virtual stevejob::models::NeuralNetwork unpack(const std::string& bytes) const = 0;
};

}  // namespace stevejob::messaging
