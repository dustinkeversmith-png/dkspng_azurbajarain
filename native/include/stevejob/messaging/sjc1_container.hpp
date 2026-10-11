#pragma once

#include "stevejob/messaging/compressor.hpp"

#include <string>

namespace stevejob::messaging {

class Sjc1Container {
public:
    static std::string pack(const ModelData& model) { return TensorCompressor::pack(model); }

    static ModelData unpack(const std::string& bytes) { return TensorCompressor::unpack(bytes); }
};

}  // namespace stevejob::messaging
