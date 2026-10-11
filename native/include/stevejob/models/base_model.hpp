#pragma once

#include <vector>

namespace stevejob::models {

class BaseModel {
public:
    virtual ~BaseModel() = default;
    virtual std::vector<std::vector<double>> forward(
        const std::vector<std::vector<double>>& inputs) const = 0;
};

}  // namespace stevejob::models
