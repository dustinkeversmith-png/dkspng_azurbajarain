#pragma once

#include <iomanip>
#include <sstream>
#include <string>

namespace stevejob::models {

struct EvalScheme {
    int samples = 50;

    static std::string format_loss(double loss) {
        std::ostringstream out;
        out.setf(std::ios::fixed);
        out << std::setprecision(4) << loss;
        return out.str();
    }
};

}  // namespace stevejob::models
