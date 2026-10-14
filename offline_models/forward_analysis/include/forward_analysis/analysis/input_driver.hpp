#pragma once

#include <cstddef>

namespace forward_analysis {

struct InputDriver {
    // Input coordinate index (x<coordinate>).
    std::size_t coordinate = 0;
    // The first-layer principal axis it correlates with more strongly: 0 = leading axis, 1 = secondary axis.
    std::size_t axis = 0;
    // +1 when that axis score rises with the coordinate, -1 when it falls.
    int direction = 1;
    // Multiple correlation with the two leading axis scores, sqrt(r1^2 + r2^2), in [0, 1].
    double strength = 0.0;
};

}  // namespace forward_analysis
