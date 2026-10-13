#pragma once

#include <string>

namespace stevejob::messaging {

struct EncodingDescriptor {
    std::string scheme_name;
    std::string format_description;
    std::string equation;
    std::string actionable_rules;
};

}  // namespace stevejob::messaging
