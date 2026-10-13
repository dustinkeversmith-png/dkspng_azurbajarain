#pragma once

#include <string>
#include <utility>
#include <vector>

namespace stevejob::models {

struct CandidateField {
    std::string key;
    std::string role;
};

struct CandidateScheme {
    std::vector<std::string> slots{"c1", "c2", "c3"};
    std::vector<CandidateField> fields{{"w", "weights"}, {"b", "bias"}};
    std::vector<std::string> groups{"candidates", "tournament", "hypotheses", "proposals", "result", "output"};
};

}  // namespace stevejob::models
