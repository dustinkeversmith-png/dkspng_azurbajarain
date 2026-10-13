#pragma once

#include "stevejob/models/candidate_scheme.hpp"

#include <sstream>
#include <string>

namespace stevejob::models {

class TournamentPrompt {
public:
    static std::string schema(const CandidateScheme& scheme, const std::string& reasoning) {
        std::ostringstream out;
        out << "Output valid JSON ONLY:\n{\n";
        out << "  \"reasoning\": \"<" << reasoning << ">\",\n";
        out << "  \"confidence\": <float 0.1 to 1.0>";
        for (const std::string& slot : scheme.slots) {
            out << ",\n  \"" << slot << "\": {";
            for (std::size_t index = 0; index < scheme.fields.size(); ++index) {
                if (index > 0) {
                    out << ", ";
                }
                out << "\"" << scheme.fields[index].key << "\": \"...\"";
            }
            out << "}";
        }
        out << "\n}";
        return out.str();
    }
};

}  // namespace stevejob::models
