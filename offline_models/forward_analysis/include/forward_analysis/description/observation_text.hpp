#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace forward_analysis {

// The four qualitative observation fields a sample's ground truth (and the model's output) consists of.
struct ObservationText {
    static constexpr std::size_t kFieldCount = 4;

    // Which units carry signal, which are silent or gated, and how routing changes across samples.
    std::string pathway_activity;
    // Which input coordinates drive the hidden activity and which are ignored.
    std::string feature_sensitivity;
    // Geometry of the hidden representations: rays, sheets, clusters, collapse.
    std::string representational_structure;
    // How predictions track, damp, flatten or distort the targets.
    std::string functional_mapping;

    static const std::array<std::string, kFieldCount>& field_names() {
        static const std::array<std::string, kFieldCount> names = {
            "pathway_activity", "feature_sensitivity", "representational_structure", "functional_mapping"};
        return names;
    }

    std::string& field(std::size_t index) {
        switch (index) {
            case 0: return pathway_activity;
            case 1: return feature_sensitivity;
            case 2: return representational_structure;
            case 3: return functional_mapping;
            default: throw std::out_of_range("ObservationText field index must be < 4");
        }
    }

    const std::string& field(std::size_t index) const {
        return const_cast<ObservationText*>(this)->field(index);
    }

    nlohmann::json to_json() const {
        nlohmann::json out = nlohmann::json::object();
        for (std::size_t f = 0; f < kFieldCount; ++f) {
            out[field_names()[f]] = field(f);
        }
        return out;
    }

    static ObservationText from_json(const nlohmann::json& node) {
        if (!node.is_object()) {
            throw std::invalid_argument("target_text must be an object with the four observation fields");
        }
        ObservationText out;
        for (std::size_t f = 0; f < kFieldCount; ++f) {
            const std::string& name = field_names()[f];
            if (!node.contains(name) || !node.at(name).is_string() || node.at(name).get<std::string>().empty()) {
                throw std::invalid_argument("target_text." + name + " must be a non-empty string");
            }
            out.field(f) = node.at(name).get<std::string>();
        }
        return out;
    }
};

}  // namespace forward_analysis
