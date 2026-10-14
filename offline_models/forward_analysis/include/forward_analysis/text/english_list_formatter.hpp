#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

struct EnglishListFormatter {
    // {"a"} -> "a", {"a","b"} -> "a and b", {"a","b","c"} -> "a, b and c".
    static std::string join(const std::vector<std::string>& items) {
        std::string out;
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i > 0) {
                out += (i + 1 == items.size()) ? " and " : ", ";
            }
            out += items[i];
        }
        return out;
    }

    static std::string indices(const std::vector<std::size_t>& values, const std::string& prefix = "") {
        std::vector<std::string> items;
        items.reserve(values.size());
        for (std::size_t v : values) {
            items.push_back(prefix + std::to_string(v));
        }
        return join(items);
    }

    static std::string coordinates(const std::vector<std::size_t>& values) {
        return indices(values, "x");
    }
};

}  // namespace forward_analysis
