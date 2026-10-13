#pragma once

#include <string>

namespace stevejob::storage {

struct BlobPath {
    static std::string bare(std::string name, const char* prefix) {
        const std::string marker(prefix);
        if (name.rfind(marker, 0) == 0) {
            name = name.substr(marker.size());
        }
        return name;
    }
};

}  // namespace stevejob::storage
