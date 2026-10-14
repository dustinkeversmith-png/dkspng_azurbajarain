#pragma once

#include "forward_analysis/description/observation_text.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>

namespace forward_analysis {

// Word-overlap F1 between generated and reference observation text (punctuation ignored, case-folded).
struct ObservationSimilarity {
    static double field_f1(const std::string& generated, const std::string& reference) {
        const std::map<std::string, std::size_t> g = word_counts(generated);
        const std::map<std::string, std::size_t> r = word_counts(reference);
        std::size_t g_total = 0;
        std::size_t r_total = 0;
        std::size_t overlap = 0;
        for (const auto& [word, count] : g) {
            g_total += count;
            const auto it = r.find(word);
            if (it != r.end()) {
                overlap += std::min(count, it->second);
            }
        }
        for (const auto& entry : r) {
            r_total += entry.second;
        }
        if (g_total == 0 || r_total == 0 || overlap == 0) {
            return 0.0;
        }
        const double precision = static_cast<double>(overlap) / static_cast<double>(g_total);
        const double recall = static_cast<double>(overlap) / static_cast<double>(r_total);
        return 2.0 * precision * recall / (precision + recall);
    }

    static double mean_f1(const ObservationText& generated, const ObservationText& reference) {
        double sum = 0.0;
        for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
            sum += field_f1(generated.field(f), reference.field(f));
        }
        return sum / static_cast<double>(ObservationText::kFieldCount);
    }

private:
    static std::map<std::string, std::size_t> word_counts(const std::string& text) {
        std::map<std::string, std::size_t> counts;
        for (const std::string& lex : LexemeSplitter::split(LexemeSplitter::lowercase(text))) {
            if (lex.size() == 1 && LexemeSplitter::is_punctuation(lex[0])) {
                continue;
            }
            ++counts[lex];
        }
        return counts;
    }
};

}  // namespace forward_analysis
