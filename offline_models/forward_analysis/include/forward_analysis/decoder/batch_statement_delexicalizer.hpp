#pragma once

#include "forward_analysis/description/batch_statement_checker.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Replaces every number of a batch-statistic statement with the digit slot of the quantity at that position:
//   "Across 14 observations, the hidden widths are 27 and 14. Layer 1 has 0 coordinates that remain zero ..."
//   -> "Across <COUNT:N|d> observations , the hidden widths are <COUNT:W_1|d> and <COUNT:W_2|d> . Layer 1 has
//       <COUNT:ZERO_L1|d> coordinates that remain zero ..."
// Layer numbers ("Layer 1") stay literal; a slot for a layer the network lacks is unfillable at generation time.
// The slots are filled by ObservationDelexicalizer::relexicalize from ShapeSlotLexicon counts.
struct BatchStatementDelexicalizer {
    static std::vector<std::string> delexicalize(const std::string& text) {
        const std::vector<std::string> words = LexemeSplitter::split(text);
        std::vector<std::string> lower;
        for (const std::string& w : words) {
            lower.push_back(LexemeSplitter::lowercase(w));
        }
        std::vector<std::string> out;
        std::size_t layer_context = 0;
        for (std::size_t i = 0; i < words.size();) {
            std::vector<std::size_t> values;
            std::vector<std::size_t> at;
            std::vector<std::string> names;
            std::size_t end = i;
            if (BatchStatementChecker::match(lower, i, {"across", "#", "observations"}, values, at)) {
                names = {"N"};
            } else if (BatchStatementChecker::match(lower, i, {"hidden", "widths", "are", "#*"}, values, at)) {
                for (std::size_t k = 0; k < at.size(); ++k) {
                    names.push_back("W_" + std::to_string(k + 1));
                }
            } else if (BatchStatementChecker::match(lower, i, {"layer", "#", "has", "#", "coordinates"}, values, at)) {
                layer_context = values[0];
                names = {"", "ZERO_L" + std::to_string(values[0])};
            } else if (layer_context > 0 &&
                       BatchStatementChecker::match(lower, i, {"#", "observations", "with", "no", "positive"}, values, at)) {
                names = {"NONPOS_L" + std::to_string(layer_context)};
            } else if (BatchStatementChecker::match(lower, i, {"input", "coordinates", "#", "and", "#", "have", "the", "strongest"},
                                                    values, at)) {
                names = {"STRONG_1", "STRONG_2"};
            } else if (BatchStatementChecker::match(lower, i, {"coordinates", "#", "and", "#", "have", "the", "weakest"}, values, at)) {
                names = {"WEAK_1", "WEAK_2"};
            } else if (BatchStatementChecker::match(lower, i, {"numerical", "ranks", "#*"}, values, at)) {
                for (std::size_t k = 0; k < at.size(); ++k) {
                    names.push_back("RANK_L" + std::to_string(k + 1));
                }
            } else if (BatchStatementChecker::match(lower, i, {"over", "#", "output", "coordinates"}, values, at)) {
                names = {"D_OUT"};
            } else if (BatchStatementChecker::match(lower, i, {"#", "have", "a", "smaller"}, values, at)) {
                names = {"SMALLER_RANGE"};
            } else if (BatchStatementChecker::match(lower, i, {"#", "have", "a", "larger"}, values, at)) {
                names = {"LARGER_RANGE"};
            } else if (BatchStatementChecker::match(lower, i, {"#", "have", "broadly", "comparable"}, values, at)) {
                names = {"COMPARABLE_RANGE"};
            } else if (BatchStatementChecker::match(lower, i, {"agree", "in", "#", "of", "#", "entries"}, values, at)) {
                names = {"SIGN_AGREE", "SIGN_ENTRIES"};
            }
            if (names.empty()) {
                out.push_back(words[i]);
                ++i;
                continue;
            }
            end = at.back() + 1;
            std::size_t slot = 0;
            for (std::size_t k = i; k < end; ++k) {
                if (slot < at.size() && at[slot] == k) {
                    out.push_back(names[slot].empty() ? words[k] : "<COUNT:" + names[slot] + "|d>");
                    ++slot;
                } else {
                    out.push_back(words[k]);
                }
            }
            i = end;
        }
        return out;
    }
};

}  // namespace forward_analysis
