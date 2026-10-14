#pragma once

#include "forward_analysis/decoder/shape_slot_lexicon.hpp"
#include "forward_analysis/description/number_words.hpp"
#include "forward_analysis/text/english_list_formatter.hpp"
#include "forward_analysis/text/lexeme_splitter.hpp"

#include <array>
#include <cctype>
#include <cstddef>
#include <set>
#include <string>
#include <vector>

namespace forward_analysis {

// Replaces shape and index references in observation text with slot tokens bound to measured quantities,
// and fills slot tokens back in from another network's measurements:
//   "Seven of the twenty-four first-layer units (units 1, 5, 9, 12, 16, 20 and 23) never fire"
//   -> "<COUNT:DEAD_L1|W> of the <DIM:W_1|w> <ORD:LAYER_1|wl> units ( units <LIST:DEAD_L1> ) never fire"
// Forms: |w lowercase word, |W capitalized word, |d digits, |wl / |Wl ordinal + "-layer".
struct ObservationDelexicalizer {
    static bool is_slot(const std::string& lexeme) {
        return lexeme.size() > 2 && lexeme.front() == '<' && lexeme.back() == '>' && lexeme.find(':') != std::string::npos;
    }

    static std::vector<std::string> delexicalize(const std::string& text, const ShapeSlotLexicon& lex) {
        const std::vector<std::string> words = LexemeSplitter::split(text);
        std::vector<std::string> lower;
        for (const std::string& w : words) {
            lower.push_back(LexemeSplitter::lowercase(w));
        }
        std::vector<std::string> out;
        for (std::size_t i = 0; i < words.size();) {
            std::size_t consumed = 0;
            const std::string list = list_slot(words, i, lex, consumed);
            if (!list.empty()) {
                out.push_back(list);
                i += consumed;
                continue;
            }
            out.push_back(single_slot(words, lower, i, lex));
            ++i;
        }
        return out;
    }

    // Fills every slot from lex; false when a slot names a quantity this network does not have.
    static bool relexicalize(const std::vector<std::string>& lexemes, const ShapeSlotLexicon& lex, std::string& text) {
        std::vector<std::string> filled;
        for (const std::string& lexeme : lexemes) {
            if (!is_slot(lexeme)) {
                filled.push_back(lexeme);
                continue;
            }
            std::string value;
            if (!fill(lexeme, lex, value)) {
                return false;
            }
            filled.push_back(value);
        }
        text = LexemeSplitter::join(filled);
        return true;
    }

private:
    static const std::array<std::string, 4>& ordinals() {
        static const std::array<std::string, 4> words = {"first", "second", "third", "fourth"};
        return words;
    }

    static bool capitalized(const std::string& word) {
        return !word.empty() && std::isupper(static_cast<unsigned char>(word[0])) != 0;
    }

    static bool digits(const std::string& word) {
        if (word.empty()) {
            return false;
        }
        for (char c : word) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return true;
    }

    static std::string form_of(const std::string& word) {
        return digits(word) ? "d" : (capitalized(word) ? "W" : "w");
    }

    static std::string slot(const std::string& kind, const std::string& name, const std::string& form = "") {
        return "<" + kind + ":" + name + (form.empty() ? "" : "|" + form) + ">";
    }

    // A run of indices written exactly as EnglishListFormatter writes one of the measured lists.
    static std::string list_slot(const std::vector<std::string>& words,
                                 std::size_t i,
                                 const ShapeSlotLexicon& lex,
                                 std::size_t& consumed) {
        if (!digits(words[i])) {
            return "";
        }
        static const std::set<std::string> single_cues = {"unit", "units", "sample", "samples"};
        const bool cued = i > 0 && single_cues.count(LexemeSplitter::lowercase(words[i - 1])) > 0;
        std::string best;
        for (const auto& [name, values] : lex.lists) {
            if (values.empty() || (values.size() == 1 && !cued)) {
                continue;
            }
            const std::vector<std::string> canonical = LexemeSplitter::split(EnglishListFormatter::indices(values));
            if (i + canonical.size() > words.size() || canonical.size() <= consumed) {
                continue;
            }
            bool match = true;
            for (std::size_t k = 0; k < canonical.size() && match; ++k) {
                match = words[i + k] == canonical[k];
            }
            const bool continues = i + canonical.size() < words.size() &&
                                   (words[i + canonical.size()] == "," || words[i + canonical.size()] == "and") &&
                                   i + canonical.size() + 1 < words.size() && digits(words[i + canonical.size() + 1]);
            if (match && !continues) {
                consumed = canonical.size();
                best = slot("LIST", name);
            }
        }
        return best;
    }

    static std::string noun_group(const std::vector<std::string>& lower, std::size_t i) {
        static const std::set<std::string> units = {"units", "unit", "first-layer", "second-layer", "hidden", "neurons"};
        static const std::set<std::string> samples = {"samples", "sample"};
        static const std::set<std::string> outputs = {"outputs"};
        static const std::set<std::string> coordinates = {"coordinates", "inputs"};
        for (std::size_t k = i + 1; k <= i + 2 && k < lower.size(); ++k) {
            if (k == i + 2 && (lower[i + 1] == "of" || LexemeSplitter::is_punctuation(lower[i + 1][0]))) {
                break;
            }
            const std::string& w = lower[k];
            if (units.count(w)) return "units";
            if (samples.count(w)) return "samples";
            if (outputs.count(w)) return "outputs";
            if (coordinates.count(w)) return "coordinates";
        }
        return "";
    }

    static std::vector<std::string> group_names(const std::string& group, std::size_t layer_hint, std::size_t layers) {
        std::vector<std::string> names;
        if (group == "units") {
            for (std::size_t pass = 0; pass < 2; ++pass) {
                for (std::size_t l = 1; l <= layers; ++l) {
                    if ((pass == 0) != (layer_hint == 0 || l == layer_hint)) {
                        continue;
                    }
                    const std::string s = std::to_string(l);
                    for (const char* base : {"W_", "DEAD_L", "LIVE_L", "ACTIVE_L", "PARTIAL_L"}) {
                        names.push_back(base + s);
                    }
                }
            }
        } else if (group == "samples") {
            names = {"N", "CLOSED", "OPEN"};
        } else if (group == "outputs") {
            names = {"D_OUT"};
        } else if (group == "coordinates") {
            names = {"D_IN", "DRIVERS", "OTHER_INPUTS", "D_IN_MINUS_1"};
        }
        return names;
    }

    static std::string match_count(std::size_t value, const std::vector<std::string>& names, const ShapeSlotLexicon& lex) {
        for (const std::string& name : names) {
            const auto it = lex.counts.find(name);
            if (it != lex.counts.end() && it->second == value) {
                return name;
            }
        }
        return "";
    }

    // Parts that can be stated "of the <whole>" for each whole quantity.
    static std::vector<std::string> parts_of(const std::string& whole) {
        if (whole.rfind("W_", 0) == 0) {
            const std::string s = whole.substr(2);
            return {"DEAD_L" + s, "LIVE_L" + s, "ACTIVE_L" + s, "PARTIAL_L" + s};
        }
        if (whole == "N") return {"CLOSED", "OPEN"};
        if (whole == "D_IN") return {"DRIVERS", "OTHER_INPUTS", "D_IN_MINUS_1"};
        return {};
    }

    static std::string whole_name(const std::vector<std::string>& lower, std::size_t j, const ShapeSlotLexicon& lex) {
        std::size_t value = 0;
        if (!NumberWords::parse(lower[j], value)) {
            return "";
        }
        std::string group = noun_group(lower, j);
        std::vector<std::string> names = group.empty() ? std::vector<std::string>{} : group_names(group, 0, lex.layer_count);
        if (names.empty()) {
            names = group_names("units", 0, lex.layer_count);
            names.insert(names.end(), {"N", "D_IN"});
        }
        std::vector<std::string> wholes;
        for (const std::string& n : names) {
            if (n.rfind("W_", 0) == 0 || n == "N" || n == "D_IN" || n == "D_OUT") {
                wholes.push_back(n);
            }
        }
        return match_count(value, wholes, lex);
    }

    static std::string single_slot(const std::vector<std::string>& words,
                                   const std::vector<std::string>& lower,
                                   std::size_t i,
                                   const ShapeSlotLexicon& lex) {
        const std::string& word = words[i];
        const std::string& w = lower[i];

        if (w.size() > 1 && w[0] == 'x' && digits(w.substr(1))) {
            const std::string name = lex.coordinate_name(static_cast<std::size_t>(std::stoul(w.substr(1))));
            return name.empty() ? word : slot("COORD", name);
        }

        for (std::size_t k = 0; k < ordinals().size(); ++k) {
            const std::string& ord = ordinals()[k];
            const std::string next = i + 1 < lower.size() ? lower[i + 1] : "";
            const std::string name = "LAYER_" + std::to_string(k + 1);
            if (w == ord + "-layer") {
                return slot("ORD", name, capitalized(word) ? "Wl" : "wl");
            }
            if (w == ord && (next == "layer" || next == "layer's" || next == "layers")) {
                return slot("ORD", name, capitalized(word) ? "W" : "w");
            }
        }

        std::size_t value = 0;
        if (!NumberWords::parse(w, value) || value < 2) {
            return word;
        }
        if (i + 3 < lower.size() && lower[i + 1] == "of" && (lower[i + 2] == "the" || lower[i + 2] == "its")) {
            const std::string whole = whole_name(lower, i + 3, lex);
            if (!whole.empty()) {
                const std::string part = match_count(value, parts_of(whole), lex);
                if (!part.empty()) {
                    return slot("COUNT", part, form_of(word));
                }
            }
        }
        if (i >= 3 && lower[i - 2] == "of" && (lower[i - 1] == "the" || lower[i - 1] == "its")) {
            const std::string whole = whole_name(lower, i, lex);
            if (!whole.empty()) {
                return slot("DIM", whole, form_of(word));
            }
        }
        const std::string group = noun_group(lower, i);
        if (group.empty()) {
            return word;
        }
        const std::string name = match_count(value, group_names(group, 0, lex.layer_count), lex);
        if (name.empty()) {
            return word;
        }
        const bool dimension = name.rfind("W_", 0) == 0 || name == "N" || name == "D_IN" || name == "D_OUT";
        return slot(dimension ? "DIM" : "COUNT", name, form_of(word));
    }

    static bool fill(const std::string& token, const ShapeSlotLexicon& lex, std::string& value) {
        const std::string body = token.substr(1, token.size() - 2);
        const std::size_t colon = body.find(':');
        const std::size_t bar = body.find('|');
        const std::string kind = body.substr(0, colon);
        const std::string name = body.substr(colon + 1, bar == std::string::npos ? std::string::npos : bar - colon - 1);
        const std::string form = bar == std::string::npos ? "" : body.substr(bar + 1);

        if (kind == "COORD") {
            const auto it = lex.coordinates.find(name);
            if (it == lex.coordinates.end()) {
                return false;
            }
            value = "x" + std::to_string(it->second);
            return true;
        }
        if (kind == "LIST") {
            const auto it = lex.lists.find(name);
            if (it == lex.lists.end() || it->second.empty()) {
                return false;
            }
            value = EnglishListFormatter::indices(it->second);
            return true;
        }
        if (kind == "ORD") {
            const std::size_t k = static_cast<std::size_t>(std::stoul(name.substr(name.find('_') + 1)));
            if (k == 0 || k > lex.layer_count || k > ordinals().size()) {
                return false;
            }
            value = ordinals()[k - 1];
            if (form == "W" || form == "Wl") {
                value[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(value[0])));
            }
            if (form == "wl" || form == "Wl") {
                value += "-layer";
            }
            return true;
        }
        const auto it = lex.counts.find(name);
        if (it == lex.counts.end()) {
            return false;
        }
        value = form == "d" ? std::to_string(it->second)
                            : (form == "W" ? NumberWords::capitalized(it->second) : NumberWords::word(it->second));
        return true;
    }
};

}  // namespace forward_analysis
