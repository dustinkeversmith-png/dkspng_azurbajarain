#pragma once

#include "stevejob/models/architecture_catalog.hpp"
#include "stevejob/storage/json_text.hpp"

#include <cctype>
#include <map>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace stevejob::compiler {

class CandidateReader {
public:
    static py::dict extract(const std::string& architecture,
                            const std::string& text,
                            int rows,
                            int cols,
                            const py::dict& active_fields,
                            const py::dict& scheme_override) {
        if (rows < 1 || cols < 1) {
            throw std::invalid_argument("slice shape is missing");
        }
        const storage::JsonText root = load_object(text);
        const std::map<std::string, std::string> active = read_active(active_fields);
        return stevejob::models::ArchitectureCatalog::make(architecture, [&](auto& model) {
            const stevejob::models::CandidateScheme scheme = resolve(model.candidate_scheme(), scheme_override);
            const stevejob::models::SliceRef slice = shape_slice(rows, cols);
            const stevejob::messaging::IEncodingScheme& weights = model.get_scheme_for_slice(slice);
            const stevejob::messaging::IEncodingScheme& bias = model.get_bias_scheme(slice);
            const std::map<std::string, const storage::JsonText*> slots = collect(root, scheme);
            py::list candidates;
            py::dict gammas;
            for (const std::string& slot : scheme.slots) {
                const auto found = slots.find(slot);
                if (found == slots.end()) {
                    throw std::invalid_argument(slot + " is missing from response");
                }
                py::dict row;
                py::dict fields;
                row["name"] = slot;
                for (const stevejob::models::CandidateField& field : scheme.fields) {
                    const stevejob::messaging::IEncodingScheme& encoding = field.role == "bias" ? bias : weights;
                    stevejob::messaging::AcceptedValue accepted =
                        take_field(slot, field, *found->second, encoding, active, static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
                    fields[py::str(field.key)] = accepted.token;
                    if (field.role == "weights") {
                        row["w"] = accepted.token;
                        for (const std::pair<std::string, std::string>& note : accepted.notes) {
                            if (note.first == "gamma") {
                                gammas[py::str(slot)] = note.second;
                            }
                        }
                    } else if (field.role == "bias") {
                        row["b"] = accepted.token;
                    }
                }
                row["fields"] = fields;
                candidates.append(row);
            }
            py::dict result;
            result["candidates"] = candidates;
            result["reasoning"] = text_field(root, "reasoning");
            result["confidence"] = number_field(root, "confidence", 0.5);
            result["gammas"] = gammas;
            return result;
        });
    }

private:
    static storage::JsonText load_object(const std::string& text) {
        const std::size_t start = text.find('{');
        const std::size_t end = text.rfind('}');
        if (start == std::string::npos || end == std::string::npos || end <= start) {
            throw std::invalid_argument("completion has no JSON object");
        }
        try {
            storage::JsonText value = storage::JsonText::parse(text.substr(start, end - start + 1));
            if (value.kind != storage::JsonText::Kind::Object) {
                throw std::invalid_argument("completion JSON must be an object");
            }
            return value;
        } catch (const std::invalid_argument& exc) {
            if (std::string(exc.what()) == "completion JSON must be an object") {
                throw;
            }
            throw std::invalid_argument("completion JSON is incomplete");
        }
    }

    static std::map<std::string, std::string> read_active(const py::dict& active_fields) {
        std::map<std::string, std::string> active;
        for (const auto& item : active_fields) {
            active.emplace(item.first.cast<std::string>(), item.second.cast<std::string>());
        }
        return active;
    }

    static bool has_list(const py::dict& node, const char* key) {
        if (!node.contains(py::str(key)) || node[py::str(key)].is_none()) {
            return false;
        }
        return py::isinstance<py::list>(node[py::str(key)]) || py::isinstance<py::tuple>(node[py::str(key)]);
    }

    static stevejob::models::CandidateScheme resolve(stevejob::models::CandidateScheme scheme, const py::dict& override) {
        if (has_list(override, "slots")) {
            scheme.slots.clear();
            for (const py::handle item : override["slots"].cast<py::list>()) {
                scheme.slots.push_back(item.cast<std::string>());
            }
        }
        if (has_list(override, "fields")) {
            scheme.fields.clear();
            for (const py::handle item : override["fields"].cast<py::list>()) {
                const py::dict row = py::reinterpret_borrow<py::dict>(item);
                scheme.fields.push_back(stevejob::models::CandidateField{row["key"].cast<std::string>(), row["role"].cast<std::string>()});
            }
        }
        if (has_list(override, "groups")) {
            scheme.groups.clear();
            for (const py::handle item : override["groups"].cast<py::list>()) {
                scheme.groups.push_back(item.cast<std::string>());
            }
        }
        return scheme;
    }

    static stevejob::models::SliceRef shape_slice(int rows, int cols) {
        stevejob::models::SliceRef slice;
        slice.rows = static_cast<std::size_t>(rows);
        slice.cols = static_cast<std::size_t>(cols);
        return slice;
    }

    static const storage::JsonText* find_key(const storage::JsonText& node, const std::string& key) {
        if (node.kind != storage::JsonText::Kind::Object) {
            return nullptr;
        }
        for (const std::pair<std::string, storage::JsonText>& entry : node.object) {
            if (entry.first == key) {
                return &entry.second;
            }
        }
        return nullptr;
    }

    static std::string fold(const std::string& name) {
        std::string folded;
        for (unsigned char ch : name) {
            if (ch == ' ' || ch == '-') {
                continue;
            }
            folded.push_back(static_cast<char>(std::tolower(ch)));
        }
        return folded;
    }

    static std::string slot_of(const std::string& name, const stevejob::models::CandidateScheme& scheme) {
        const std::string folded = fold(name);
        for (const std::string& slot : scheme.slots) {
            if (folded == fold(slot)) {
                return slot;
            }
        }
        const std::string aliased = alias_slot(folded);
        if (!aliased.empty()) {
            for (const std::string& slot : scheme.slots) {
                if (fold(slot) == aliased) {
                    return slot;
                }
            }
        }
        return "";
    }

    static std::string alias_slot(const std::string& folded) {
        if (folded == "candidate1" || folded == "candidate_1") {
            return "c1";
        }
        if (folded == "candidate2" || folded == "candidate_2") {
            return "c2";
        }
        if (folded == "candidate3" || folded == "candidate_3") {
            return "c3";
        }
        return "";
    }

    static bool complete(const std::map<std::string, const storage::JsonText*>& found, const stevejob::models::CandidateScheme& scheme) {
        for (const std::string& slot : scheme.slots) {
            if (found.find(slot) == found.end()) {
                return false;
            }
        }
        return !scheme.slots.empty();
    }

    static void merge(std::map<std::string, const storage::JsonText*>& found, const std::map<std::string, const storage::JsonText*>& extra) {
        for (const auto& entry : extra) {
            found.emplace(entry.first, entry.second);
        }
    }

    static void fill_named(std::map<std::string, const storage::JsonText*>& found,
                           const storage::JsonText& node,
                           const stevejob::models::CandidateScheme& scheme) {
        if (node.kind != storage::JsonText::Kind::Object) {
            return;
        }
        for (const std::pair<std::string, storage::JsonText>& entry : node.object) {
            const std::string slot = slot_of(entry.first, scheme);
            if (!slot.empty()) {
                found.emplace(slot, &entry.second);
            }
        }
    }

    static std::map<std::string, const storage::JsonText*> slots_from_list(const storage::JsonText& array,
                                                                           const stevejob::models::CandidateScheme& scheme) {
        std::map<std::string, const storage::JsonText*> found;
        std::vector<const storage::JsonText*> positional;
        for (const storage::JsonText& item : array.array) {
            if (item.kind != storage::JsonText::Kind::Object) {
                positional.push_back(&item);
                continue;
            }
            std::string named;
            for (const char* key : {"name", "id", "candidate"}) {
                const storage::JsonText* value = find_key(item, key);
                if (value != nullptr && value->kind == storage::JsonText::Kind::String) {
                    named = slot_of(value->text, scheme);
                    if (!named.empty()) {
                        break;
                    }
                }
            }
            if (!named.empty()) {
                found.emplace(named, &item);
            } else {
                positional.push_back(&item);
            }
        }
        if (found.size() < scheme.slots.size() && positional.size() >= scheme.slots.size()) {
            for (std::size_t index = 0; index < scheme.slots.size(); ++index) {
                found.emplace(scheme.slots[index], positional[index]);
            }
        }
        return found;
    }

    static std::map<std::string, const storage::JsonText*> collect(const storage::JsonText& parsed,
                                                                   const stevejob::models::CandidateScheme& scheme) {
        std::map<std::string, const storage::JsonText*> found;
        fill_named(found, parsed, scheme);
        if (complete(found, scheme)) {
            return found;
        }
        for (const std::string& group : scheme.groups) {
            const storage::JsonText* block = find_key(parsed, group);
            if (block == nullptr) {
                continue;
            }
            if (block->kind == storage::JsonText::Kind::Array) {
                merge(found, slots_from_list(*block, scheme));
            } else if (block->kind == storage::JsonText::Kind::Object) {
                merge(found, collect(*block, scheme));
            }
            if (complete(found, scheme)) {
                return found;
            }
        }
        return found;
    }

    static const storage::JsonText* field_value(const storage::JsonText& item, const stevejob::models::CandidateField& field) {
        if (item.kind != storage::JsonText::Kind::Object) {
            return &item;
        }
        const storage::JsonText* direct = find_key(item, field.key);
        if (direct != nullptr) {
            return direct;
        }
        if (field.role == "weights") {
            for (const char* key : {"weight", "weights", "totr", "token"}) {
                const storage::JsonText* alias = find_key(item, key);
                if (alias != nullptr) {
                    return alias;
                }
            }
            return &item;
        }
        if (field.role == "bias") {
            for (const char* key : {"bias", "biases"}) {
                const storage::JsonText* alias = find_key(item, key);
                if (alias != nullptr) {
                    return alias;
                }
            }
        }
        return nullptr;
    }

    static stevejob::messaging::AcceptedValue take_field(const std::string& slot,
                                                         const stevejob::models::CandidateField& field,
                                                         const storage::JsonText& item,
                                                         const stevejob::messaging::IEncodingScheme& encoding,
                                                         const std::map<std::string, std::string>& active,
                                                         std::size_t rows,
                                                         std::size_t cols) {
        const storage::JsonText* value = field_value(item, field);
        if (value == nullptr) {
            const auto fallback = active.find(field.key);
            if (field.role == "bias" && fallback != active.end() && !fallback->second.empty()) {
                return stevejob::messaging::AcceptedValue{fallback->second, {}};
            }
            if (field.role == "bias") {
                throw std::invalid_argument("candidate bias is missing");
            }
            throw std::invalid_argument(slot + " is missing field " + field.key);
        }
        try {
            return encoding.accept(*value, rows, cols);
        } catch (const std::invalid_argument&) {
            if (field.role == "bias") {
                const auto fallback = active.find(field.key);
                if (fallback != active.end() && !fallback->second.empty()) {
                    return stevejob::messaging::AcceptedValue{fallback->second, {}};
                }
                throw std::invalid_argument("candidate bias is missing");
            }
            throw std::invalid_argument(slot + " has no valid " + encoding.get_descriptor().scheme_name + " value with " +
                                        std::to_string(rows) + " by " + std::to_string(cols) + " factors");
        }
    }

    static std::string text_field(const storage::JsonText& node, const char* key) {
        const storage::JsonText* value = find_key(node, key);
        if (value != nullptr && value->kind == storage::JsonText::Kind::String) {
            return value->text;
        }
        return "";
    }

    static double number_field(const storage::JsonText& node, const char* key, double fallback) {
        const storage::JsonText* value = find_key(node, key);
        if (value != nullptr && value->kind == storage::JsonText::Kind::Number) {
            return value->number;
        }
        return fallback;
    }
};

}  // namespace stevejob::compiler
