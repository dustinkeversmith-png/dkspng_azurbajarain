#pragma once

#include "stevejob/core/helpers.hpp"

#include <cstddef>
#include <deque>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace stevejob::storage {

struct JournalEntry {
    double loss = 0.0;
    double max_residual = 0.0;
    double std_residual = 0.0;
    std::string checksum;
    std::map<std::string, double> metrics;
    std::map<std::string, std::string> meta;
};

class SlidingJournal {
public:
    explicit SlidingJournal(std::size_t max_history = 5) : max_history_(max_history == 0 ? 1 : max_history) {}

    void set_capacity(std::size_t capacity) {
        max_history_ = capacity == 0 ? 1 : capacity;
        while (history_.size() > max_history_) {
            history_.pop_front();
        }
    }

    void add_entry(double loss, double max_residual, double std_residual, const std::string& checksum) {
        add_record(loss, max_residual, std_residual, checksum, {}, {});
    }

    void add_record(double loss,
                    double max_residual,
                    double std_residual,
                    const std::string& checksum,
                    const std::map<std::string, double>& metrics,
                    const std::map<std::string, std::string>& meta) {
        stevejob::core::require_finite(loss);
        stevejob::core::require_finite(max_residual);
        stevejob::core::require_finite(std_residual);
        for (const auto& item : metrics) {
            stevejob::core::require_finite(item.second);
        }
        if (history_.size() >= max_history_) {
            history_.pop_front();
        }
        history_.push_back(JournalEntry{loss, max_residual, std_residual, checksum, metrics, meta});
    }

    std::string serialize() const {
        std::ostringstream out;
        out << "{\n  \"task\": \"regression\",\n";
        out << "  \"last_checksum\": \"" << (history_.empty() ? "" : history_.back().checksum) << "\",\n";
        out << "  \"history\": [\n";
        out << std::fixed << std::setprecision(4);
        for (std::size_t index = 0; index < history_.size(); ++index) {
            const JournalEntry& entry = history_[index];
            out << "    {\n"
                << "      \"loss\": " << entry.loss << ",\n"
                << "      \"max_residual\": " << entry.max_residual << ",\n"
                << "      \"std_residual\": " << entry.std_residual << ",\n"
                << "      \"checksum\": \"" << entry.checksum << "\"";
            if (!entry.metrics.empty()) {
                out << ",\n      \"metrics\": {";
                bool first = true;
                for (const auto& item : entry.metrics) {
                    if (!first) {
                        out << ", ";
                    }
                    first = false;
                    out << "\"" << item.first << "\": " << item.second;
                }
                out << "}";
            }
            if (!entry.meta.empty()) {
                out << ",\n      \"meta\": {";
                bool first = true;
                for (const auto& item : entry.meta) {
                    if (!first) {
                        out << ", ";
                    }
                    first = false;
                    out << "\"" << item.first << "\": \"" << item.second << "\"";
                }
                out << "}";
            }
            out << "\n    }" << (index + 1 < history_.size() ? "," : "") << "\n";
        }
        out << "  ]\n}";
        return out.str();
    }

    void deserialize(const std::string& json_text) {
        history_.clear();
        std::size_t pos = 0;
        while ((pos = json_text.find("\"loss\":", pos)) != std::string::npos) {
            JournalEntry entry;
            pos += 7;
            const std::size_t loss_end = json_text.find(',', pos);
            entry.loss = std::stod(json_text.substr(pos, loss_end - pos));
            const std::size_t object_begin = json_text.rfind('{', pos);
            std::size_t object_end = json_text.find('}', pos);
            int depth = 0;
            for (std::size_t cursor = object_begin; cursor < json_text.size(); ++cursor) {
                if (json_text[cursor] == '{') {
                    ++depth;
                } else if (json_text[cursor] == '}') {
                    --depth;
                    if (depth == 0) {
                        object_end = cursor;
                        break;
                    }
                }
            }
            const std::size_t max_pos = json_text.find("\"max_residual\":", pos);
            if (max_pos != std::string::npos && max_pos < object_end) {
                const std::size_t value_pos = max_pos + 16;
                entry.max_residual = std::stod(json_text.substr(value_pos, json_text.find(',', value_pos) - value_pos));
            }
            const std::size_t std_pos = json_text.find("\"std_residual\":", pos);
            if (std_pos != std::string::npos && std_pos < object_end) {
                const std::size_t value_pos = std_pos + 16;
                entry.std_residual = std::stod(json_text.substr(value_pos, json_text.find(',', value_pos) - value_pos));
            }
            const std::size_t checksum_pos = json_text.find("\"checksum\": \"", pos);
            if (checksum_pos != std::string::npos && checksum_pos < object_end) {
                const std::size_t value_pos = checksum_pos + 13;
                entry.checksum = json_text.substr(value_pos, json_text.find('"', value_pos) - value_pos);
            }
            const std::string object = json_text.substr(pos, object_end - pos);
            entry.metrics = read_number_object(object, "metrics");
            entry.meta = read_string_object(object, "meta");
            history_.push_back(entry);
            if (history_.size() > max_history_) {
                history_.pop_front();
            }
        }
    }

    const std::deque<JournalEntry>& history() const { return history_; }
    std::size_t size() const { return history_.size(); }

private:
    std::size_t max_history_;
    std::deque<JournalEntry> history_;

    static std::map<std::string, double> read_number_object(const std::string& object, const char* key) {
        std::map<std::string, double> values;
        const std::string marker = std::string("\"") + key + "\": {";
        const std::size_t start = object.find(marker);
        if (start == std::string::npos) {
            return values;
        }
        const std::size_t body = start + marker.size();
        const std::size_t end = object.find('}', body);
        std::size_t cursor = body;
        while (cursor < end) {
            const std::size_t key_begin = object.find('"', cursor);
            if (key_begin == std::string::npos || key_begin >= end) {
                break;
            }
            const std::size_t key_end = object.find('"', key_begin + 1);
            const std::string name = object.substr(key_begin + 1, key_end - key_begin - 1);
            const std::size_t colon = object.find(':', key_end);
            values[name] = std::stod(object.substr(colon + 1, object.find_first_of(",}", colon) - colon - 1));
            cursor = key_end + 1;
        }
        return values;
    }

    static std::map<std::string, std::string> read_string_object(const std::string& object, const char* key) {
        std::map<std::string, std::string> values;
        const std::string marker = std::string("\"") + key + "\": {";
        const std::size_t start = object.find(marker);
        if (start == std::string::npos) {
            return values;
        }
        const std::size_t body = start + marker.size();
        const std::size_t end = object.find('}', body);
        std::size_t cursor = body;
        while (cursor < end) {
            const std::size_t key_begin = object.find('"', cursor);
            if (key_begin == std::string::npos || key_begin >= end) {
                break;
            }
            const std::size_t key_end = object.find('"', key_begin + 1);
            const std::string name = object.substr(key_begin + 1, key_end - key_begin - 1);
            const std::size_t value_begin = object.find('"', object.find(':', key_end));
            const std::size_t value_end = object.find('"', value_begin + 1);
            values[name] = object.substr(value_begin + 1, value_end - value_begin - 1);
            cursor = value_end + 1;
        }
        return values;
    }
};

}  // namespace stevejob::storage
