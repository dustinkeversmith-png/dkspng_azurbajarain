#pragma once

#include "stevejob/core/helpers.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace stevejob::storage {

enum class MetricType { Number, Integer, Text, Token };

struct MetricField {
    std::string name;
    MetricType type = MetricType::Number;
    std::variant<double, std::int64_t, std::string> value;
    std::string prompt_description;

    std::string to_display_string() const {
        if (std::holds_alternative<double>(value)) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.4f", std::get<double>(value));
            return buf;
        }
        if (std::holds_alternative<std::int64_t>(value)) {
            return std::to_string(std::get<std::int64_t>(value));
        }
        return std::get<std::string>(value);
    }

    static std::string type_name(MetricType kind) {
        if (kind == MetricType::Integer) {
            return "integer";
        }
        if (kind == MetricType::Text) {
            return "text";
        }
        if (kind == MetricType::Token) {
            return "token";
        }
        return "number";
    }

    static MetricType type_from_name(const std::string& name) {
        if (name == "integer") {
            return MetricType::Integer;
        }
        if (name == "text") {
            return MetricType::Text;
        }
        if (name == "token") {
            return MetricType::Token;
        }
        return MetricType::Number;
    }
};

struct JournalRecord {
    double loss = 0.0;
    std::string checksum;
    std::vector<MetricField> metrics;

    double metric_number(const std::string& name, double fallback = 0.0) const {
        for (const MetricField& field : metrics) {
            if (field.name != name) {
                continue;
            }
            if (std::holds_alternative<double>(field.value)) {
                return std::get<double>(field.value);
            }
            if (std::holds_alternative<std::int64_t>(field.value)) {
                return static_cast<double>(std::get<std::int64_t>(field.value));
            }
        }
        return fallback;
    }

    std::string metric_text(const std::string& name) const {
        for (const MetricField& field : metrics) {
            if (field.name == name && std::holds_alternative<std::string>(field.value)) {
                return std::get<std::string>(field.value);
            }
        }
        return "";
    }

    bool counts_as_commit() const {
        const std::string status = metric_text("status");
        return status.empty() || status == "seed" || status == "committed_natively";
    }
};

class SlidingJournal {
public:
    explicit SlidingJournal(std::size_t capacity = 5,
                            std::string general_description =
                                "Sliding window of tournament trials. loss is the scored error. "
                                "A checkpoint write requires the loss to fall by more than 0.0001.")
        : capacity_(capacity == 0 ? 1 : capacity), general_description_(std::move(general_description)) {}

    void set_task(std::string task) { task_ = std::move(task); }
    const std::string& task() const { return task_; }

    void set_capacity(std::size_t capacity) {
        capacity_ = capacity == 0 ? 1 : capacity;
        while (records_.size() > capacity_) {
            records_.pop_front();
        }
    }

    void set_general_description(std::string description) { general_description_ = std::move(description); }
    const std::string& get_general_description() const { return general_description_; }

    void append_record(const JournalRecord& record) {
        stevejob::core::require_finite(record.loss);
        for (const MetricField& field : record.metrics) {
            if (std::holds_alternative<double>(field.value)) {
                stevejob::core::require_finite(std::get<double>(field.value));
            }
        }
        if (records_.size() >= capacity_) {
            records_.pop_front();
        }
        const bool committed = record.counts_as_commit();
        records_.push_back(record);
        if (committed) {
            last_committed_loss_ = record.loss;
            has_committed_loss_ = true;
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
        JournalRecord record;
        record.loss = loss;
        record.checksum = checksum;
        record.metrics.push_back(number_field("max_residual", max_residual, "Peak absolute residual on the eval grid."));
        record.metrics.push_back(number_field("std_residual", std_residual, "Residual standard deviation on the eval grid."));
        for (const auto& item : metrics) {
            record.metrics.push_back(number_field(item.first, item.second, description_for(item.first)));
        }
        for (const auto& item : meta) {
            record.metrics.push_back(text_field(item.first, item.second, description_for(item.first)));
        }
        const auto winner = meta.find("winner");
        if (winner != meta.end() && winner->second == "seed" && record.metric_text("status").empty()) {
            record.metrics.push_back(text_field("status", "seed", description_for("status")));
        }
        append_record(record);
    }

    std::string serialize() const {
        std::ostringstream out;
        out << "{\n  \"task\": \"" << escape(task_) << "\",\n";
        out << "  \"general_description\": \"" << escape(general_description_) << "\",\n";
        out << "  \"last_checksum\": \"" << (records_.empty() ? "" : records_.back().checksum) << "\",\n";
        if (has_committed_loss_) {
            out << std::fixed << std::setprecision(4);
            out << "  \"last_committed_loss\": " << last_committed_loss_ << ",\n";
        }
        out << "  \"history\": [\n";
        for (std::size_t index = 0; index < records_.size(); ++index) {
            const JournalRecord& record = records_[index];
            out << std::fixed << std::setprecision(4);
            out << "    {\n"
                << "      \"loss\": " << record.loss << ",\n"
                << "      \"checksum\": \"" << record.checksum << "\",\n"
                << "      \"metrics\": [";
            for (std::size_t metric_index = 0; metric_index < record.metrics.size(); ++metric_index) {
                const MetricField& field = record.metrics[metric_index];
                if (metric_index > 0) {
                    out << ", ";
                }
                out << "{\"name\": \"" << escape(field.name) << "\", \"type\": \"" << MetricField::type_name(field.type)
                    << "\", \"value\": ";
                if (std::holds_alternative<double>(field.value)) {
                    out << std::get<double>(field.value);
                } else if (std::holds_alternative<std::int64_t>(field.value)) {
                    out << std::get<std::int64_t>(field.value);
                } else {
                    out << "\"" << escape(std::get<std::string>(field.value)) << "\"";
                }
                out << ", \"prompt_description\": \"" << escape(field.prompt_description) << "\"}";
            }
            out << "]\n    }" << (index + 1 < records_.size() ? "," : "") << "\n";
        }
        out << "  ]\n}";
        return out.str();
    }

    void deserialize(const std::string& json_text) {
        records_.clear();
        has_committed_loss_ = false;
        const std::string description = read_string_field(json_text, "general_description");
        if (!description.empty()) {
            general_description_ = description;
        }
        const std::string task = read_string_field(json_text, "task");
        if (!task.empty()) {
            task_ = task;
        }
        std::size_t pos = 0;
        while ((pos = json_text.find("\"loss\":", pos)) != std::string::npos) {
            JournalRecord record;
            pos += 7;
            const std::size_t loss_end = json_text.find_first_of(",}", pos);
            record.loss = std::stod(json_text.substr(pos, loss_end - pos));
            const std::size_t object_begin = json_text.rfind('{', pos);
            std::size_t object_end = json_text.size();
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
            const std::string object = json_text.substr(object_begin, object_end - object_begin + 1);
            const std::size_t max_pos = object.find("\"max_residual\":");
            if (max_pos != std::string::npos) {
                const std::size_t value_pos = max_pos + 16;
                record.metrics.push_back(number_field(
                    "max_residual", std::stod(object.substr(value_pos)), "Peak absolute residual on the eval grid."));
            }
            const std::size_t std_pos = object.find("\"std_residual\":");
            if (std_pos != std::string::npos) {
                const std::size_t value_pos = std_pos + 16;
                record.metrics.push_back(number_field(
                    "std_residual", std::stod(object.substr(value_pos)), "Residual standard deviation on the eval grid."));
            }
            const std::string checksum = read_string_field(object, "checksum");
            if (!checksum.empty()) {
                record.checksum = checksum;
            }
            absorb_metrics(object, record);
            append_record(record);
        }
        const std::size_t committed_pos = json_text.find("\"last_committed_loss\":");
        if (committed_pos != std::string::npos) {
            last_committed_loss_ = std::stod(json_text.substr(committed_pos + 23));
            has_committed_loss_ = true;
        }
    }

    const JournalRecord* latest() const { return records_.empty() ? nullptr : &records_.back(); }

    std::string format_visible(std::size_t visible) const {
        std::ostringstream out;
        if (records_.empty()) {
            out << "No prior trials.\n";
            return out.str();
        }
        const std::size_t window = visible == 0 ? records_.size() : visible;
        const std::size_t begin = records_.size() > window ? records_.size() - window : 0;
        std::vector<std::string> seen;
        for (std::size_t index = begin; index < records_.size(); ++index) {
            for (const MetricField& field : records_[index].metrics) {
                bool known = false;
                for (const std::string& name : seen) {
                    if (name == field.name) {
                        known = true;
                    }
                }
                if (known) {
                    continue;
                }
                seen.push_back(field.name);
                out << "- " << field.name << ": " << field.prompt_description << "\n";
            }
        }
        for (std::size_t index = begin; index < records_.size(); ++index) {
            const JournalRecord& record = records_[index];
            char loss_text[32];
            std::snprintf(loss_text, sizeof(loss_text), "%.4f", record.loss);
            out << "- T" << (index + 1) << " loss=" << loss_text;
            for (const MetricField& field : record.metrics) {
                out << " " << field.name << "=" << field.to_display_string();
            }
            out << "\n";
        }
        return out.str();
    }

    const std::deque<JournalRecord>& records() const { return records_; }
    const std::deque<JournalRecord>& history() const { return records_; }
    std::size_t size() const { return records_.size(); }
    std::size_t capacity() const { return capacity_; }

    double committed_loss() const {
        if (!has_committed_loss_) {
            return std::numeric_limits<double>::infinity();
        }
        return last_committed_loss_;
    }

    static std::string description_for(const std::string& name) {
        if (name == "delta") {
            return "Signed loss change against the last committed checkpoint.";
        }
        if (name == "winner") {
            return "Candidate name that scored the lowest finite loss.";
        }
        if (name == "status") {
            return "Trial result: seed, committed_natively, rejected, or rejected_divergent.";
        }
        if (name == "winner_step") {
            return "Winning gamma and integer distance, written as g=<gamma> d=<distance>.";
        }
        if (name == "slice") {
            return "Slice id this trial edited.";
        }
        return "Recorded metric.";
    }

private:
    std::size_t capacity_;
    std::string task_ = "regression";
    std::string general_description_;
    std::deque<JournalRecord> records_;
    double last_committed_loss_ = 0.0;
    bool has_committed_loss_ = false;

    static MetricField number_field(const std::string& name, double value, const std::string& description) {
        MetricField field;
        field.name = name;
        field.type = MetricType::Number;
        field.value = value;
        field.prompt_description = description;
        return field;
    }

    static MetricField text_field(const std::string& name, const std::string& value, const std::string& description) {
        MetricField field;
        field.name = name;
        field.type = name == "winner_step" ? MetricType::Token : MetricType::Text;
        field.value = value;
        field.prompt_description = description;
        return field;
    }

    static std::string escape(const std::string& text) {
        std::string out;
        for (char character : text) {
            if (character == '"' || character == '\\') {
                out.push_back('\\');
            }
            out.push_back(character);
        }
        return out;
    }

    static std::string read_string_field(const std::string& object, const std::string& key) {
        const std::string marker = "\"" + key + "\": \"";
        const std::size_t pos = object.find(marker);
        if (pos == std::string::npos) {
            return "";
        }
        const std::size_t begin = pos + marker.size();
        std::string value;
        for (std::size_t cursor = begin; cursor < object.size(); ++cursor) {
            if (object[cursor] == '\\' && cursor + 1 < object.size()) {
                value.push_back(object[cursor + 1]);
                ++cursor;
                continue;
            }
            if (object[cursor] == '"') {
                break;
            }
            value.push_back(object[cursor]);
        }
        return value;
    }

    static void absorb_metrics(const std::string& object, JournalRecord& record) {
        const std::size_t marker = object.find("\"metrics\":");
        if (marker == std::string::npos) {
            absorb_legacy_object(object, "meta", record);
            return;
        }
        const std::size_t body = object.find_first_not_of(" \n\r\t", marker + 10);
        if (body == std::string::npos) {
            return;
        }
        if (object[body] == '[') {
            absorb_metric_array(object.substr(body), record);
            return;
        }
        absorb_legacy_object(object, "metrics", record);
        absorb_legacy_object(object, "meta", record);
    }

    static void absorb_metric_array(const std::string& array_text, JournalRecord& record) {
        std::size_t cursor = 0;
        while ((cursor = array_text.find("\"name\":", cursor)) != std::string::npos) {
            MetricField field;
            field.name = read_string_field(array_text.substr(cursor), "name");
            const std::string type_name = read_string_field(array_text.substr(cursor), "type");
            field.type = MetricField::type_from_name(type_name);
            field.prompt_description = read_string_field(array_text.substr(cursor), "prompt_description");
            const std::size_t value_pos = array_text.find("\"value\":", cursor);
            if (value_pos == std::string::npos) {
                break;
            }
            const std::size_t value_begin = array_text.find_first_not_of(" \n\r\t", value_pos + 8);
            if (array_text[value_begin] == '"') {
                field.value = read_string_field(array_text.substr(value_pos), "value");
                field.type = field.type == MetricType::Number ? MetricType::Text : field.type;
            } else if (field.type == MetricType::Integer) {
                field.value = static_cast<std::int64_t>(std::stoll(array_text.substr(value_begin)));
            } else {
                field.value = std::stod(array_text.substr(value_begin));
            }
            if (field.prompt_description.empty()) {
                field.prompt_description = description_for(field.name);
            }
            record.metrics.push_back(field);
            cursor = value_begin + 1;
        }
    }

    static void absorb_legacy_object(const std::string& object, const char* key, JournalRecord& record) {
        const std::string marker = std::string("\"") + key + "\": {";
        const std::size_t start = object.find(marker);
        if (start == std::string::npos) {
            return;
        }
        const std::size_t body = start + marker.size();
        const std::size_t end = object.find('}', body);
        std::size_t cursor = body;
        const bool textual = std::string(key) == "meta";
        while (cursor < end) {
            const std::size_t key_begin = object.find('"', cursor);
            if (key_begin == std::string::npos || key_begin >= end) {
                break;
            }
            const std::size_t key_end = object.find('"', key_begin + 1);
            const std::string name = object.substr(key_begin + 1, key_end - key_begin - 1);
            if (textual) {
                const std::size_t value_begin = object.find('"', object.find(':', key_end));
                const std::size_t value_end = object.find('"', value_begin + 1);
                record.metrics.push_back(
                    text_field(name, object.substr(value_begin + 1, value_end - value_begin - 1), description_for(name)));
                cursor = value_end + 1;
            } else {
                const std::size_t colon = object.find(':', key_end);
                record.metrics.push_back(
                    number_field(name, std::stod(object.substr(colon + 1)), description_for(name)));
                cursor = object.find_first_of(",}", colon);
                if (cursor == std::string::npos) {
                    break;
                }
            }
        }
    }
};

}  // namespace stevejob::storage
