#pragma once

#include "stevejob/core/metric_descriptor.hpp"
#include "stevejob/models/evaluation_score.hpp"
#include "stevejob/storage/sliding_journal.hpp"

#include <string>
#include <variant>

namespace stevejob::storage {

class MetricFields {
public:
    static MetricField number(const std::string& name, double value, const std::string& description) {
        MetricField field;
        field.name = name;
        field.type = MetricType::Number;
        field.value = value;
        field.prompt_description = description;
        return field;
    }

    static MetricField number(const std::string& name, double value) {
        return number(name, value, SlidingJournal::description_for(name));
    }

    static MetricField text(const std::string& name, const std::string& value, MetricType type = MetricType::Text) {
        MetricField field;
        field.name = name;
        field.type = type;
        field.value = value;
        field.prompt_description = SlidingJournal::description_for(name);
        return field;
    }

    static void append_score(JournalRecord& record, const stevejob::models::ModelEvaluation& score) {
        for (const stevejob::core::DescribedField& field : score.metrics) {
            if (field.key == "loss" || !std::holds_alternative<double>(field.value)) {
                continue;
            }
            record.metrics.push_back(number(field.key, std::get<double>(field.value), field.actionable_rule));
        }
    }
};

}  // namespace stevejob::storage
