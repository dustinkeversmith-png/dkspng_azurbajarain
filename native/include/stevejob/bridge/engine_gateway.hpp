#pragma once

#include "stevejob/bridge/tournament.hpp"
#include "stevejob/compiler/prompt_compiler.hpp"
#include "stevejob/models/architecture_catalog.hpp"
#include "stevejob/models/model_checkpoint.hpp"

#include <pybind11/pybind11.h>

#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace stevejob::bridge {

class EngineGateway {
public:
    template <typename Model>
    static py::dict seed_model(Model& model,
                               const std::string& target_blob,
                               const std::string& journal_blob,
                               std::size_t journal_capacity) {
        std::lock_guard<std::mutex> lock(gate());
        return stevejob::models::ModelCheckpoint::store(model, target_blob, journal_blob, journal_capacity);
    }

    static py::dict seed_kind(const std::string& kind,
                              const std::string& target_blob,
                              const std::string& journal_blob,
                              std::size_t journal_capacity) {
        return stevejob::models::ArchitectureCatalog::make(kind, [&](auto& model) {
            return seed_model(model, target_blob, journal_blob, journal_capacity);
        });
    }

    static py::dict read_meta(const std::string& target_blob, const std::string& journal_blob) {
        (void)journal_blob;
        const stevejob::models::LoadedCheckpoint loaded = stevejob::models::ModelCheckpoint::open(target_blob);
        const stevejob::models::ModelView view = stevejob::models::ModelCheckpoint::inspect(loaded.model);
        return stevejob::models::ModelCheckpoint::ready_receipt(loaded, view);
    }

    static py::dict read_slice(const std::string& target_blob,
                               const std::string& journal_blob,
                               const std::string& slice_id) {
        (void)journal_blob;
        const stevejob::models::LoadedCheckpoint loaded = stevejob::models::ModelCheckpoint::open(target_blob);
        return stevejob::models::ModelCheckpoint::slice_receipt(loaded, slice_id);
    }

    static std::string compile_prompt(const std::string& target_blob,
                                     const std::string& journal_blob,
                                     const std::string& slice_id,
                                     std::size_t visible_history,
                                     int int_bound,
                                     const std::string& retry_note) {
        std::lock_guard<std::mutex> lock(gate());
        stevejob::models::LoadedCheckpoint loaded = stevejob::models::ModelCheckpoint::open(target_blob);
        stevejob::storage::AzuriteClient client;
        return stevejob::models::ArchitectureCatalog::visit(std::move(loaded.model), [&](auto& model) {
            const stevejob::compiler::PromptCompilerOptions options =
                prompt_options(model.config(), visible_history, int_bound, retry_note);
            model.bind_journal(0);
            stevejob::models::ModelCheckpoint::read_journal(
                client,
                stevejob::storage::BlobPath::bare(journal_blob, "evals/"),
                model.journal());
            const double current_loss = model.evaluate_model().primary_loss;
            return stevejob::compiler::PromptCompiler::compile_slice_prompt(
                model, slice_id, model.journal(), current_loss, options);
        });
    }

    static py::dict commit_layers(const std::vector<std::string>& weight_tokens,
                                  const std::vector<std::string>& bias_tokens,
                                  const std::vector<std::vector<int>>& shapes,
                                  const std::string& target_blob,
                                  const std::string& journal_blob,
                                  std::size_t journal_capacity) {
        std::lock_guard<std::mutex> lock(gate());
        return Tournament::commit_layers(weight_tokens, bias_tokens, shapes, target_blob, journal_blob, journal_capacity);
    }

    static py::dict commit_tournament(const std::string& slice_id,
                                      const std::vector<TournamentCandidate>& candidates,
                                      const std::string& target_blob,
                                      const std::string& journal_blob,
                                      std::size_t journal_capacity) {
        std::lock_guard<std::mutex> lock(gate());
        return Tournament::commit(slice_id, candidates, target_blob, journal_blob, journal_capacity);
    }

private:
    static std::mutex& gate() {
        static std::mutex mutex;
        return mutex;
    }

    static stevejob::compiler::PromptCompilerOptions prompt_options(const stevejob::models::LearningConfig& cfg,
                                                                    std::size_t visible_history,
                                                                    int int_bound,
                                                                    const std::string& retry_note) {
        stevejob::compiler::PromptCompilerOptions options;
        options.visible_history_count = visible_history == 0 ? cfg.visible_history_count : visible_history;
        options.int_bound = int_bound == 0 ? cfg.int_bound : int_bound;
        options.retry_note = retry_note;
        return options;
    }
};

}  // namespace stevejob::bridge
