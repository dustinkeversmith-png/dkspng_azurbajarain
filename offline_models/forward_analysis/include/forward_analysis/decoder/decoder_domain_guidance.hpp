#pragma once

#include "forward_analysis/decoder/observation_tokenizer.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <string>
#include <vector>

namespace forward_analysis {

// Applies domain-head logit calibration during autoregressive decoding using only the predicted conditioning vector d_hat:
//   d_hat(3): NarrowRepresentationHead probability p_narrow in [0, 1]
//   d_hat(4): LayerStructureHead probability p_two_layer in [0, 1]
//   d_hat(7): FeatureAssociationHead probability p_nonzero_assoc in [0, 1]
struct DecoderDomainGuidance {
    static void apply(std::size_t field,
                      const std::vector<int>& emitted_ids,
                      const Eigen::VectorXd& descriptor,
                      const ObservationTokenizer& tokenizer,
                      Eigen::VectorXd& logits) {
        if (descriptor.size() < 9) {
            return;
        }
        // Only apply when the tokenizer contains the batch-statistics observation slots.
        if (token_id(tokenizer, "<COUNT:W_1|d>") < 0 || token_id(tokenizer, "<COUNT:RANK_L1|d>") < 0) {
            return;
        }
        // Predicted narrow-representation probability from NarrowRepresentationHead.
        const double p_narrow = descriptor(3);
        // Predicted two-layer topology probability from LayerStructureHead.
        const double p_two_layer = descriptor(4);
        // Predicted nonzero-association probability from FeatureAssociationHead.
        const double p_nonzero_assoc = descriptor(7);
        // True when LayerStructureHead predicts a 2-layer network.
        const bool is_two_layer = p_two_layer >= 0.5;

        // Previous token string if available.
        const std::string prev_tok =
            (!emitted_ids.empty() && emitted_ids.back() >= 0 &&
             static_cast<std::size_t>(emitted_ids.back()) < tokenizer.vocab_size())
                ? tokenizer.token(emitted_ids.back())
                : "";
        // Second-previous token string if available.
        const std::string prev2_tok =
            (emitted_ids.size() >= 2 && emitted_ids[emitted_ids.size() - 2] >= 0 &&
             static_cast<std::size_t>(emitted_ids[emitted_ids.size() - 2]) < tokenizer.vocab_size())
                ? tokenizer.token(emitted_ids[emitted_ids.size() - 2])
                : "";

        // 1. Structural layer-count constraints (LayerStructureHead):
        if (!is_two_layer) {
            // Predicted 1-layer network: suppress 2nd-layer slot tokens.
            for (const char* slot2 : {"<COUNT:W_2|d>", "<COUNT:ZERO_L2|d>", "<COUNT:NONPOS_L2|d>", "<COUNT:RANK_L2|d>"}) {
                // Token id of the 2nd-layer slot.
                const int id = token_id(tokenizer, slot2);
                if (id >= 0 && id < logits.size()) {
                    logits(id) -= 25.0;
                }
            }
        }

        // Field 0 (pathway_activity) structural guidance:
        if (field == 0) {
            // True when Layer 1 zero-count slot has already been emitted.
            const bool has_zero_l1 = contains_lexeme(emitted_ids, tokenizer, "<COUNT:ZERO_L1|d>");
            // True when Layer 2 zero-count slot has already been emitted.
            const bool has_zero_l2 = contains_lexeme(emitted_ids, tokenizer, "<COUNT:ZERO_L2|d>");
            // True when Layer 1 nonpositive-sample slot has already been emitted.
            const bool has_nonpos_l1 = contains_lexeme(emitted_ids, tokenizer, "<COUNT:NONPOS_L1|d>");
            // True when Layer 2 nonpositive-sample slot has already been emitted.
            const bool has_nonpos_l2 = contains_lexeme(emitted_ids, tokenizer, "<COUNT:NONPOS_L2|d>");

            if (prev_tok == "<COUNT:W_1|d>") {
                if (is_two_layer) {
                    add_logit(tokenizer, "and", 15.0, logits);
                    add_logit(tokenizer, ".", -15.0, logits);
                } else {
                    add_logit(tokenizer, ".", 15.0, logits);
                    add_logit(tokenizer, "and", -15.0, logits);
                }
            } else if (prev2_tok == "<COUNT:W_1|d>" && prev_tok == "and") {
                add_logit(tokenizer, "<COUNT:W_2|d>", 15.0, logits);
            } else if (prev_tok == "<COUNT:W_2|d>") {
                add_logit(tokenizer, ".", 15.0, logits);
            } else if (prev_tok == "Layer") {
                if (!has_zero_l1) {
                    add_logit(tokenizer, "1", 15.0, logits);
                    add_logit(tokenizer, "2", -15.0, logits);
                } else {
                    add_logit(tokenizer, "2", 15.0, logits);
                    add_logit(tokenizer, "1", -15.0, logits);
                }
            } else if (prev_tok == "has") {
                if (!has_zero_l1) {
                    add_logit(tokenizer, "<COUNT:ZERO_L1|d>", 15.0, logits);
                    add_logit(tokenizer, "<COUNT:ZERO_L2|d>", -15.0, logits);
                } else if (!has_zero_l2) {
                    add_logit(tokenizer, "<COUNT:ZERO_L2|d>", 15.0, logits);
                    add_logit(tokenizer, "<COUNT:ZERO_L1|d>", -15.0, logits);
                }
            } else if (prev2_tok == "batch" && prev_tok == "and") {
                if (!has_nonpos_l1) {
                    add_logit(tokenizer, "<COUNT:NONPOS_L1|d>", 15.0, logits);
                    add_logit(tokenizer, "<COUNT:NONPOS_L2|d>", -15.0, logits);
                } else if (!has_nonpos_l2) {
                    add_logit(tokenizer, "<COUNT:NONPOS_L2|d>", 15.0, logits);
                    add_logit(tokenizer, "<COUNT:NONPOS_L1|d>", -15.0, logits);
                }
            } else if (prev_tok == ".") {
                if (!has_nonpos_l1) {
                    // End of width sentence -> start Layer 1 sentence.
                    add_logit(tokenizer, "Layer", 15.0, logits);
                    logits(ObservationTokenizer::kEosToken) -= 15.0;
                } else if (!has_nonpos_l2) {
                    // End of Layer 1 sentence -> EOS for 1-layer, or Layer 2 for 2-layer.
                    if (is_two_layer) {
                        add_logit(tokenizer, "Layer", 15.0, logits);
                        logits(ObservationTokenizer::kEosToken) -= 15.0;
                    } else {
                        logits(ObservationTokenizer::kEosToken) += 15.0;
                        add_logit(tokenizer, "Layer", -15.0, logits);
                    }
                } else {
                    // End of Layer 2 sentence -> EOS.
                    logits(ObservationTokenizer::kEosToken) += 15.0;
                    add_logit(tokenizer, "Layer", -15.0, logits);
                }
            }
        }

        // 2. Field 2 (representational_structure) structural + geometric guidance (LayerStructureHead + NarrowRepresentationHead):
        if (field == 2) {
            if (prev_tok == "<COUNT:RANK_L1|d>") {
                if (is_two_layer) {
                    add_logit(tokenizer, ",", 15.0, logits);
                    add_logit(tokenizer, "across", -15.0, logits);
                } else {
                    add_logit(tokenizer, "across", 15.0, logits);
                    add_logit(tokenizer, ",", -15.0, logits);
                }
            } else if (prev2_tok == "<COUNT:RANK_L1|d>" && prev_tok == ",") {
                add_logit(tokenizer, "<COUNT:RANK_L2|d>", 15.0, logits);
            } else if (prev_tok == "<COUNT:RANK_L2|d>") {
                add_logit(tokenizer, "across", 15.0, logits);
                add_logit(tokenizer, ",", -15.0, logits);
            } else if (prev2_tok == "layers" && prev_tok == ".") {
                // Start of second sentence: "This is a narrow observed representation..." vs "The observed activation variation..."
                const double narrow_bias = (p_narrow - 0.5) * 30.0;
                add_logit(tokenizer, "This", narrow_bias, logits);
                add_logit(tokenizer, "The", -narrow_bias, logits);
            } else if ((prev2_tok == "widths" || prev2_tok == "layer") && prev_tok == ".") {
                // End of second sentence -> EOS.
                logits(ObservationTokenizer::kEosToken) += 15.0;
            }
        }

        // 3. Field 1 (feature_sensitivity) statistical guidance (FeatureAssociationHead):
        if (field == 1 && prev_tok == ":") {
            // Logit bias when association is predicted to be nonzero vs zero.
            const double assoc_bias = (p_nonzero_assoc - 0.5) * 24.0;
            add_logit(tokenizer, "input", assoc_bias, logits);
            add_logit(tokenizer, "no", -assoc_bias, logits);
        }
    }

private:
    static int token_id(const ObservationTokenizer& tokenizer, const std::string& lexeme) {
        // Encoded token sequence [BOS, token, EOS].
        const std::vector<int> ids = tokenizer.encode_known_lexemes({lexeme}, 4);
        if (ids.size() >= 2 && ids[1] != ObservationTokenizer::kUnkToken) {
            return ids[1];
        }
        return -1;
    }

    static void add_logit(const ObservationTokenizer& tokenizer,
                          const std::string& lexeme,
                          double delta,
                          Eigen::VectorXd& logits) {
        // Vocabulary id of `lexeme`.
        const int id = token_id(tokenizer, lexeme);
        if (id >= 0 && id < logits.size()) {
            logits(id) += delta;
        }
    }

    static bool contains_lexeme(const std::vector<int>& ids,
                                const ObservationTokenizer& tokenizer,
                                const std::string& lexeme) {
        // Target token id.
        const int target = token_id(tokenizer, lexeme);
        if (target < 0) {
            return false;
        }
        for (int id : ids) {
            if (id == target) {
                return true;
            }
        }
        return false;
    }
};

}  // namespace forward_analysis
