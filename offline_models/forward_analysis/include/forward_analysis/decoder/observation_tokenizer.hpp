#pragma once

#include "forward_analysis/text/lexeme_splitter.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace forward_analysis {

// Word-level vocabulary for observation text: each word and each punctuation mark is one token.
class ObservationTokenizer {
public:
    static constexpr int kPadToken = 0;
    static constexpr int kBosToken = 1;
    static constexpr int kEosToken = 2;
    static constexpr int kUnkToken = 3;

    ObservationTokenizer() {
        for (const char* special : {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}) {
            register_token(special);
        }
    }

    int register_token(const std::string& token) {
        const auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        const int id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_.emplace(token, id);
        return id;
    }

    // BOS, word ids, EOS; at most max_tokens ids, always ending with EOS.
    std::vector<int> encode(const std::string& text, std::size_t max_tokens, bool register_new) {
        return encode_lexemes(LexemeSplitter::split(text), max_tokens, register_new);
    }

    // Like encode, but unseen words map to UNK and the vocabulary is left unchanged.
    std::vector<int> encode_known(const std::string& text, std::size_t max_tokens) const {
        return encode_known_lexemes(LexemeSplitter::split(text), max_tokens);
    }

    // encode over an already split lexeme sequence (slot tokens such as "<DIM:W_1|w>" stay whole).
    std::vector<int> encode_lexemes(const std::vector<std::string>& lexemes, std::size_t max_tokens, bool register_new) {
        std::vector<int> ids{kBosToken};
        for (const std::string& lex : lexemes) {
            if (ids.size() + 1 >= std::max<std::size_t>(2, max_tokens)) {
                break;
            }
            const auto it = token_to_id_.find(lex);
            ids.push_back(it != token_to_id_.end() ? it->second : (register_new ? register_token(lex) : kUnkToken));
        }
        ids.push_back(kEosToken);
        return ids;
    }

    std::vector<int> encode_known_lexemes(const std::vector<std::string>& lexemes, std::size_t max_tokens) const {
        std::vector<int> ids{kBosToken};
        for (const std::string& lex : lexemes) {
            if (ids.size() + 1 >= std::max<std::size_t>(2, max_tokens)) {
                break;
            }
            const auto it = token_to_id_.find(lex);
            ids.push_back(it != token_to_id_.end() ? it->second : kUnkToken);
        }
        ids.push_back(kEosToken);
        return ids;
    }

    // Token strings up to the first EOS, skipping PAD/BOS and out-of-range ids.
    std::vector<std::string> decode_lexemes(const std::vector<int>& ids) const {
        std::vector<std::string> lexemes;
        for (int id : ids) {
            if (id == kEosToken) {
                break;
            }
            if (id == kPadToken || id == kBosToken || id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size()) {
                continue;
            }
            lexemes.push_back(id_to_token_[static_cast<std::size_t>(id)]);
        }
        return lexemes;
    }

    std::string decode(const std::vector<int>& ids) const {
        return LexemeSplitter::join(decode_lexemes(ids));
    }

    std::size_t vocab_size() const noexcept { return id_to_token_.size(); }

    const std::string& token(int id) const { return id_to_token_.at(static_cast<std::size_t>(id)); }

private:
    // Token string for each id; index = token id.
    std::vector<std::string> id_to_token_;
    // Reverse lookup from token string to id.
    std::unordered_map<std::string, int> token_to_id_;
};

}  // namespace forward_analysis
