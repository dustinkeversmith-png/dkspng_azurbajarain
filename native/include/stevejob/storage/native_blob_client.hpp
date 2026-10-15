#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <wincrypt.h>
#include <winhttp.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace stevejob::storage {

struct QueryItem {
    std::string key;
    std::string value;
};

class NativeBlobClient {
public:
    static constexpr std::size_t kMaxBlobBytes = 128u * 1024u * 1024u;

    explicit NativeBlobClient(std::string connection_string = "") {
        account_ = parse_connection(resolve_connection(std::move(connection_string)));
    }

    std::string download_blob(const std::string& container, const std::string& blob_name) {
        const std::string name = normalize_name(blob_name, container);
        ensure_container(container);
        const std::string payload = exchange("GET", "/" + container + "/" + name, {}, "", "");
        if (payload.size() > kMaxBlobBytes) {
            throw std::invalid_argument(container + "/" + name + " is too large for this tiny run");
        }
        return payload;
    }

    void upload_blob(const std::string& container,
                     const std::string& blob_name,
                     const std::string& payload,
                     const std::string& content_type = "application/octet-stream") {
        const std::string name = normalize_name(blob_name, container);
        if (payload.size() > kMaxBlobBytes) {
            throw std::invalid_argument(container + "/" + name + " would exceed the tiny-checkpoint size cap");
        }
        ensure_container(container);
        exchange("PUT", "/" + container + "/" + name, {}, payload, content_type);
    }

    std::string exchange(const std::string& method,
                         const std::string& path,
                         const std::vector<QueryItem>& query,
                         const std::string& body,
                         const std::string& content_type) {
        const std::string stamp = utc_stamp();
        std::map<std::string, std::string> ms_headers;
        ms_headers["x-ms-date"] = stamp;
        ms_headers["x-ms-version"] = "2020-10-02";
        if (method == "PUT" && query.empty()) {
            ms_headers["x-ms-blob-type"] = "BlockBlob";
        }
        const std::string quoted_path = url_quote(path);
        const std::string resource = canonical_resource(quoted_path, query);
        const std::string canonical = canonical_headers(ms_headers);
        const std::string length = body.empty() ? "" : std::to_string(body.size());
        const std::string signed_text = string_to_sign(method, length, content_type, canonical, resource);
        const std::string signature = base64_encode(hmac_sha256(base64_decode(account_.key), signed_text));
        const std::string object_name = account_.base_path + quoted_path + query_suffix(query);
        const std::string response = send(method, object_name, ms_headers, content_type, signature, body);
        return response;
    }

    std::string sha256_hex16(const std::string& payload) const {
        const std::vector<unsigned char> digest = sha256_bytes(payload);
        static const char* hex = "0123456789abcdef";
        std::string text;
        text.resize(16);
        for (std::size_t index = 0; index < 8; ++index) {
            text[index * 2] = hex[digest[index] >> 4];
            text[index * 2 + 1] = hex[digest[index] & 0x0f];
        }
        return text;
    }

private:
    struct Account {
        std::string name;
        std::string key;
        std::string host;
        int port = 10000;
        bool https = false;
        std::string base_path;
    };

    Account account_;

    static std::string resolve_connection(std::string raw) {
        if (raw.empty()) {
            const char* env = std::getenv("BLOB_STORAGE_CONNECTION");
            raw = env == nullptr ? "UseDevelopmentStorage=true" : env;
        }
        if (raw == "UseDevelopmentStorage=true") {
            return "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
                   "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
                   "K1SZFPTOtr/KBHBeksoGMGw==;"
                   "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;";
        }
        return raw;
    }

    static Account parse_connection(const std::string& raw) {
        std::map<std::string, std::string> parts;
        std::size_t start = 0;
        while (start < raw.size()) {
            const std::size_t end = raw.find(';', start) == std::string::npos ? raw.size() : raw.find(';', start);
            const std::size_t eq = raw.find('=', start);
            if (eq != std::string::npos && eq > start && eq < end) {
                parts[raw.substr(start, eq - start)] = raw.substr(eq + 1, end - (eq + 1));
            }
            if (end == raw.size()) {
                break;
            }
            start = end + 1;
        }
        Account account;
        account.name = parts["AccountName"];
        account.key = parts["AccountKey"];
        const std::string endpoint = parts["BlobEndpoint"];
        if (account.name.empty() || account.key.empty() || endpoint.empty()) {
            throw std::invalid_argument(
                "BLOB_STORAGE_CONNECTION must include AccountName, AccountKey, and BlobEndpoint");
        }
        const bool https = endpoint.rfind("https://", 0) == 0;
        const bool http = endpoint.rfind("http://", 0) == 0;
        if (!https && !http) {
            throw std::invalid_argument("BlobEndpoint must be an http URL");
        }
        const std::size_t scheme = endpoint.find("://");
        const std::size_t host_begin = scheme + 3;
        const std::size_t path_begin = endpoint.find('/', host_begin);
        const std::string host_port = endpoint.substr(
            host_begin, path_begin == std::string::npos ? std::string::npos : path_begin - host_begin);
        const std::size_t colon = host_port.rfind(':');
        account.https = https;
        if (colon == std::string::npos) {
            account.host = host_port;
            account.port = https ? 443 : 80;
        } else {
            account.host = host_port.substr(0, colon);
            account.port = std::stoi(host_port.substr(colon + 1));
        }
        std::string path = path_begin == std::string::npos ? "" : endpoint.substr(path_begin);
        while (!path.empty() && path.back() == '/') {
            path.pop_back();
        }
        account.base_path = path;
        return account;
    }

    static std::string normalize_name(std::string name, const std::string& container) {
        std::size_t cursor = 0;
        while (cursor < name.size() && std::isspace(static_cast<unsigned char>(name[cursor]))) {
            ++cursor;
        }
        std::size_t end = name.size();
        while (end > cursor && std::isspace(static_cast<unsigned char>(name[end - 1]))) {
            --end;
        }
        name = name.substr(cursor, end - cursor);
        for (char& character : name) {
            if (character == '\\') {
                character = '/';
            }
        }
        while (!name.empty() && name.front() == '/') {
            name.erase(name.begin());
        }
        const std::string prefix = container + "/";
        if (name.rfind(prefix, 0) == 0) {
            name = name.substr(prefix.size());
        }
        if (name.empty() || name.find('/') != std::string::npos || name.find("..") != std::string::npos) {
            throw std::invalid_argument("blob name must be a single file, optionally prefixed by its container");
        }
        return name;
    }

    void ensure_container(const std::string& container) {
        exchange("PUT", "/" + container, {QueryItem{"restype", "container"}}, "", "");
    }

    std::string canonical_headers(const std::map<std::string, std::string>& headers) const {
        std::string text;
        for (const auto& item : headers) {
            text += item.first;
            text.push_back(':');
            text += item.second;
            text.push_back('\n');
        }
        return text;
    }

    std::string canonical_resource(const std::string& quoted_path, const std::vector<QueryItem>& query) const {
        std::string resource = "/" + account_.name + account_.base_path + quoted_path;
        std::vector<QueryItem> ordered = query;
        std::sort(ordered.begin(), ordered.end(), [](const QueryItem& left, const QueryItem& right) {
            return left.key < right.key;
        });
        for (const QueryItem& item : ordered) {
            resource.push_back('\n');
            for (char character : item.key) {
                resource.push_back(character >= 'A' && character <= 'Z'
                                       ? static_cast<char>(character - 'A' + 'a')
                                       : character);
            }
            resource.push_back(':');
            resource += item.value;
        }
        return resource;
    }

    static std::string query_suffix(const std::vector<QueryItem>& query) {
        std::string text;
        for (const QueryItem& item : query) {
            text.push_back(text.empty() ? '?' : '&');
            text += url_quote(item.key);
            text.push_back('=');
            text += url_quote(item.value);
        }
        return text;
    }

    static std::string string_to_sign(const std::string& method,
                                      const std::string& length,
                                      const std::string& content_type,
                                      const std::string& canonical_headers,
                                      const std::string& resource) {
        return method + "\n\n\n" + length + "\n\n" + content_type + "\n\n\n\n\n\n\n" + canonical_headers + resource;
    }

    static std::string utc_stamp() {
        const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm fields{};
#ifdef _WIN32
        gmtime_s(&fields, &now);
#else
        gmtime_r(&now, &fields);
#endif
        static const char* days[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
        static const char* months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
        char buffer[64];
        std::snprintf(buffer,
                      sizeof(buffer),
                      "%s, %02d %s %04d %02d:%02d:%02d GMT",
                      days[fields.tm_wday],
                      fields.tm_mday,
                      months[fields.tm_mon],
                      fields.tm_year + 1900,
                      fields.tm_hour,
                      fields.tm_min,
                      fields.tm_sec);
        return buffer;
    }

    static std::string url_quote(const std::string& raw) {
        static const char* hex = "0123456789ABCDEF";
        std::string out;
        out.reserve(raw.size());
        for (unsigned char character : raw) {
            const bool plain = (character >= 'A' && character <= 'Z') || (character >= 'a' && character <= 'z') ||
                               (character >= '0' && character <= '9') || character == '-' || character == '_' ||
                               character == '.' || character == '~' || character == '/';
            if (plain) {
                out.push_back(static_cast<char>(character));
            } else {
                out.push_back('%');
                out.push_back(hex[character >> 4]);
                out.push_back(hex[character & 0x0f]);
            }
        }
        return out;
    }

    static std::string base64_encode(const std::vector<unsigned char>& raw) {
#ifdef _WIN32
        DWORD size = 0;
        if (!CryptBinaryToStringA(raw.data(),
                                  static_cast<DWORD>(raw.size()),
                                  CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF,
                                  nullptr,
                                  &size)) {
            throw std::runtime_error("base64 encode failed");
        }
        std::string text(size, '\0');
        if (!CryptBinaryToStringA(raw.data(),
                                  static_cast<DWORD>(raw.size()),
                                  CRYPT_STRING_BASE64 | CRYPT_STRING_NOCRLF,
                                  text.data(),
                                  &size)) {
            throw std::runtime_error("base64 encode failed");
        }
        while (!text.empty() && (text.back() == '\0' || text.back() == '\n' || text.back() == '\r')) {
            text.pop_back();
        }
        return text;
#else
        static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string text;
        unsigned int accumulator = 0;
        int bits = 0;
        for (unsigned char byte : raw) {
            accumulator = (accumulator << 8) | byte;
            bits += 8;
            while (bits >= 6) {
                bits -= 6;
                text.push_back(alphabet[(accumulator >> bits) & 0x3f]);
            }
        }
        if (bits > 0) {
            text.push_back(alphabet[(accumulator << (6 - bits)) & 0x3f]);
        }
        while (text.size() % 4 != 0) {
            text.push_back('=');
        }
        return text;
#endif
    }

    static std::vector<unsigned char> base64_decode(const std::string& text) {
#ifdef _WIN32
        DWORD size = 0;
        if (!CryptStringToBinaryA(text.c_str(), static_cast<DWORD>(text.size()), CRYPT_STRING_BASE64, nullptr, &size, nullptr, nullptr)) {
            throw std::runtime_error("account key is not valid base64");
        }
        std::vector<unsigned char> raw(size);
        if (!CryptStringToBinaryA(text.c_str(),
                                  static_cast<DWORD>(text.size()),
                                  CRYPT_STRING_BASE64,
                                  raw.data(),
                                  &size,
                                  nullptr,
                                  nullptr)) {
            throw std::runtime_error("account key is not valid base64");
        }
        raw.resize(size);
        return raw;
#else
        std::vector<int> values;
        for (unsigned char character : text) {
            if (character == '=' || std::isspace(character)) {
                continue;
            }
            int value = -1;
            if (character >= 'A' && character <= 'Z') {
                value = character - 'A';
            } else if (character >= 'a' && character <= 'z') {
                value = character - 'a' + 26;
            } else if (character >= '0' && character <= '9') {
                value = character - '0' + 52;
            } else if (character == '+') {
                value = 62;
            } else if (character == '/') {
                value = 63;
            }
            if (value < 0) {
                throw std::runtime_error("account key is not valid base64");
            }
            values.push_back(value);
        }
        std::vector<unsigned char> raw;
        for (std::size_t index = 0; index + 1 < values.size(); index += 4) {
            const int b0 = values[index];
            const int b1 = values[index + 1];
            const int b2 = index + 2 < values.size() ? values[index + 2] : 0;
            const int b3 = index + 3 < values.size() ? values[index + 3] : 0;
            raw.push_back(static_cast<unsigned char>((b0 << 2) | (b1 >> 4)));
            if (index + 2 < values.size()) {
                raw.push_back(static_cast<unsigned char>(((b1 & 0x0f) << 4) | (b2 >> 2)));
            }
            if (index + 3 < values.size()) {
                raw.push_back(static_cast<unsigned char>(((b2 & 0x03) << 6) | b3));
            }
        }
        return raw;
#endif
    }

    static std::vector<unsigned char> sha256_bytes(const std::string& payload) {
#ifdef _WIN32
        return bcrypt_sha256({}, payload, false);
#else
        return sha256_raw(payload);
#endif
    }

    static std::vector<unsigned char> hmac_sha256(const std::vector<unsigned char>& key, const std::string& message) {
#ifdef _WIN32
        return bcrypt_sha256(key, message, true);
#else
        return portable_hmac(key, message);
#endif
    }

    static std::vector<unsigned char> hmac_sha256(const std::string& key, const std::string& message) {
        return hmac_sha256(std::vector<unsigned char>(key.begin(), key.end()), message);
    }

#ifdef _WIN32
    static std::vector<unsigned char> bcrypt_sha256(const std::vector<unsigned char>& key,
                                                    const std::string& message,
                                                    bool hmac) {
        BCRYPT_ALG_HANDLE algorithm = nullptr;
        const NTSTATUS opened = BCryptOpenAlgorithmProvider(
            &algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, hmac ? BCRYPT_ALG_HANDLE_HMAC_FLAG : 0);
        if (opened != 0) {
            throw std::runtime_error("SHA256 provider failed");
        }
        DWORD object_size = 0;
        DWORD ignored = 0;
        BCryptGetProperty(algorithm,
                          BCRYPT_OBJECT_LENGTH,
                          reinterpret_cast<PUCHAR>(&object_size),
                          sizeof(object_size),
                          &ignored,
                          0);
        DWORD hash_size = 0;
        BCryptGetProperty(algorithm,
                          BCRYPT_HASH_LENGTH,
                          reinterpret_cast<PUCHAR>(&hash_size),
                          sizeof(hash_size),
                          &ignored,
                          0);
        std::vector<unsigned char> object(object_size);
        BCRYPT_HASH_HANDLE hash = nullptr;
        const NTSTATUS created = BCryptCreateHash(algorithm,
                                                  &hash,
                                                  object.data(),
                                                  object_size,
                                                  hmac ? const_cast<PUCHAR>(key.data()) : nullptr,
                                                  hmac ? static_cast<ULONG>(key.size()) : 0,
                                                  0);
        if (created != 0) {
            BCryptCloseAlgorithmProvider(algorithm, 0);
            throw std::runtime_error("SHA256 hash failed");
        }
        if (!message.empty()) {
            BCryptHashData(hash,
                           reinterpret_cast<PUCHAR>(const_cast<char*>(message.data())),
                           static_cast<ULONG>(message.size()),
                           0);
        }
        std::vector<unsigned char> digest(hash_size);
        const NTSTATUS finished = BCryptFinishHash(hash, digest.data(), hash_size, 0);
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        if (finished != 0) {
            throw std::runtime_error("SHA256 finish failed");
        }
        return digest;
    }

    std::string send(const std::string& method,
                     const std::string& object_name,
                     const std::map<std::string, std::string>& ms_headers,
                     const std::string& content_type,
                     const std::string& signature,
                     const std::string& body) const {
        if (account_.https) {
            throw std::runtime_error("native blob client currently speaks the local HTTP Azurite endpoint");
        }
        HINTERNET session = WinHttpOpen(L"SteveJob-NativeEngine/1.0",
                                        WINHTTP_ACCESS_TYPE_NO_PROXY,
                                        WINHTTP_NO_PROXY_NAME,
                                        WINHTTP_NO_PROXY_BYPASS,
                                        0);
        if (session == nullptr) {
            throw std::runtime_error("WinHttpOpen failed");
        }
        WinHttpSetTimeouts(session, 30000, 30000, 30000, 30000);
        const std::wstring host(account_.host.begin(), account_.host.end());
        HINTERNET connection = WinHttpConnect(session, host.c_str(), static_cast<INTERNET_PORT>(account_.port), 0);
        if (connection == nullptr) {
            WinHttpCloseHandle(session);
            throw std::runtime_error("WinHttpConnect failed");
        }
        const std::wstring verb(method.begin(), method.end());
        const std::wstring object(object_name.begin(), object_name.end());
        HINTERNET request = WinHttpOpenRequest(connection,
                                               verb.c_str(),
                                               object.c_str(),
                                               nullptr,
                                               WINHTTP_NO_REFERER,
                                               WINHTTP_DEFAULT_ACCEPT_TYPES,
                                               0);
        if (request == nullptr) {
            WinHttpCloseHandle(connection);
            WinHttpCloseHandle(session);
            throw std::runtime_error("WinHttpOpenRequest failed");
        }
        std::wstring headers;
        for (const auto& item : ms_headers) {
            headers += std::wstring(item.first.begin(), item.first.end());
            headers += L": ";
            headers += std::wstring(item.second.begin(), item.second.end());
            headers += L"\r\n";
        }
        if (!content_type.empty()) {
            headers += L"Content-Type: ";
            headers += std::wstring(content_type.begin(), content_type.end());
            headers += L"\r\n";
        }
        headers += L"Authorization: SharedKey ";
        headers += std::wstring(account_.name.begin(), account_.name.end());
        headers += L":";
        headers += std::wstring(signature.begin(), signature.end());
        headers += L"\r\n";
        const DWORD body_size = static_cast<DWORD>(body.size());
        const BOOL sent = WinHttpSendRequest(request,
                                              headers.c_str(),
                                              static_cast<DWORD>(headers.size()),
                                              body.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(body.data()),
                                              body_size,
                                              body_size,
                                              0);
        BOOL received = FALSE;
        if (sent) {
            received = WinHttpReceiveResponse(request, nullptr);
        }
        std::string payload;
        DWORD status = 0;
        if (received) {
            DWORD status_size = sizeof(status);
            WinHttpQueryHeaders(request,
                                WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                                WINHTTP_HEADER_NAME_BY_INDEX,
                                &status,
                                &status_size,
                                WINHTTP_NO_HEADER_INDEX);
            DWORD available = 0;
            do {
                available = 0;
                if (!WinHttpQueryDataAvailable(request, &available) || available == 0) {
                    break;
                }
                if (payload.size() + available > kMaxBlobBytes) {
                    WinHttpCloseHandle(request);
                    WinHttpCloseHandle(connection);
                    WinHttpCloseHandle(session);
                    throw std::runtime_error("blob response is too large");
                }
                const std::size_t offset = payload.size();
                payload.resize(offset + available);
                DWORD downloaded = 0;
                if (!WinHttpReadData(request, payload.data() + offset, available, &downloaded)) {
                    break;
                }
                payload.resize(offset + downloaded);
            } while (available > 0);
        }
        WinHttpCloseHandle(request);
        WinHttpCloseHandle(connection);
        WinHttpCloseHandle(session);
        if (!sent || !received) {
            throw std::runtime_error("blob " + method + " request failed before a response");
        }
        if (status == 409 && payload.find("ContainerAlreadyExists") != std::string::npos) {
            return "";
        }
        if (status < 200 || status >= 300) {
            if (payload.size() > 400) {
                payload.resize(400);
            }
            throw std::runtime_error("blob " + method + " " + object_name + " failed with HTTP " +
                                     std::to_string(status) + ": " + payload);
        }
        return payload;
    }
#else
    static std::vector<unsigned char> portable_hmac(const std::vector<unsigned char>& key, const std::string& message) {
        std::vector<unsigned char> hashed_key = key;
        if (hashed_key.size() > 64) {
            const std::string key_text(hashed_key.begin(), hashed_key.end());
            hashed_key = sha256_raw(key_text);
        }
        hashed_key.resize(64, 0);
        std::string inner(64, '\0');
        std::string outer(64, '\0');
        for (std::size_t index = 0; index < 64; ++index) {
            inner[index] = static_cast<char>(hashed_key[index] ^ 0x36);
            outer[index] = static_cast<char>(hashed_key[index] ^ 0x5c);
        }
        inner += message;
        const std::vector<unsigned char> inner_hash = sha256_raw(inner);
        outer.append(reinterpret_cast<const char*>(inner_hash.data()), inner_hash.size());
        return sha256_raw(outer);
    }

    static std::vector<unsigned char> sha256_raw(const std::string& payload) {
        throw std::runtime_error("POSIX SHA256 fallback is not linked on this build");
        (void)payload;
    }

    std::string send(const std::string&,
                     const std::string&,
                     const std::map<std::string, std::string>&,
                     const std::string&,
                     const std::string&,
                     const std::string&) const {
        throw std::runtime_error("native blob sockets are built with WinHTTP on Windows");
    }
#endif
};

}  // namespace stevejob::storage
