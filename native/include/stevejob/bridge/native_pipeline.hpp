#pragma once

#include "stevejob/storage/json_text.hpp"
#include "stevejob/storage/native_blob_client.hpp"

#include <pybind11/pybind11.h>

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace stevejob::bridge {

class StorageExchange {
public:
    static py::object load(const std::string& raw) { return json_to_py(stevejob::storage::JsonText::parse(raw)); }

    static std::string dump(py::handle value) {
        if (!py::isinstance<py::dict>(value)) {
            throw std::invalid_argument("JSON payload must be an object");
        }
        std::string out;
        dump_json_into(out, value, 0);
        out.push_back('\n');
        return out;
    }

    static std::string checksum(const std::string& payload) {
        return stevejob::storage::NativeBlobClient("UseDevelopmentStorage=true").sha256_hex16(payload);
    }

    static std::string exchange(const std::string& connection,
                                const std::string& method,
                                const std::string& path,
                                const std::vector<stevejob::storage::QueryItem>& query,
                                const std::string& body,
                                const std::string& content_type) {
        return stevejob::storage::NativeBlobClient(connection).exchange(method, path, query, body, content_type);
    }

private:
    static py::object json_to_py(const stevejob::storage::JsonText& value) {
        using Kind = stevejob::storage::JsonText::Kind;
        if (value.kind == Kind::Null) {
            return py::none();
        }
        if (value.kind == Kind::Bool) {
            return py::bool_(value.boolean);
        }
        if (value.kind == Kind::Number) {
            if (value.number_is_int) {
                return py::int_(value.integer);
            }
            return py::float_(value.number);
        }
        if (value.kind == Kind::String) {
            return py::str(value.text);
        }
        if (value.kind == Kind::Array) {
            py::list items;
            for (const stevejob::storage::JsonText& item : value.array) {
                items.append(json_to_py(item));
            }
            return items;
        }
        py::dict mapping;
        for (const auto& item : value.object) {
            mapping[py::str(item.first)] = json_to_py(item.second);
        }
        return mapping;
    }

    static void append_json_string(std::string& out, const std::string& text) {
        out.push_back('"');
        for (unsigned char character : text) {
            switch (character) {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    if (character < 0x20) {
                        char buffer[8];
                        std::snprintf(buffer, sizeof(buffer), "\\u%04x", character);
                        out += buffer;
                    } else {
                        out.push_back(static_cast<char>(character));
                    }
            }
        }
        out.push_back('"');
    }

    static void dump_json_into(std::string& out, py::handle value, int indent) {
        const std::string pad(static_cast<std::size_t>(indent) * 2, ' ');
        const std::string nested(static_cast<std::size_t>(indent + 1) * 2, ' ');
        if (value.is_none()) {
            out += "null";
            return;
        }
        if (py::isinstance<py::bool_>(value)) {
            out += value.cast<bool>() ? "true" : "false";
            return;
        }
        if (py::isinstance<py::int_>(value)) {
            out += std::to_string(value.cast<long long>());
            return;
        }
        if (py::isinstance<py::float_>(value)) {
            std::ostringstream number;
            number.precision(17);
            number << value.cast<double>();
            out += number.str();
            return;
        }
        if (py::isinstance<py::str>(value)) {
            append_json_string(out, value.cast<std::string>());
            return;
        }
        if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
            py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
            if (py::len(sequence) == 0) {
                out += "[]";
                return;
            }
            out += "[\n";
            bool first = true;
            for (py::handle item : sequence) {
                if (!first) {
                    out += ",\n";
                }
                first = false;
                out += nested;
                dump_json_into(out, item, indent + 1);
            }
            out += "\n";
            out += pad;
            out += "]";
            return;
        }
        if (py::isinstance<py::dict>(value)) {
            py::dict mapping = py::reinterpret_borrow<py::dict>(value);
            if (py::len(mapping) == 0) {
                out += "{}";
                return;
            }
            out += "{\n";
            bool first = true;
            for (auto item : mapping) {
                if (!first) {
                    out += ",\n";
                }
                first = false;
                out += nested;
                append_json_string(out, py::cast<std::string>(py::str(item.first)));
                out += ": ";
                dump_json_into(out, py::reinterpret_borrow<py::object>(item.second), indent + 1);
            }
            out += "\n";
            out += pad;
            out += "}";
            return;
        }
        throw std::invalid_argument("JSON payload must contain objects, arrays, strings, and numbers");
    }
};

}  // namespace stevejob::bridge
