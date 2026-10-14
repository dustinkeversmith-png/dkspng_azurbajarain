#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace forward_analysis {

// RFC 4180 CSV cells: values containing a comma, quote or newline are quoted with inner quotes doubled.
struct CsvFieldFormatter {
    static std::string escape(const std::string& value) {
        if (value.find_first_of(",\"\r\n") == std::string::npos) {
            return value;
        }
        std::string out = "\"";
        for (char c : value) {
            out += c == '"' ? std::string("\"\"") : std::string(1, c);
        }
        return out + "\"";
    }

    static std::string number(double value) {
        std::ostringstream out;
        out << std::setprecision(8) << value;
        return out.str();
    }

    static std::string row(const std::vector<std::string>& cells) {
        std::string out;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            out += (i > 0 ? "," : "") + escape(cells[i]);
        }
        return out;
    }
};

}  // namespace forward_analysis
