#pragma once
// Minimal JSON reader for Safetensors headers and engine configuration.
// Objects, arrays, strings (with escapes), numbers, true/false/null.
#include <cctype>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace q38 {
struct Json {
    enum class Kind { null, boolean, number, string, array, object } kind = Kind::null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    const Json& operator[](const std::string& key) const {
        auto it = obj.find(key);
        if(kind != Kind::object || it == obj.end()) throw std::runtime_error("JSON key missing: " + key);
        return it->second;
    }
    bool has(const std::string& key) const { return kind == Kind::object && obj.count(key); }
    std::int64_t integer() const { return static_cast<std::int64_t>(num); }

    static Json parse(std::string_view text) {
        std::size_t i = 0;
        Json value = parse_value(text, i);
        skip(text, i);
        if(i != text.size()) throw std::runtime_error("Trailing JSON data");
        return value;
    }
private:
    static void skip(std::string_view t, std::size_t& i) {
        while(i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) ++i;
    }
    static Json parse_value(std::string_view t, std::size_t& i) {
        skip(t, i);
        if(i >= t.size()) throw std::runtime_error("Unexpected end of JSON");
        Json v;
        const char c = t[i];
        if(c == '{') {
            v.kind = Kind::object; ++i; skip(t, i);
            if(i < t.size() && t[i] == '}') { ++i; return v; }
            for(;;) {
                skip(t, i);
                Json key = parse_value(t, i);
                if(key.kind != Kind::string) throw std::runtime_error("JSON object key must be a string");
                skip(t, i);
                if(i >= t.size() || t[i] != ':') throw std::runtime_error("Expected ':' in JSON");
                ++i;
                Json item = parse_value(t, i);
                if(!v.obj.emplace(key.str, std::move(item)).second) throw std::runtime_error("Duplicate JSON key " + key.str);
                skip(t, i);
                if(i < t.size() && t[i] == ',') { ++i; continue; }
                if(i < t.size() && t[i] == '}') { ++i; return v; }
                throw std::runtime_error("Expected ',' or '}' in JSON");
            }
        }
        if(c == '[') {
            v.kind = Kind::array; ++i; skip(t, i);
            if(i < t.size() && t[i] == ']') { ++i; return v; }
            for(;;) {
                v.arr.push_back(parse_value(t, i));
                skip(t, i);
                if(i < t.size() && t[i] == ',') { ++i; continue; }
                if(i < t.size() && t[i] == ']') { ++i; return v; }
                throw std::runtime_error("Expected ',' or ']' in JSON");
            }
        }
        if(c == '"') {
            v.kind = Kind::string; ++i;
            while(i < t.size() && t[i] != '"') {
                if(t[i] == '\\') {
                    ++i;
                    if(i >= t.size()) break;
                    const char e = t[i++];
                    switch(e) {
                    case 'n': v.str += '\n'; break; case 't': v.str += '\t'; break;
                    case 'r': v.str += '\r'; break; case 'b': v.str += '\b'; break;
                    case 'f': v.str += '\f'; break;
                    case 'u': {
                        if(i + 4 > t.size()) throw std::runtime_error("Bad JSON escape");
                        unsigned code = std::stoul(std::string(t.substr(i, 4)), nullptr, 16);
                        i += 4;
                        if(code < 0x80) v.str += char(code);
                        else if(code < 0x800) { v.str += char(0xC0 | (code >> 6)); v.str += char(0x80 | (code & 0x3F)); }
                        else { v.str += char(0xE0 | (code >> 12)); v.str += char(0x80 | ((code >> 6) & 0x3F)); v.str += char(0x80 | (code & 0x3F)); }
                        break;
                    }
                    default: v.str += e;
                    }
                } else v.str += t[i++];
            }
            if(i >= t.size()) throw std::runtime_error("Unterminated JSON string");
            ++i;
            return v;
        }
        if(t.substr(i, 4) == "true") { v.kind = Kind::boolean; v.b = true; i += 4; return v; }
        if(t.substr(i, 5) == "false") { v.kind = Kind::boolean; i += 5; return v; }
        if(t.substr(i, 4) == "null") { i += 4; return v; }
        std::size_t start = i;
        while(i < t.size() && (std::isdigit(static_cast<unsigned char>(t[i])) || t[i] == '-' || t[i] == '+' || t[i] == '.' || t[i] == 'e' || t[i] == 'E')) ++i;
        if(start == i) throw std::runtime_error("Unexpected JSON character");
        v.kind = Kind::number;
        v.num = std::stod(std::string(t.substr(start, i - start)));
        return v;
    }
};
}
