#pragma once
// Read-only memory-mapped Safetensors file with a parsed tensor directory.
#include "json.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace q38 {
struct TensorInfo {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::size_t begin = 0, end = 0; // absolute file offsets
    std::size_t bytes() const { return end - begin; }
};

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if(fd_ < 0) throw std::runtime_error("Cannot open " + path);
        struct stat st{};
        if(fstat(fd_, &st) != 0) throw std::runtime_error("Cannot stat " + path);
        size_ = static_cast<std::size_t>(st.st_size);
        base_ = static_cast<const std::byte*>(mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0));
        if(base_ == MAP_FAILED) throw std::runtime_error("Cannot map " + path);
        std::uint64_t header_bytes = 0;
        std::memcpy(&header_bytes, base_, 8);
        if(header_bytes + 8 > size_) throw std::runtime_error("Bad Safetensors header in " + path);
        auto header = Json::parse(std::string_view(reinterpret_cast<const char*>(base_ + 8), header_bytes));
        const std::size_t data = 8 + header_bytes;
        for(auto& [name, entry] : header.obj) {
            if(name == "__metadata__") continue;
            TensorInfo info;
            info.dtype = entry["dtype"].str;
            for(auto& d : entry["shape"].arr) info.shape.push_back(d.integer());
            info.begin = data + static_cast<std::size_t>(entry["data_offsets"].arr.at(0).integer());
            info.end = data + static_cast<std::size_t>(entry["data_offsets"].arr.at(1).integer());
            if(info.end < info.begin || info.end > size_) throw std::runtime_error("Tensor extent outside file: " + name);
            tensors_.emplace(name, std::move(info));
        }
    }
    ~SafetensorsFile() {
        if(base_ && base_ != MAP_FAILED) munmap(const_cast<std::byte*>(base_), size_);
        if(fd_ >= 0) ::close(fd_);
    }
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorInfo& info(const std::string& name) const {
        auto it = tensors_.find(name);
        if(it == tensors_.end()) throw std::runtime_error("Tensor missing: " + name);
        return it->second;
    }
    std::span<const std::byte> bytes(const std::string& name) const {
        const auto& t = info(name);
        return {base_ + t.begin, t.bytes()};
    }
    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    const std::string& path() const { return path_; }
private:
    std::string path_;
    int fd_ = -1;
    std::size_t size_ = 0;
    const std::byte* base_ = nullptr;
    std::map<std::string, TensorInfo> tensors_;
};
}
