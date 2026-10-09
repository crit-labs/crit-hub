// C ABI for the Python chat adapter (ctypes). One resident engine per handle.
#include "engine.h"
#include <cstring>
#include <fstream>
#include <sstream>

namespace {
void put_error(char* err, int len, const std::string& what) {
    if(!err || len <= 0) return;
    std::strncpy(err, what.c_str(), std::size_t(len - 1));
    err[len - 1] = 0;
}
}

extern "C" {
typedef int (*q38_token_cb)(int token, void* user);

void* q38_open(const char* model_dir, const char* config_json, char* err, int errlen) {
    try {
        auto cfg = q38::Config::parse(config_json ? config_json : "", model_dir);
        auto rt = q38::create_runtime(cfg);
        return rt.release();
    } catch(const std::exception& e) {
        put_error(err, errlen, e.what());
        return nullptr;
    }
}

void* q38_open_files(const char* main_file, const char* mtp_file, const char* config_json, char* err, int errlen) {
    try {
        auto cfg = q38::Config::parse(config_json ? config_json : "", ".");
        cfg.main_file = main_file;
        cfg.mtp_file = mtp_file;
        auto rt = q38::create_runtime(cfg);
        return rt.release();
    } catch(const std::exception& e) {
        put_error(err, errlen, e.what());
        return nullptr;
    }
}

// stats: [0] prompt tokens, [1] prefill tokens, [2] prefill ms, [3] decode tokens, [4] decode ms,
//        [5] ttft ms, [6] rounds, [7] finish (0 length, 1 stop, 2 cancelled, 3 failed)
int q38_generate(void* handle, const int* prompt, int n, int max_out, const int* stops, int nstops, int default_stops,
                 q38_token_cb cb, void* user, int* out, int out_cap, double* stats, char* err, int errlen) {
    auto* rt = static_cast<q38::Runtime*>(handle);
    try {
        if(!rt || n <= 0 || max_out <= 0 || out_cap < max_out) throw std::invalid_argument("Invalid generation arguments");
        crit::core::RequestOptions options;
        if(!default_stops) options.stop_tokens = std::vector<crit::core::Token>(stops, stops + nstops);
        const auto t0 = std::chrono::steady_clock::now();
        auto h = rt->engine->submit(std::vector<crit::core::Token>(prompt, prompt + n), std::uint32_t(max_out), std::move(options));
        std::chrono::steady_clock::time_point first{};
        int count = 0;
        bool cancelled = false;
        auto result = h.wait([&](crit::core::Token t) {
            if(count == 0) first = std::chrono::steady_clock::now();
            ++count;
            if(cb && !cancelled && cb(t, user)) { cancelled = true; h.cancel(); }
        });
        const auto t1 = std::chrono::steady_clock::now();
        const int m = int(std::min<std::size_t>(result.tokens.size(), std::size_t(out_cap)));
        std::copy(result.tokens.begin(), result.tokens.begin() + m, out);
        const auto& st = rt->core->stats;
        const double decode_ms = std::chrono::duration<double, std::milli>(t1 - st.prefill_end).count();
        if(stats) {
            stats[0] = n;
            stats[1] = st.prefill_tokens;
            stats[2] = st.prefill_ms;
            stats[3] = m;
            stats[4] = m ? decode_ms : 0;
            stats[5] = m ? std::chrono::duration<double, std::milli>(first - t0).count() : 0;
            stats[6] = double(st.rounds);
            stats[7] = result.finish == crit::core::Finish::length ? 0 : result.finish == crit::core::Finish::stop ? 1
                     : result.finish == crit::core::Finish::cancelled ? 2 : 3;
        }
        return m;
    } catch(const std::exception& e) {
        put_error(err, errlen, e.what());
        return -1;
    }
}

void q38_close(void* handle) {
    delete static_cast<q38::Runtime*>(handle);
}
}
