// Resident benchmark and check harness for the clean-room engine.
//   q38bench --model DIR [--config JSON] [--audit] [--nll REF.json] [--functional CASE.json]
//            [--bench CASE.json ...] [--warmup N] [--reps N] [--out REPORT.json] [--tokens-out FILE]
#include "engine.h"
#include "forge/device_round.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/resource.h>
#include <thread>

using namespace q38;
using clk = std::chrono::steady_clock;
namespace {
std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if(!f) throw std::runtime_error("Cannot read " + p);
    std::stringstream s; s << f.rdbuf(); return s.str();
}
std::vector<int> ints(const Json& a) { std::vector<int> v; for(auto& x : a.arr) v.push_back(int(x.integer())); return v; }
double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
std::size_t free_vram() { std::size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return f; }
std::size_t total_vram() { std::size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return t; }
std::size_t avail_ram() {
    std::ifstream m("/proc/meminfo"); std::string k; std::size_t v; std::string unit;
    while(m >> k >> v >> unit) if(k == "MemAvailable:") return v * 1024; return 0;
}
std::size_t peak_rss() { rusage u{}; getrusage(RUSAGE_SELF, &u); return std::size_t(u.ru_maxrss) * 1024; }
std::string jarr(const std::vector<double>& v) {
    std::ostringstream o; o << std::setprecision(10) << "[";
    for(std::size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]"; return o.str();
}
std::string jints(const std::vector<int>& v) {
    std::ostringstream o; o << "[";
    for(std::size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << "]"; return o.str();
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

struct Gen { std::vector<int> tokens; crit::core::Finish finish; double prefill_ms, decode_ms, ttft_ms, request_ms; std::uint32_t prefill_tokens; std::uint64_t rounds; double cancel_ms = 0; };
Gen generate(Runtime& rt, const std::vector<int>& prompt, int out, std::optional<std::vector<int>> stops, int cancel_after = -1) {
    crit::core::RequestOptions o;
    if(stops) o.stop_tokens = *stops;
    const auto t0 = clk::now();
    auto h = rt.engine->submit(prompt, std::uint32_t(out), std::move(o));
    clk::time_point first{}, cancel_at{};
    int n = 0;
    auto r = h.wait([&](int) {
        if(n == 0) first = clk::now();
        ++n;
        if(cancel_after >= 0 && n == cancel_after) { cancel_at = clk::now(); h.cancel(); }
    });
    const auto t1 = clk::now();
    const auto& st = rt.core->stats;
    Gen g{r.tokens, r.finish, st.prefill_ms, ms(st.prefill_end, t1), ms(t0, first), ms(t0, t1), st.prefill_tokens, st.rounds};
    if(cancel_after >= 0) g.cancel_ms = ms(cancel_at, t1);
    return g;
}
}

int main(int argc, char** argv) {
    try {
        std::string model_dir, config, nll_ref, functional, out_path, tokens_out;
        std::vector<std::string> cases;
        bool audit = false, nll_decode = false;
        std::vector<int> profile_cols, micro_cols;
        int warmup = 2, reps = 5;
        for(int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() { if(i + 1 >= argc) throw std::invalid_argument("Missing value for " + a); return std::string(argv[++i]); };
            if(a == "--model") model_dir = next();
            else if(a == "--config") config = next();
            else if(a == "--nll") nll_ref = next();
            else if(a == "--audit") audit = true;
            else if(a == "--nll-decode") nll_decode = true;
            else if(a == "--functional") functional = next();
            else if(a == "--bench") { while(i + 1 < argc && argv[i + 1][0] != '-') cases.push_back(argv[++i]); }
            else if(a == "--warmup") warmup = std::stoi(next());
            else if(a == "--reps") reps = std::stoi(next());
            else if(a == "--out") out_path = next();
            else if(a == "--tokens-out") tokens_out = next();
            else if(a == "--profile") profile_cols.push_back(std::stoi(next()));
            else if(a == "--micro") micro_cols.push_back(std::stoi(next()));
            else throw std::invalid_argument("Unknown argument " + a);
        }
        const auto cfg = Config::parse(config, model_dir);
        std::ostringstream rep;
        rep << std::setprecision(10) << "{\"config\":" << (config.empty() ? "{}" : config);
        const std::size_t vram_before = free_vram(), ram_before = avail_ram();
        const auto tl0 = clk::now();
        auto rt = create_runtime(cfg);
        // Force factory construction (weights load) through a tiny warm request.
        const auto tl1 = clk::now();
        auto& m = rt->core->model();
        rep << ",\"load_ms\":" << ms(tl0, tl1) << ",\"weight_bytes\":" << m.weight_bytes() << ",\"state_bytes\":" << m.state_bytes()
            << ",\"free_vram_before_load\":" << vram_before << ",\"free_vram_after_load\":" << free_vram()
            << ",\"total_vram\":" << total_vram() << ",\"ram_available_before\":" << ram_before;
        std::cerr << "loaded in " << ms(tl0, tl1) << " ms, weights " << m.weight_bytes() / 1e9 << " GB, state " << m.state_bytes() / 1e9
                  << " GB, free VRAM " << free_vram() / 1e9 << " GB\n";
        for(int T : profile_cols) std::cerr << m.profile(T, 5);
        for(int T : micro_cols) std::cerr << "T=" << T << "\n" << m.micro(T);
        if(audit) {
            std::string report;
            const auto bad = m.audit(report);
            std::cerr << report;
            auto summary = report.substr(report.rfind("tensors"));
            while(!summary.empty() && summary.back() == '\n') summary.pop_back();
            rep << ",\"audit\":{\"mismatched_slices\":" << bad << ",\"summary\":\"" << summary << "\"}";
        }
        if(!nll_ref.empty()) {
            const auto ref = Json::parse(read_file(nll_ref));
            rep << ",\"nll\":{";
            bool firstc = true;
            for(auto& [name, r] : ref["results"].obj) {
                const auto ids = ints(r["input_ids"]);
                const int first = int(r["first_target"].integer());
                const auto t0 = clk::now();
                crit::core::Result res;
                if(nll_decode) res.scores = m.score_decode(ids, first);
                else res = rt->engine->submit_score(ids, std::uint32_t(first)).wait();
                const auto t1 = clk::now();
                double sum = 0, maxdiff = 0;
                const auto& lp = r["log_probabilities"].arr;
                for(std::size_t i = 0; i < res.scores.size(); ++i) {
                    sum -= res.scores[i];
                    maxdiff = std::max(maxdiff, std::abs(double(res.scores[i]) - lp.at(i).num));
                }
                const double nll = sum / res.scores.size(), refnll = r["mean_nll"].num;
                std::cerr << name << ": engine NLL " << nll << " reference " << refnll << " delta " << nll - refnll
                          << " max |dlogp| " << maxdiff << " (" << res.scores.size() << " targets, " << ms(t0, t1) << " ms)\n";
                rep << (firstc ? "" : ",") << "\"" << name << "\":{\"targets\":" << res.scores.size() << ",\"engine_mean_nll\":" << nll
                    << ",\"reference_mean_nll\":" << refnll << ",\"delta\":" << nll - refnll << ",\"max_abs_logprob_diff\":" << maxdiff
                    << ",\"passed\":" << (std::isfinite(nll) && nll - refnll <= 0.05 ? "true" : "false") << "}";
                firstc = false;
            }
            rep << "}";
        }
        if(!functional.empty()) {
            const auto c = Json::parse(read_file(functional));
            const auto prompt = ints(c["input_ids"]);
            const std::vector<int> none;
            auto a = generate(*rt, prompt, 64, none);
            auto b = generate(*rt, prompt, 64, none);
            const bool repeat = a.tokens == b.tokens && a.tokens.size() == 64;
            const int stop = a.tokens.at(20);
            auto s = generate(*rt, prompt, 64, std::vector<int>{stop});
            const auto cut = std::size_t(std::find(a.tokens.begin(), a.tokens.end(), stop) - a.tokens.begin()) + 1;
            const bool stopped = s.finish == crit::core::Finish::stop && s.tokens == std::vector<int>(a.tokens.begin(), a.tokens.begin() + cut);
            auto x = generate(*rt, prompt, 512, none, 16);
            const bool cancelled = x.finish == crit::core::Finish::cancelled && x.cancel_ms < 1000.0;
            auto after = generate(*rt, prompt, 64, none);
            const bool after_ok = after.tokens == a.tokens;
            bool overflow_rejected = false;
            try { generate(*rt, std::vector<int>(std::size_t(cfg.context), 198), 4, none); } catch(const std::exception&) { overflow_rejected = true; }
            auto again = generate(*rt, prompt, 64, none);
            const bool after_reject = again.tokens == a.tokens;
            std::cerr << "functional: repeat " << repeat << " stop " << stopped << " (cut " << cut << ") cancel " << cancelled
                      << " (" << x.cancel_ms << " ms, " << x.tokens.size() << " tokens) after-cancel " << after_ok
                      << " overflow-rejected " << overflow_rejected << " after-reject " << after_reject << "\n";
            rep << ",\"functional\":{\"repeat_identical\":" << repeat << ",\"stop_trim\":" << stopped << ",\"stop_cut\":" << cut
                << ",\"cancelled\":" << cancelled << ",\"cancel_ms\":" << x.cancel_ms << ",\"cancel_tokens\":" << x.tokens.size()
                << ",\"after_cancel_identical\":" << after_ok << ",\"overflow_rejected\":" << overflow_rejected
                << ",\"after_reject_identical\":" << after_reject << ",\"reference_tokens\":" << jints(a.tokens) << "}";
        }
        if(!cases.empty()) {
            rep << ",\"benchmarks\":[";
            std::ofstream tok_file;
            if(!tokens_out.empty()) tok_file.open(tokens_out);
            for(std::size_t ci = 0; ci < cases.size(); ++ci) {
                const auto c = Json::parse(read_file(cases[ci]));
                const auto prompt = ints(c["input_ids"]);
                const int out = int(c["output_tokens"].integer());
                std::vector<double> pre, dec, req, ttft, rounds, minfree, minram, rss;
                std::vector<int> last;
                bool identical = true;
                for(int r = 0; r < warmup + reps; ++r) {
                    const std::size_t f0 = free_vram(), r0 = avail_ram();
                    auto g = generate(*rt, prompt, out, std::vector<int>{});
                    const std::size_t f1 = free_vram(), r1 = avail_ram();
                    if(int(g.tokens.size()) != out) throw std::runtime_error("Benchmark delivered " + std::to_string(g.tokens.size()) + " tokens");
                    if(!last.empty() && g.tokens != last) identical = false;
                    last = g.tokens;
                    if(r < warmup) continue;
                    pre.push_back(g.prefill_tokens / g.prefill_ms * 1000.0);
                    dec.push_back(out / g.decode_ms * 1000.0);
                    req.push_back(g.request_ms);
                    ttft.push_back(g.ttft_ms);
                    rounds.push_back(double(g.rounds));
                    minfree.push_back(double(std::min(f0, f1)));
                    minram.push_back(double(std::min(r0, r1)));
                    rss.push_back(double(peak_rss()));
                }
                std::cerr << c["case_id"].str << " (" << c["corpus"].str << "): decode " << median(dec) << " tok/s, prefill " << median(pre)
                          << " tok/s, ttft " << median(ttft) << " ms, tokens/round " << out / median(rounds) << ", reps " << jarr(dec) << "\n";
                if(tok_file) tok_file << c["case_id"].str << " " << jints(last) << "\n";
                rep << (ci ? "," : "") << "{\"case_id\":\"" << c["case_id"].str << "\",\"corpus\":\"" << c["corpus"].str
                    << "\",\"prompt_tokens\":" << prompt.size() << ",\"output_tokens\":" << out << ",\"warmup\":" << warmup
                    << ",\"decode_tps\":" << jarr(dec) << ",\"prefill_tps\":" << jarr(pre) << ",\"request_ms\":" << jarr(req)
                    << ",\"ttft_ms\":" << jarr(ttft) << ",\"rounds\":" << jarr(rounds) << ",\"min_free_vram\":" << jarr(minfree)
                    << ",\"min_avail_ram\":" << jarr(minram) << ",\"peak_rss\":" << jarr(rss) << ",\"repetitions_identical\":" << (identical ? "true" : "false")
                    << ",\"tokens\":" << jints(last) << "}";
            }
            rep << "]";
        }
        rep << ",\"peak_rss\":" << peak_rss() << ",\"total_vram_bytes\":" << total_vram() << ",\"free_vram_end\":" << free_vram() << "}";
        if(!out_path.empty()) { std::ofstream f(out_path); f << rep.str() << "\n"; }
        rt.reset();
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
