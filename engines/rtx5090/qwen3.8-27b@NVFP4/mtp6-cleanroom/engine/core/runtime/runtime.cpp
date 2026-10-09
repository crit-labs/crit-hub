#include "crit/core.h"
#include "trace.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace crit::core {
namespace {
struct Request {
    std::uint64_t id = 0;
    RequestKey trace_key;
    std::chrono::steady_clock::time_point submitted;
    Reservation storage, staging_storage;
    std::vector<Token> prompt;
    RequestOptions options;
    std::uint32_t output = 0, first_target = 0;
    bool scoring = false;
    std::vector<Token> staging;
    // The model may still have queued writes into its returned score allocation.
    // Keep that allocation in the quarantinable owner until drain succeeds.
    std::vector<float> score_output;
    std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
    std::mutex mutex;
    std::condition_variable cv;
    Result result;
    std::exception_ptr error;
    bool done = false;
};
void validate_tokens(std::span<const Token> tokens, std::uint32_t vocabulary) {
    for(auto token:tokens) if(token < 0 || std::uint32_t(token) >= vocabulary)
        throw std::invalid_argument("token outside declared vocabulary");
}
bool has_sampling(const SamplingOptions& s) {
    return s.temperature || s.top_p || s.top_k || s.seed;
}
void validate_options(const RequestOptions& options, const ModelInfo& info, bool scoring) {
    const auto& s = options.sampling;
    if(scoring && (has_sampling(s) || options.stop_tokens))
        throw std::invalid_argument("scoring does not accept generation options");
    if((s.temperature && !info.sampling.temperature) || (s.top_p && !info.sampling.top_p) ||
       (s.top_k && !info.sampling.top_k) || (s.seed && !info.sampling.seed))
        throw std::invalid_argument("model does not support a requested sampling option");
    if(s.temperature && (!std::isfinite(*s.temperature) || *s.temperature < 0))
        throw std::invalid_argument("temperature must be finite and nonnegative");
    if(s.top_p && (!std::isfinite(*s.top_p) || *s.top_p <= 0 || *s.top_p > 1))
        throw std::invalid_argument("top_p must be finite and in (0, 1]");
    if(s.top_k && *s.top_k > info.vocabulary)
        throw std::invalid_argument("top_k exceeds vocabulary");
    if(options.stop_tokens) validate_tokens(*options.stop_tokens,info.vocabulary);
    if(options.context && !info.request_context)
        throw std::invalid_argument("model does not support request context");
}
void charge_bytes(std::size_t& total, std::size_t count, std::size_t width = 1) {
    if(count > (std::numeric_limits<std::size_t>::max()-total)/width)
        throw std::overflow_error("request storage accounting overflow");
    total += count*width;
}
std::atomic<std::size_t> quarantined_count{0};
class OwnedTextDecoder final : public TextDecoder {
    // Destroy the decoder before the frontend whose tables it may borrow.
    std::shared_ptr<const TextFrontend> owner_;
    std::unique_ptr<TextDecoder> decoder_;
    std::uint32_t vocabulary_;
    bool closed_ = false;
public:
    OwnedTextDecoder(std::shared_ptr<const TextFrontend> owner,
                     std::unique_ptr<TextDecoder> decoder, std::uint32_t vocabulary)
        : owner_(std::move(owner)), decoder_(std::move(decoder)), vocabulary_(vocabulary) {}
    std::string push(std::span<const Token> tokens) override {
        if(closed_) throw std::logic_error("text decoder is closed");
        validate_tokens(tokens,vocabulary_);
        try { return decoder_->push(tokens); }
        catch(...) { closed_ = true; throw; }
    }
    std::string finish() override {
        if(closed_) throw std::logic_error("text decoder is closed");
        closed_ = true;
        return decoder_->finish();
    }
};
}
std::size_t quarantined_engines() noexcept { return quarantined_count.load(); }
std::unique_ptr<TextDecoder> TextFrontend::create_decoder() const {
    throw std::logic_error("text frontend lacks incremental decoding");
}
std::unique_ptr<Session> Model::open_request(std::span<const Token> prompt, const RequestOptions& options) {
    if(has_sampling(options.sampling) || options.context)
        throw std::logic_error("model must implement open_request for adapter options");
    return open(prompt);
}
std::vector<float> Model::score_request(std::span<const Token> prompt, std::uint32_t first, const RequestOptions& options) {
    if(has_sampling(options.sampling) || options.stop_tokens || options.context)
        throw std::logic_error("model must implement score_request for adapter options");
    return score(prompt,first);
}
std::unique_ptr<Session> ConcurrentModel::open(std::span<const Token>) {
    throw std::logic_error("concurrent model requires identified request admission");
}
std::span<const Token> Session::ready_tokens() {
    throw std::logic_error("session returned deferred_tokens without ready_tokens");
}
std::span<const Token> ConcurrentModel::ready_tokens(const RequestKey&) {
    throw std::logic_error("concurrent model returned deferred_tokens without ready_tokens");
}
struct Engine::Impl : std::enable_shared_from_this<Engine::Impl> {
    Profile profile;
    Resources resources;
    Reservation trace_storage;
    detail::TraceBuffer trace_buffer;
    std::shared_ptr<Backend> backend;
    std::unique_ptr<Model> model;
    std::shared_ptr<const TextFrontend> frontend;
    ModelInfo declaration;
    ConcurrentModel* concurrent = nullptr;
    struct Lane {
        RequestKey key;
        std::shared_ptr<Request> request;
        std::unique_ptr<ConcurrentSession> session;
        std::unique_ptr<Retirement> receipt;
        std::uint32_t frontier = 0;
        bool prefilled = false, retiring = false;
        Finish finish = Finish::length;
        std::chrono::steady_clock::time_point retirement_started;
        std::uint64_t trace_retirement = 0, trace_started = 0;
    };
    std::vector<Lane> lanes;
    std::vector<std::uint64_t> generations;
    std::deque<std::shared_ptr<Request>> waiting;
    std::shared_ptr<std::atomic<std::size_t>> outstanding = std::make_shared<std::atomic<std::size_t>>(0);
    std::uint64_t next_request = 0;
    std::size_t prefill_cursor = 0;
    bool prefer_decode = true;
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<Request> pending;
    std::shared_ptr<Request> active_score;
    std::shared_ptr<Request> retained_request;
    std::unique_ptr<Session> retained_session;
    bool busy = false, stopping = false, poisoned = false;
    Impl* retained_next = nullptr;
    std::thread worker;
    Impl(Profile p, std::shared_ptr<Backend> b):profile(p),resources(p.host_limit,p.device_limit),
        trace_storage(resources.reserve(Memory::host,p.trace.capacity <= 65536 ? p.trace.capacity*sizeof(TraceEvent) : 0)),
        trace_buffer(p.trace),backend(std::move(b)) {
        if(!p.context || !p.prefill_chunk || p.prefill_chunk > p.context || !p.host_limit || !p.device_limit || !backend ||
           !p.max_concurrency || p.max_concurrency > 8 || p.max_pending > 1024 || !p.pending_timeout_ms || !p.retirement_timeout_ms)
            throw std::invalid_argument("invalid core profile/backend/factory");
    }
    void drain_backend(RequestKey key = {}) {
        detail::TraceScope span(trace_buffer,TracePhase::drain,key);
        backend->drain();
    }
    void quiesce_model() {
        if(concurrent) {
            detail::TraceScope span(trace_buffer,TracePhase::quiesce);
            concurrent->quiesce();
        }
    }
    void initialize(Factory factory) {
        if(!factory) throw std::invalid_argument("null model factory");
        backend->initialize(resources);
        backend->bind_thread();
        model = factory(profile,{resources,*backend});
        if(!model) throw std::invalid_argument("model factory returned null");
        concurrent = dynamic_cast<ConcurrentModel*>(model.get());
        drain_backend();
        declaration = model->info();
        if(declaration.implementation.empty() || !declaration.vocabulary || declaration.context < profile.context || declaration.prefill_chunk < profile.prefill_chunk)
            throw std::invalid_argument("model declaration does not cover core profile");
        validate_tokens(declaration.stop_tokens,declaration.vocabulary);
        frontend = model->text_frontend();
        if(!declaration.max_concurrency || declaration.max_concurrency > 8 || declaration.max_concurrency < profile.max_concurrency ||
           (!concurrent && (profile.max_concurrency != 1 || profile.max_pending)))
            throw std::invalid_argument("model concurrency declaration does not cover core profile");
        if(concurrent) { lanes.resize(profile.max_concurrency); generations.resize(profile.max_concurrency); }
        worker = std::thread([this]{run();});
    }
    bool shutdown() noexcept {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            if(pending) *pending->cancelled = true;
        }
        cv.notify_all();
        if(worker.joinable()) worker.join();
        try { backend->bind_thread(); quiesce_model(); drain_backend(); } catch(...) {return false;}
        for(auto& lane:lanes) {
            lane.receipt.reset();
            if(lane.session) lane.session->close(Finish::failed);
            lane.session.reset();
            lane = Lane{};
        }
        waiting.clear();
        active_score.reset();
        if(retained_session) {retained_session->close(Finish::failed); retained_session.reset();}
        retained_request.reset();
        model.reset();
        return true;
    }
    Handle submit(std::vector<Token> prompt, std::uint32_t output, bool scoring, std::uint32_t first, RequestOptions);
    std::span<const Token> stops(const Request& r) const {
        return r.options.stop_tokens ? std::span<const Token>(*r.options.stop_tokens) : std::span<const Token>(declaration.stop_tokens);
    }
    void complete(const std::shared_ptr<Request>& r, Finish finish, std::exception_ptr error = {}) {
        trace_buffer.point(TracePhase::completion,r->trace_key,finish);
        {
            std::lock_guard lock(r->mutex);
            // Staging is worker-private and no longer needed after retirement.
            std::vector<Token>().swap(r->staging);
            r->staging_storage = {};
            r->result.finish = finish; r->error = error; r->done = true;
        }
        r->cv.notify_all();
    }
    void begin_retirement(Lane& lane, Finish finish) {
        lane.finish = finish;
        lane.retirement_started = std::chrono::steady_clock::now();
        if(trace_buffer.enabled()) {
            lane.trace_retirement=trace_buffer.operation(); lane.trace_started=trace_buffer.now();
            trace_buffer.write({0,lane.trace_retirement,lane.trace_started,lane.key,TracePhase::retirement,TraceMark::begin});
        }
        lane.receipt = concurrent->retire(lane.key,*lane.session,finish);
        if(!lane.receipt) throw std::runtime_error("null retirement receipt");
        lane.retiring = true;
    }
    void poll_retirements() {
        for(auto& lane:lanes) if(lane.request) {
            if(!lane.retiring && *lane.request->cancelled) begin_retirement(lane,Finish::cancelled);
            if(!lane.retiring) continue;
            if(lane.receipt->ready()) {
                // Receipt destruction can reference the session; release it first.
                lane.receipt.reset(); lane.session->close(lane.finish); lane.session.reset();
                if(trace_buffer.enabled()) {
                    const auto now=trace_buffer.now();
                    trace_buffer.write({0,lane.trace_retirement,now,lane.key,TracePhase::retirement,TraceMark::end,now-lane.trace_started});
                }
                const auto request=lane.request; const auto finish=lane.finish;
                lane = Lane{};
                {
                    std::lock_guard lock(mutex);
                    busy=!waiting.empty() || std::any_of(lanes.begin(),lanes.end(),[](const auto& value){return bool(value.request);});
                }
                complete(request,finish);
            } else if(std::chrono::steady_clock::now()-lane.retirement_started >= std::chrono::milliseconds(profile.retirement_timeout_ms))
                throw std::runtime_error("retirement deadline expired without settlement");
        }
    }
    void admit() {
        std::lock_guard lock(mutex);
        const auto now = std::chrono::steady_clock::now();
        for(auto it=waiting.begin();it!=waiting.end();) {
            auto r = *it;
            if(*r->cancelled || stopping) {
                complete(r,Finish::cancelled); it = waiting.erase(it);
            } else if(now-r->submitted >= std::chrono::milliseconds(profile.pending_timeout_ms)) {
                complete(r,Finish::failed,std::make_exception_ptr(std::runtime_error("pending request deadline expired")));
                it = waiting.erase(it);
            } else ++it;
        }
        for(std::size_t i=0;i<lanes.size() && !waiting.empty();++i) {
            auto& lane = lanes[i];
            if(lane.request) continue;
            auto r = waiting.front();
            if(r->scoring) break; // An exclusive FIFO barrier, not a generation lane.
            if(!r->output) { waiting.pop_front(); complete(r,Finish::length); --i; continue; }
            if(generations[i] == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("lane generation exhausted");
            RequestKey key{r->id,generations[i]+1,std::uint32_t(i)};
            RequestView view{key,r->prompt,r->output,&r->options};
            const auto admission = [&]{
                detail::TraceScope span(trace_buffer,TracePhase::admission,key);
                return concurrent->admission(view);
            }();
            if(admission == Admission::deferred) break; // FIFO; no hidden backfill policy.
            if(admission == Admission::infeasible) {
                waiting.pop_front();
                complete(r,Finish::failed,std::make_exception_ptr(std::runtime_error("request memory entitlement infeasible")));
                --i; continue;
            }
            if(admission != Admission::ready) throw std::runtime_error("invalid admission decision");
            // Retain request before open: a throwing open may leave live readers.
            lane.key = key; lane.request = r; generations[i] = key.generation;
            r->trace_key=key;
            waiting.pop_front();
            detail::TraceScope span(trace_buffer,TracePhase::open,key);
            lane.session = concurrent->open(view);
            if(!lane.session) throw std::runtime_error("model opened null concurrent session");
        }
    }
    WorkItem work(const Lane& lane) const {
        return {lane.key,lane.session.get(),{
            std::min(profile.prefill_chunk,std::uint32_t(lane.request->prompt.size())-lane.frontier),
            lane.request->output-std::uint32_t(lane.request->result.tokens.size()),lane.prefilled}};
    }
    void publish_group(std::span<const WorkItem> items, std::vector<RowProgress> rows, std::uint64_t operation) {
        if(items.size()!=rows.size()) throw std::runtime_error("decode row membership count mismatch");
        struct Staged {
            Lane* lane; std::span<const Token> tokens;
            std::uint32_t frontier; bool prefilled; Finish finish;
        };
        std::vector<Staged> staged;
        std::vector<CommitDecision> decisions;
        staged.reserve(items.size()); decisions.reserve(items.size());
        for(const auto& item:items) {
            const RowProgress* row = nullptr;
            for(const auto& candidate:rows) if(candidate.key==item.key) {
                if(row) throw std::runtime_error("duplicate row identity");
                row = &candidate;
            }
            if(!row) throw std::runtime_error("stale or missing row identity");
            if(row->progress.readiness) {
                detail::TraceScope span(trace_buffer,TracePhase::readiness,item.key);
                row->progress.readiness->wait();
            }
        }
        for(auto& row:rows) if(row.progress.deferred_tokens) {
            if(!row.progress.readiness || !row.progress.tokens.empty() || !row.progress.prefill_complete)
                throw std::runtime_error("invalid deferred concurrent progress");
            row.progress.tokens = concurrent->ready_tokens(row.key);
        }
        if(trace_buffer.enabled()) {
            try {
                for(const auto& timing:concurrent->device_timings())
                    for(const auto& item:items) trace_buffer.device(operation,item.key,timing);
            } catch(...) { trace_buffer.drop(); }
        }
        // Every borrowed row is materialized before commit invalidates any row.
        for(const auto& item:items) {
            const auto& p = std::find_if(rows.begin(),rows.end(),[&](const auto& row){return row.key==item.key;})->progress;
            auto& lane = lanes[item.key.lane];
            if(p.prompt_tokens > item.budget.prefill_tokens || (lane.prefilled && p.prompt_tokens) ||
               (lane.prefilled && !p.prefill_complete) ||
               p.prefill_complete != (lane.frontier+p.prompt_tokens==lane.request->prompt.size()) ||
               (!p.prefill_complete && !p.tokens.empty()) || p.tokens.size()>item.budget.output_tokens ||
               (!p.prompt_tokens && p.tokens.empty())) throw std::runtime_error("invalid or stalled concurrent progress");
            validate_tokens(p.tokens,declaration.vocabulary);
            auto accepted = std::uint32_t(p.tokens.size());
            auto finish = Finish::length;
            const auto stop_tokens = stops(*lane.request);
            for(std::uint32_t j=0;j<accepted;++j) if(std::find(stop_tokens.begin(),stop_tokens.end(),p.tokens[j])!=stop_tokens.end()) {
                accepted=j+1; finish=Finish::stop; break;
            }
            const bool cancelled = *lane.request->cancelled;
            if(cancelled) { accepted=0; finish=Finish::cancelled; }
            const bool terminal = cancelled || finish==Finish::stop || accepted==item.budget.output_tokens;
            // Admission has reserved the largest possible accepted row. Reuse
            // it so a later submission cannot consume this group's headroom.
            auto& copied=lane.request->staging;
            copied.clear();
            if(accepted) copied.assign(p.tokens.begin(),p.tokens.begin()+accepted);
            staged.push_back({&lane,copied,lane.frontier+p.prompt_tokens,p.prefill_complete,finish});
            decisions.push_back({item.key,item.session,accepted,terminal,cancelled});
        }
        {
            detail::TraceScope span(trace_buffer,TracePhase::commit,items);
            concurrent->commit(decisions);
        }
        for(std::size_t i=0;i<staged.size();++i) {
            auto& value = staged[i]; auto& lane = *value.lane;
            lane.frontier=value.frontier; lane.prefilled=value.prefilled;
            {
                std::lock_guard lock(lane.request->mutex);
                lane.request->result.tokens.insert(lane.request->result.tokens.end(),value.tokens.begin(),value.tokens.end());
            }
            trace_buffer.point(TracePhase::publication,lane.key);
            lane.request->cv.notify_all();
        }
        for(std::size_t i=0;i<staged.size();++i) if(decisions[i].terminal) begin_retirement(*staged[i].lane,staged[i].finish);
    }
    bool step_concurrent() {
        std::vector<WorkItem> decode;
        std::size_t prefill = lanes.size();
        for(std::size_t n=0;n<lanes.size();++n) {
            const auto i=(prefill_cursor+n)%lanes.size(); const auto& lane=lanes[i];
            if(!lane.request || lane.retiring) continue;
            if(lane.prefilled) decode.push_back(work(lane));
            else if(prefill==lanes.size()) prefill=i;
        }
        if(!decode.empty() && (prefer_decode || prefill==lanes.size())) {
            detail::TraceScope step(trace_buffer,TracePhase::step,std::span<const WorkItem>(decode));
            auto rows=[&]{
                detail::TraceScope span(trace_buffer,TracePhase::decode,std::span<const WorkItem>(decode));
                return concurrent->decode(decode);
            }();
            publish_group(decode,std::move(rows),step.operation()); prefer_decode=false; return true;
        }
        if(prefill!=lanes.size()) {
            const auto item=work(lanes[prefill]);
            detail::TraceScope step(trace_buffer,TracePhase::step,item.key);
            auto row=[&]{
                detail::TraceScope span(trace_buffer,TracePhase::prefill,item.key);
                return concurrent->prefill(item);
            }();
            publish_group(std::span<const WorkItem>(&item,1),{std::move(row)},step.operation());
            prefill_cursor=(prefill+1)%lanes.size(); prefer_decode=true; return true;
        }
        return false;
    }
    Finish score(const std::shared_ptr<Request>& request) {
        if(*request->cancelled) return Finish::cancelled;
        request->score_output = [&]{
            detail::TraceScope span(trace_buffer,TracePhase::score,request->trace_key);
            return model->score_request(request->prompt,request->first_target,request->options);
        }();
        // Retain returned storage through the fence, including cancellation.
        drain_backend(request->trace_key);
        const auto& scores = request->score_output;
        if(scores.size() != request->prompt.size()-request->first_target ||
           !std::all_of(scores.begin(),scores.end(),[](float x){return std::isfinite(x);}))
            throw std::runtime_error("scoring output count/finiteness mismatch");
        if(*request->cancelled) {
            std::vector<float>().swap(request->score_output);
            return Finish::cancelled;
        }
        // Keep Core's pre-reserved extent, not an arbitrary backend capacity.
        request->result.scores.assign(scores.begin(),scores.end());
        std::vector<float>().swap(request->score_output);
        trace_buffer.point(TracePhase::publication,request->trace_key);
        return Finish::length;
    }
    bool score_concurrent() {
        {
            std::lock_guard lock(mutex);
            if(waiting.empty() || !waiting.front()->scoring ||
               std::any_of(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);}))
                return false;
            // Retain before invoking the model: failure may leave live readers.
            active_score = waiting.front(); waiting.pop_front();
        }
        trace_buffer.point(TracePhase::admission,active_score->trace_key);
        const auto finish = score(active_score);
        {
            std::lock_guard lock(mutex);
            busy = !waiting.empty();
        }
        complete(active_score,finish);
        active_score.reset();
        return true;
    }
    void fail_concurrent(std::exception_ptr error) noexcept {
        { std::lock_guard lock(mutex); poisoned=true; }
        bool settled=false;
        try { quiesce_model(); drain_backend(); settled=true; } catch(...) {}
        if(active_score) {
            const auto request=active_score;
            if(settled) {
                std::vector<float>().swap(request->score_output);
                active_score.reset();
            }
            complete(request,Finish::failed,error);
        }
        for(auto& lane:lanes) if(lane.request) {
            const auto request=lane.request;
            if(settled) {
                lane.receipt.reset(); if(lane.session) lane.session->close(Finish::failed);
                lane = Lane{};
            }
            // Match normal retirement: a completed handle must not race
            // session close or the release of settled session reservations.
            // Unknown work still retains the complete lane for quarantine.
            complete(request,Finish::failed,error);
        }
        std::lock_guard lock(mutex);
        for(auto& r:waiting) complete(r,Finish::failed,error);
        if(settled) waiting.clear();
        busy=false;
    }
    void run_concurrent() noexcept {
        try {
            backend->bind_thread();
            for(;;) {
                poll_retirements(); admit();
                if(score_concurrent()) continue;
                if(step_concurrent()) continue;
                std::unique_lock lock(mutex);
                const bool occupied=std::any_of(lanes.begin(),lanes.end(),[](const auto& lane){return bool(lane.request);});
                busy=occupied || !waiting.empty();
                if(stopping && !busy) return;
                if(!busy) cv.wait(lock,[&]{return stopping || !waiting.empty();});
                else cv.wait_for(lock,std::chrono::milliseconds(1));
            }
        } catch(...) {fail_concurrent(std::current_exception());}
    }
    void execute(const std::shared_ptr<Request>& request) {
        request->trace_key={request->id,request->scoring ? 0u : 1u,0};
        const auto key=request->trace_key;
        trace_buffer.point(TracePhase::admission,key);
        std::unique_ptr<Session> session;
        Finish finish = Finish::failed;
        std::exception_ptr error;
        try {
            backend->bind_thread();
            if(*request->cancelled) finish = Finish::cancelled;
            else if(request->scoring) finish = score(request);
            else if(!request->output) finish = Finish::length;
            else {
                {
                    detail::TraceScope span(trace_buffer,TracePhase::open,key);
                    session = model->open_request(request->prompt,request->options);
                }
                if(!session) throw std::runtime_error("model opened null session");
                std::uint32_t frontier = 0;
                bool prefill_complete = false;
                finish = Finish::length;
                while(request->result.tokens.size() < request->output) {
                    detail::TraceScope step(trace_buffer,TracePhase::step,key);
                    if(*request->cancelled) {finish = Finish::cancelled; break;}
                    const auto remaining = request->output-std::uint32_t(request->result.tokens.size());
                    const auto chunk = std::min(profile.prefill_chunk,std::uint32_t(request->prompt.size())-frontier);
                    auto progress = [&]{
                        detail::TraceScope span(trace_buffer,prefill_complete ? TracePhase::decode : TracePhase::prefill,key);
                        return session->advance({chunk,remaining,prefill_complete});
                    }();
                    if(progress.deferred_tokens && (!progress.readiness || !progress.tokens.empty() || !progress.prefill_complete))
                        throw std::runtime_error("invalid deferred model progress");
                    if(progress.readiness) {
                        detail::TraceScope span(trace_buffer,TracePhase::readiness,key);
                        progress.readiness->wait();
                    }
                    if(progress.deferred_tokens) progress.tokens = session->ready_tokens();
                    if(trace_buffer.enabled()) {
                        try { for(const auto& timing:session->device_timings()) trace_buffer.device(step.operation(),key,timing); }
                        catch(...) { trace_buffer.drop(); }
                    }
                    if(*request->cancelled) {finish = Finish::cancelled; break;}
                    if(progress.prompt_tokens > chunk || (prefill_complete && progress.prompt_tokens) ||
                       (prefill_complete && !progress.prefill_complete) ||
                       progress.prefill_complete != (frontier+progress.prompt_tokens == request->prompt.size()) ||
                       (!progress.prefill_complete && !progress.tokens.empty()) ||
                       progress.tokens.size() > remaining ||
                       (!progress.prompt_tokens && progress.tokens.empty()))
                        throw std::runtime_error("invalid or stalled model progress");
                    frontier += progress.prompt_tokens;
                    prefill_complete = progress.prefill_complete;
                    validate_tokens(progress.tokens,declaration.vocabulary);
                    auto accepted = std::uint32_t(progress.tokens.size());
                    const auto stop_tokens = stops(*request);
                    for(std::uint32_t i=0;i<accepted;++i) if(std::find(stop_tokens.begin(),stop_tokens.end(),progress.tokens[i]) != stop_tokens.end()) {
                        accepted = i+1; finish = Finish::stop; break;
                    }
                    const bool terminal = finish == Finish::stop || accepted == remaining;
                    {
                        std::lock_guard lock(request->mutex);
                        const auto previous = request->result.tokens.size();
                        // Commit may invalidate the borrowed batch. Stage its accepted
                        // values first, withholding publication under the request lock.
                        if(accepted) request->result.tokens.insert(request->result.tokens.end(),progress.tokens.begin(),progress.tokens.begin()+accepted);
                        try {
                            detail::TraceScope span(trace_buffer,TracePhase::commit,key);
                            session->commit(accepted,terminal);
                        }
                        catch(...) {request->result.tokens.resize(previous); throw;}
                    }
                    trace_buffer.point(TracePhase::publication,key);
                    request->cv.notify_all();
                    if(terminal) break;
                }
            }
        } catch(...) { error = std::current_exception(); finish = Finish::failed; }
        try {
            drain_backend(key);
            std::vector<float>().swap(request->score_output);
        }
        catch(...) {
            if(!error) error = std::current_exception();
            finish = Finish::failed;
            std::lock_guard lock(mutex); poisoned = true;
            retained_session = std::move(session);
            retained_request = request;
        }
        if(session) {
            detail::TraceScope span(trace_buffer,TracePhase::retirement,key);
            session->close(finish); session.reset();
        }
        {
            std::lock_guard lock(mutex);
            busy = false;
        }
        complete(request,finish,error);
    }
    void run() noexcept {
        if(concurrent) { run_concurrent(); return; }
        for(;;) {
            std::shared_ptr<Request> request;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock,[&]{return stopping || pending;});
                if(stopping && !pending) return;
                request = std::move(pending);
            }
            execute(request);
        }
    }
};
struct Handle::Impl {
    std::shared_ptr<Engine::Impl> engine;
    std::shared_ptr<Request> request;
    std::shared_ptr<std::atomic<std::size_t>> outstanding;
    ~Impl() {
        if(request) {
            *request->cancelled = true;
            std::unique_lock lock(request->mutex);
            request->cv.wait(lock,[&]{return request->done;});
        }
        if(outstanding) --*outstanding;
    }
};
Handle::Handle() noexcept = default;
Handle::Handle(std::unique_ptr<Impl> p):impl_(std::move(p)),cancellation_(impl_ ? impl_->request->cancelled : nullptr),id_(impl_ ? impl_->request->id : 0) {}
Handle::~Handle() = default;
Handle::Handle(Handle&&) noexcept = default;
Handle& Handle::operator=(Handle&&) noexcept = default;
void Handle::cancel() noexcept { if(cancellation_) *cancellation_ = true; }
std::uint64_t Handle::id() const noexcept { return cancellation_ ? id_ : 0; }
Result Handle::wait(TokenSink sink) {
    if(!impl_ || !impl_->request) throw std::logic_error("request handle is empty/consumed");
    auto state = std::move(impl_);
    auto request = state->request;
    std::size_t consumed = 0;
    std::unique_lock lock(request->mutex);
    for(;;) {
        request->cv.wait(lock,[&]{return request->done || (sink && consumed < request->result.tokens.size());});
        if(sink && consumed < request->result.tokens.size()) {
            const auto token = request->result.tokens[consumed++];
            lock.unlock();
            try { sink(token); } catch(...) {
                *request->cancelled = true;
                lock.lock();
                request->cv.wait(lock,[&]{return request->done;});
                lock.unlock();
                throw;
            }
            lock.lock();
            continue;
        }
        if(request->done) break;
    }
    const auto error = request->error;
    auto result = std::move(request->result);
    lock.unlock();
    state->request.reset();
    if(error) std::rethrow_exception(error);
    return result;
}
Engine::Engine(Profile p,std::shared_ptr<Backend> b,Factory f) {
    impl_ = std::shared_ptr<Impl>(new Impl(p,std::move(b)),[](Impl* owner) noexcept {
        if(owner->shutdown()) {delete owner; return;}
        // Keep the services facades, model, prompt and storage owners at their
        // original addresses after an unprovable final drain. Never call their
        // destructors on this path. The process must recover its failed backend.
        static Impl* retained = nullptr;
        static std::mutex retention_mutex;
        try {std::lock_guard lock(retention_mutex); owner->retained_next = retained; retained = owner;}
        catch(...) { /* Never free unknown live work even if diagnostics fail. */ }
        ++quarantined_count;
        std::fprintf(stderr,"crit core: failed backend drain; complete runtime retained for process recovery\n");
    });
    impl_->initialize(std::move(f));
}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
Handle Engine::submit(std::vector<Token> prompt,std::uint32_t output,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(!impl_->declaration.generation) throw std::logic_error("model lacks generation capability");
    return impl_->submit(std::move(prompt),output,false,0,std::move(options));
}
Handle Engine::submit_score(std::vector<Token> prompt,std::uint32_t first,RequestOptions options) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(!impl_->declaration.scoring) throw std::logic_error("model lacks scoring capability");
    if(prompt.size() < 2 || !first || first >= prompt.size()) throw std::invalid_argument("invalid scoring target domain");
    return impl_->submit(std::move(prompt),0,true,first,std::move(options));
}
Handle Engine::Impl::submit(std::vector<Token> prompt,std::uint32_t output,bool scoring,std::uint32_t first,RequestOptions options) {
    if(prompt.empty() || prompt.size() > profile.context || output > profile.context-prompt.size())
        throw std::invalid_argument("prompt plus output exceeds context");
    validate_tokens(prompt,declaration.vocabulary);
    validate_options(options,declaration,scoring);
    // Reject overload before charging request memory. Keep the allowance check
    // and publication atomic with other submitters; failed construction leaves
    // neither an outstanding slot nor a retained reservation.
    std::lock_guard lock(mutex);
    if(stopping || poisoned) throw std::runtime_error("engine unavailable");
    if(!concurrent && busy) throw std::runtime_error("single-request core is busy");
    if(concurrent && outstanding->load() >= std::size_t(profile.max_concurrency)+profile.max_pending)
        throw std::runtime_error("outstanding request allowance exhausted");
    if(next_request == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("request identity exhausted");
    auto request = std::make_shared<Request>();
    // Account retained prompt and maximum result bytes, including handle lifetime.
    const auto result_items = scoring ? prompt.size()-first : output;
    std::size_t bytes = 0;
    charge_bytes(bytes,prompt.capacity(),sizeof(Token));
    charge_bytes(bytes,result_items,scoring ? sizeof(float) : sizeof(Token));
    if(options.stop_tokens) charge_bytes(bytes,options.stop_tokens->capacity(),sizeof(Token));
    if(options.context) charge_bytes(bytes,options.context->host_bytes());
    request->storage = resources.reserve(Memory::host,bytes);
    if(concurrent) {
        request->staging_storage = resources.reserve(Memory::host,std::size_t(output)*sizeof(Token));
        request->staging.reserve(output);
    }
    request->prompt = std::move(prompt);
    request->options = std::move(options);
    request->output = output; request->scoring = scoring; request->first_target = first;
    if(scoring) request->result.scores.reserve(result_items);
    else request->result.tokens.reserve(output);
    // Retain the runtime before the worker can observe the request.
    auto handle = std::make_unique<Handle::Impl>();
    handle->engine = shared_from_this();
    request->id = ++next_request; request->result.request_id=request->id;
    request->trace_key={request->id,0,0};
    request->submitted=std::chrono::steady_clock::now();
    if(concurrent) {
        waiting.push_back(request);
        ++*outstanding; handle->outstanding=outstanding;
    } else pending=request;
    // No throwing operation may follow attaching an uncompleted request to a
    // handle: its destructor waits. In particular queue growth precedes this.
    handle->request = request;
    busy = true;
    trace_buffer.point(TracePhase::submitted,request->trace_key);
    cv.notify_one();
    return Handle(std::move(handle));
}
std::vector<Token> Engine::tokenize(std::string_view text) {
    if(!impl_) throw std::logic_error("engine is moved from");
    if(impl_->frontend) {
        auto tokens = impl_->frontend->tokenize(text);
        validate_tokens(tokens,impl_->declaration.vocabulary);
        return tokens;
    }
    std::lock_guard lock(impl_->mutex);
    if(impl_->busy || impl_->poisoned) throw std::runtime_error("engine unavailable for frontend access");
    auto tokens = impl_->model->tokenize(text);
    validate_tokens(tokens,impl_->declaration.vocabulary);
    return tokens;
}
std::string Engine::detokenize(std::span<const Token> tokens) {
    if(!impl_) throw std::logic_error("engine is moved from");
    validate_tokens(tokens,impl_->declaration.vocabulary);
    if(impl_->frontend) return impl_->frontend->detokenize(tokens);
    std::lock_guard lock(impl_->mutex);
    if(impl_->busy || impl_->poisoned) throw std::runtime_error("engine unavailable for frontend access");
    return impl_->model->detokenize(tokens);
}
std::unique_ptr<TextDecoder> Engine::create_decoder() {
    if(!impl_) throw std::logic_error("engine is moved from");
    auto frontend = impl_->frontend;
    if(!frontend) throw std::logic_error("model lacks an independent text frontend");
    auto decoder = frontend->create_decoder();
    if(!decoder) throw std::runtime_error("text frontend returned null decoder");
    return std::make_unique<OwnedTextDecoder>(std::move(frontend),std::move(decoder),impl_->declaration.vocabulary);
}
ModelInfo Engine::info() const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->declaration; }
Profile Engine::profile() const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->profile; }
Usage Engine::usage(Memory memory) const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->resources.usage(memory); }
TraceSnapshot Engine::trace() const { if(!impl_) throw std::logic_error("engine is moved from"); return impl_->trace_buffer.snapshot(); }
}
