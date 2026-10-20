// Parallel small copies from a mapped file: each source row may fault a page in from disk, so the rows of one
// decode pass are copied by a pool of threads to overlap those reads.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <immintrin.h>

namespace fnl {

struct RowCopy {
    uint8_t* dst;
    const uint8_t* src;
    int bytes;
};

class CopyPool {
public:
    explicit CopyPool(int threads) {
        for(int i = 1; i < threads; ++i) workers_.emplace_back([this] { worker(); });
    }
    ~CopyPool() {
        {
            std::lock_guard<std::mutex> g(mutex_);
            stop_ = true;
            ++generation_;
        }
        wake_.notify_all();
        for(auto& t : workers_) t.join();
    }
    // Copies every row; the caller takes part and returns when all are done.
    void run(const RowCopy* copies, int n) {
        if(n <= 0) return;
        copies_ = copies;
        count_ = n;
        next_.store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> g(mutex_);
            __atomic_add_fetch(&generation_, 1, __ATOMIC_RELEASE);
        }
        wake_.notify_all();
        work();
        // Every worker finishes this generation before the job storage is reused.
        while(finished_.load(std::memory_order_acquire) != int(workers_.size())) _mm_pause();
    }

private:
    void work() {
        for(;;) {
            const int i = next_.fetch_add(1, std::memory_order_acq_rel);
            if(i >= count_) return;
            std::memcpy(copies_[i].dst, copies_[i].src, std::size_t(copies_[i].bytes));
        }
    }
    void worker() {
        std::uint64_t seen = 0;
        for(;;) {
            // Spin briefly (passes arrive every few milliseconds), then sleep.
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
            bool got = false;
            while(std::chrono::steady_clock::now() < until) {
                if(__atomic_load_n(&generation_, __ATOMIC_ACQUIRE) != seen) {
                    got = true;
                    break;
                }
                for(int i = 0; i < 64; ++i) _mm_pause();
            }
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if(!got) wake_.wait(lock, [&] { return generation_ != seen; });
                if(stop_) return;
                seen = generation_;
            }
            work();
            finished_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::uint64_t generation_ = 0;
    bool stop_ = false;
    const RowCopy* copies_ = nullptr;
    int count_ = 0;
    std::atomic<int> next_{0}, finished_{0};
};

}  // namespace fnl
