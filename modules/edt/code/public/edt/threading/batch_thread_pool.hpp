#pragma once

#include <algorithm>
#include <barrier>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "edt/threading/thread_name.hpp"

namespace edt
{
// Runs one callback on every worker thread and returns once all of them are done.
// The pool owns its threads for its whole lifetime; a batch costs two barrier
// rendezvous rather than a task allocation and a wakeup per call.
class BatchThreadPool
{
public:
    using Callback = void (*)(void* context, size_t thread_index, size_t num_threads);

    explicit BatchThreadPool(size_t threads_count, std::string thread_name_prefix = "edt_batch_")
        : thread_name_prefix_(std::move(thread_name_prefix)),
          sync_point_(static_cast<int32_t>(threads_count + 1))
    {
        for (const size_t thread_index : std::views::iota(size_t{0}, threads_count))
        {
            threads_.emplace_back([this, thread_index](std::stop_token stop_token)
                                  { ThreadEntry(stop_token, thread_index); });
        }
    }

    BatchThreadPool(const BatchThreadPool&) = delete;
    BatchThreadPool(BatchThreadPool&&) = delete;
    BatchThreadPool& operator=(const BatchThreadPool&) = delete;
    BatchThreadPool& operator=(BatchThreadPool&&) = delete;

    ~BatchThreadPool()
    {
        std::ranges::for_each(threads_, &std::jthread::request_stop);
        sync_point_.arrive_and_wait();
        std::ranges::for_each(threads_, &std::jthread::join);
    }

    [[nodiscard]] size_t GetThreadsCount() const { return threads_.size(); }

    template <std::invocable<size_t, size_t> T>
    void RunBatch(T&& callback)
    {
        using Invocable = std::remove_reference_t<T>;
        RunBatch(
            [](void* context, size_t thread_index, size_t num_threads)
            {
                auto& invocable = *reinterpret_cast<Invocable*>(context);  // NOLINT
                invocable(thread_index, num_threads);
            },
            &callback);
    }

    void RunBatch(Callback callback, void* context)
    {
        callback_ = callback;
        context_ = context;
        sync_point_.arrive_and_wait();
        sync_point_.arrive_and_wait();
    }

private:
    // Every iteration has to reach the first barrier: the destructor requests the stop and
    // then arrives there itself, so a worker that decided to exit before arriving would
    // leave that arrival missing and hang the destructor. Stop is therefore only ever
    // observed on the far side of the barrier.
    void ThreadEntry(const std::stop_token& stop_token, size_t thread_index)
    {
        SetCurrentThreadName(thread_name_prefix_ + std::to_string(thread_index));
        for (;;)
        {
            sync_point_.arrive_and_wait();
            if (stop_token.stop_requested()) break;
            callback_(context_, thread_index, threads_.size());
            sync_point_.arrive_and_wait();
        }
    }

    std::string thread_name_prefix_;
    std::barrier<> sync_point_;
    std::vector<std::jthread> threads_;
    Callback callback_ = nullptr;
    void* context_ = nullptr;
};
}  // namespace edt
