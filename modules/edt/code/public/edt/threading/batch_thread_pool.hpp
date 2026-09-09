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

#include "edt/functional/on_scope_leave.hpp"
#include "edt/threading/thread_name.hpp"

namespace edt
{
// Runs one callback on every worker thread and returns once all of them are done.
// The pool owns its threads for its whole lifetime; a batch costs two barrier
// rendezvous rather than a task allocation and a wakeup per call.
template <typename Thread>
class BasicBatchThreadPool
{
public:
    using Callback = void (*)(void* context, size_t thread_index, size_t num_threads);

    explicit BasicBatchThreadPool(size_t threads_count, std::string thread_name_prefix = "edt_batch_")
        : thread_name_prefix_(std::move(thread_name_prefix)),
          sync_point_(static_cast<int32_t>(threads_count + 1))
    {
        threads_.reserve(threads_count);
        auto cleanup_incomplete_startup = OnScopeLeave(
            [&]
            {
                if (threads_.size() == threads_count) return;
                std::ranges::for_each(threads_, &Thread::request_stop);
                for (size_t missing = threads_.size(); missing < threads_count; ++missing)
                {
                    sync_point_.arrive_and_drop();
                }
                sync_point_.arrive_and_wait();
            });
        for (size_t thread_index : std::views::iota(size_t{0}, threads_count))
        {
            threads_.emplace_back([this, thread_index](std::stop_token stop_token)
                                  { ThreadEntry(stop_token, thread_index); });
        }
    }

    BasicBatchThreadPool(const BasicBatchThreadPool&) = delete;
    BasicBatchThreadPool(BasicBatchThreadPool&&) = delete;
    BasicBatchThreadPool& operator=(const BasicBatchThreadPool&) = delete;
    BasicBatchThreadPool& operator=(BasicBatchThreadPool&&) = delete;

    ~BasicBatchThreadPool()
    {
        std::ranges::for_each(threads_, &Thread::request_stop);
        sync_point_.arrive_and_wait();
        std::ranges::for_each(threads_, &Thread::join);
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
    std::vector<Thread> threads_;
    Callback callback_ = nullptr;
    void* context_ = nullptr;
};

using BatchThreadPool = BasicBatchThreadPool<std::jthread>;
}  // namespace edt
