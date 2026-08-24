#include "edt/threading/batch_thread_pool.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <numeric>
#include <set>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#endif

#include "gtest/gtest.h"

TEST(BatchThreadPoolTest, ReportsItsThreadCount)
{
    edt::BatchThreadPool pool(3);
    EXPECT_EQ(pool.GetThreadsCount(), 3u);
}

TEST(BatchThreadPoolTest, EveryThreadRunsTheBatchExactlyOnce)
{
    constexpr size_t kThreads = 4;
    edt::BatchThreadPool pool(kThreads);

    std::vector<size_t> calls_per_thread(kThreads, 0);
    std::vector<size_t> reported_num_threads(kThreads, 0);

    pool.RunBatch(
        [&](size_t thread_index, size_t num_threads)
        {
            ++calls_per_thread[thread_index];
            reported_num_threads[thread_index] = num_threads;
        });

    EXPECT_EQ(calls_per_thread, std::vector<size_t>(kThreads, 1));
    EXPECT_EQ(reported_num_threads, std::vector<size_t>(kThreads, kThreads));
}

TEST(BatchThreadPoolTest, ThreadIndicesAreUnique)
{
    constexpr size_t kThreads = 8;
    edt::BatchThreadPool pool(kThreads);

    std::vector<size_t> seen(kThreads, 0);
    pool.RunBatch([&](size_t thread_index, size_t) { seen[thread_index] = thread_index; });

    const std::set<size_t> unique(seen.begin(), seen.end());
    EXPECT_EQ(unique.size(), kThreads);
}

TEST(BatchThreadPoolTest, RunBatchReturnsOnlyAfterEveryThreadFinished)
{
    constexpr size_t kThreads = 4;
    constexpr size_t kBatches = 32;
    edt::BatchThreadPool pool(kThreads);

    std::atomic<size_t> finished{0};
    for (size_t batch = 0; batch != kBatches; ++batch)
    {
        pool.RunBatch([&](size_t, size_t) { finished.fetch_add(1, std::memory_order_relaxed); });
        ASSERT_EQ(finished.load(std::memory_order_relaxed), (batch + 1) * kThreads);
    }
}

TEST(BatchThreadPoolTest, AcceptsAnLvalueCallable)
{
    edt::BatchThreadPool pool(2);

    std::atomic<size_t> calls{0};
    auto increment = [&](size_t, size_t)
    {
        calls.fetch_add(1, std::memory_order_relaxed);
    };
    pool.RunBatch(increment);

    EXPECT_EQ(calls.load(std::memory_order_relaxed), 2u);
}

TEST(BatchThreadPoolTest, SplitsWorkAcrossThreads)
{
    constexpr size_t kThreads = 4;
    constexpr size_t kElements = 1000;
    edt::BatchThreadPool pool(kThreads);

    std::vector<size_t> values(kElements, 1);
    std::vector<size_t> partial_sums(kThreads, 0);

    pool.RunBatch(
        [&](size_t thread_index, size_t num_threads)
        {
            for (size_t i = thread_index; i < values.size(); i += num_threads)
            {
                partial_sums[thread_index] += values[i];
            }
        });

    EXPECT_EQ(std::accumulate(partial_sums.begin(), partial_sums.end(), size_t{0}), kElements);
}

TEST(BatchThreadPoolTest, DestroysCleanlyWithoutEverRunningABatch)
{
    edt::BatchThreadPool pool(4);
    EXPECT_EQ(pool.GetThreadsCount(), 4u);
}

#if defined(__linux__)
TEST(BatchThreadPoolTest, NamesWorkerThreads)
{
    const auto read_names = [](edt::BatchThreadPool& pool)
    {
        std::array<std::array<char, 16>, 2> names{};
        std::array<int, 2> results{};
        pool.RunBatch(
            [&](size_t thread_index, size_t)
            {
                results[thread_index] =
                    pthread_getname_np(pthread_self(), names[thread_index].data(), names[thread_index].size());
            });

        constexpr std::array<int, 2> expected_results{};
        EXPECT_EQ(results, expected_results);
        return names;
    };

    edt::BatchThreadPool default_name_pool(2);
    const auto default_names = read_names(default_name_pool);
    EXPECT_STREQ(default_names[0].data(), "edt_batch_0");
    EXPECT_STREQ(default_names[1].data(), "edt_batch_1");

    edt::BatchThreadPool custom_name_pool(2, "worker_");
    const auto custom_names = read_names(custom_name_pool);
    EXPECT_STREQ(custom_names[0].data(), "worker_0");
    EXPECT_STREQ(custom_names[1].data(), "worker_1");
}
#endif
