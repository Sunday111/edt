#include "edt/threading/thread_name.hpp"

#include <array>
#include <string>
#include <thread>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

#include "gtest/gtest.h"

#if defined(__linux__) || defined(__APPLE__)
TEST(ThreadNameTest, TruncatesNamesToThePlatformLimit)
{
#if defined(__linux__)
    constexpr size_t kMaximumNameLength = 15;
#else
    constexpr size_t kMaximumNameLength = 63;
#endif

    std::array<char, kMaximumNameLength + 1> actual_name{};
    int result = 0;
    const std::string requested_name(kMaximumNameLength + 10, 'x');
    std::jthread thread(
        [&]
        {
            edt::SetCurrentThreadName(requested_name);
            result = pthread_getname_np(pthread_self(), actual_name.data(), actual_name.size());
        });
    thread.join();

    EXPECT_EQ(result, 0);
    EXPECT_EQ(actual_name.data(), std::string(kMaximumNameLength, 'x'));
}
#endif
