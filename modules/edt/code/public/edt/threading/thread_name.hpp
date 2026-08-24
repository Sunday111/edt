#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

namespace edt
{
inline void SetCurrentThreadName(std::string_view name) noexcept
{
#if defined(__linux__)
    constexpr size_t kMaximumNameLength = 15;
    std::array<char, kMaximumNameLength + 1> platform_name{};
    std::copy_n(name.data(), std::min(name.size(), kMaximumNameLength), platform_name.data());
    (void)pthread_setname_np(pthread_self(), platform_name.data());
#elif defined(__APPLE__)
    constexpr size_t kMaximumNameLength = 63;
    std::array<char, kMaximumNameLength + 1> platform_name{};
    std::copy_n(name.data(), std::min(name.size(), kMaximumNameLength), platform_name.data());
    (void)pthread_setname_np(platform_name.data());
#else
    (void)name;
#endif
}
}  // namespace edt
