#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../int_aliases.hpp"
#include "../size.hpp"
#include "../template/values_list.hpp"
#include "enum_bitset.hpp"

namespace edt
{
template <typename T>
constexpr T MakeFlagByIndex(std::size_t index)
{
    return (T(1) << index);
}

template <typename T, std::size_t index>
constexpr T MakeFlagByIndex()
{
    static_assert((index < SizeInBits<T>));
    return MakeFlagByIndex<T>(index);
}

template <typename T>
constexpr bool FlagIsSet(T flags, T flag)
{
    if constexpr (std::is_enum_v<T> && edt::enable_enum_bitset_v<T>)
    {
        using U = std::underlying_type_t<T>;
        return (static_cast<U>(flags) & static_cast<U>(flag)) != U{0};
    }
    else
    {
        return (flags & flag) != T{0};
    }
}

template <typename T>
constexpr std::size_t GetSetFlagsCount(T flags)
{
    if constexpr (std::is_enum_v<T>)
    {
        using U = std::make_unsigned_t<std::underlying_type_t<T>>;
        return static_cast<std::size_t>(std::popcount(static_cast<U>(flags)));
    }
    else
    {
        return static_cast<std::size_t>(std::popcount(static_cast<std::make_unsigned_t<T>>(flags)));
    }
}

template <auto... values>
constexpr auto FlagsToBitset(edt::ValuesList<values...>)
{
    return (values | ...);
}

template <auto bitset>
struct FlagIsSetFilter
{
    template <auto flag>
    struct Functor
    {
        static constexpr bool Value = FlagIsSet(bitset, flag);
    };
};

template <typename T>
struct MakeFlagByIndexMapFunctor
{
    template <auto index>
    struct Functor
    {
        static constexpr auto Value = MakeFlagByIndex<T, index>();
    };
};

template <std::size_t bits_count>
struct BitCountToType;

template <>
struct BitCountToType<8>
{
    using Type = u8;
};

template <>
struct BitCountToType<16>
{
    using Type = u16;
};

template <>
struct BitCountToType<32>
{
    using Type = u32;
};

template <>
struct BitCountToType<64>
{
    using Type = u64;
};

template <std::size_t bits_count>
using BitCountToTypeT = typename BitCountToType<bits_count>::Type;
}  // namespace edt
