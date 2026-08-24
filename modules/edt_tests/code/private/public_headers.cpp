// Includes and instantiates every public edt header.
//
// Most of edt is templates, so a header nothing includes compiles no further
// than its own syntax. Including every one of them here means every one gets
// compiled; the list is checked against the header tree so that a new header
// cannot be added without landing here too.

#include <array>
#include <filesystem>
#include <set>
#include <source_location>
#include <string>
#include <string_view>

#include "edt/algorithm/clamp.hpp"
#include "edt/algorithm/compare.hpp"
#include "edt/algorithm/sort/bubble_sort.hpp"
#include "edt/algorithm/sort/insertion_sort.hpp"
#include "edt/bitset/bit_iterator.hpp"
#include "edt/bitset/bitset_adapter.hpp"
#include "edt/bitset/bitset_utilities.hpp"
#include "edt/bitset/dynamic_bitset.hpp"
#include "edt/bitset/enum_bitset.hpp"
#include "edt/bitset/fixed_bitset.hpp"
#include "edt/branchless/int_if.hpp"
#include "edt/branchless/swap_if.hpp"
#include "edt/concepts/callable.hpp"
#include "edt/constexpr/hex_to_dec.hpp"
#include "edt/delegate.hpp"
#include "edt/exception/call_and_rethrow.hpp"
#include "edt/exception/checked_cast.hpp"
#include "edt/exception/print_exception.hpp"
#include "edt/exception/throw_if_failed.hpp"
#include "edt/float_aliases.hpp"
#include "edt/frame_rate_counter.hpp"
#include "edt/functional/call_for_each_arg.hpp"
#include "edt/functional/on_scope_leave.hpp"
#include "edt/functional/wrap.hpp"
#include "edt/guid.hpp"
#include "edt/int_aliases.hpp"
#include "edt/math/float_range.hpp"
#include "edt/math/int_range.hpp"
#include "edt/math/math.hpp"
#include "edt/math/matrix.hpp"
#include "edt/math/rotator.hpp"
#include "edt/math/surface_points.hpp"
#include "edt/math/transform.hpp"
#include "edt/observable.hpp"
#include "edt/pointer/intrusive_ptr.hpp"
#include "edt/preprocessor/empty_bases.hpp"
#include "edt/preprocessor/stringify.hpp"
#include "edt/ranges/array_indices2d.hpp"
#include "edt/size.hpp"
#include "edt/string/char_util.hpp"
#include "edt/template/apply_if.hpp"
#include "edt/template/fwd_decl.hpp"
#include "edt/template/is_specialization.hpp"
#include "edt/template/member_offset.hpp"
#include "edt/template/overload.hpp"
#include "edt/template/pure_same_types.hpp"
#include "edt/template/signature.hpp"
#include "edt/template/static_for.hpp"
#include "edt/template/string_literal.hpp"
#include "edt/template/tagged_identifier.hpp"
#include "edt/template/tagged_type_id.hpp"
#include "edt/template/types_list.hpp"
#include "edt/template/values_list.hpp"
#include "edt/template/worst_qualifiers.hpp"
#include "edt/template/worst_reference.hpp"
#include "edt/threading/batch_thread_pool.hpp"
#include "edt/threading/thread_name.hpp"
#include "edt/time/measure_time.hpp"
#include "gtest/gtest.h"

namespace
{
constexpr auto kIncludedHeaders = std::to_array<std::string_view>({
    "algorithm/clamp.hpp",
    "algorithm/compare.hpp",
    "algorithm/sort/bubble_sort.hpp",
    "algorithm/sort/insertion_sort.hpp",
    "bitset/bit_iterator.hpp",
    "bitset/bitset_adapter.hpp",
    "bitset/bitset_utilities.hpp",
    "bitset/dynamic_bitset.hpp",
    "bitset/enum_bitset.hpp",
    "bitset/fixed_bitset.hpp",
    "branchless/int_if.hpp",
    "branchless/swap_if.hpp",
    "concepts/callable.hpp",
    "constexpr/hex_to_dec.hpp",
    "delegate.hpp",
    "exception/call_and_rethrow.hpp",
    "exception/checked_cast.hpp",
    "exception/print_exception.hpp",
    "exception/throw_if_failed.hpp",
    "float_aliases.hpp",
    "frame_rate_counter.hpp",
    "functional/call_for_each_arg.hpp",
    "functional/on_scope_leave.hpp",
    "functional/wrap.hpp",
    "guid.hpp",
    "int_aliases.hpp",
    "math/float_range.hpp",
    "math/int_range.hpp",
    "math/math.hpp",
    "math/matrix.hpp",
    "math/rotator.hpp",
    "math/surface_points.hpp",
    "math/transform.hpp",
    "observable.hpp",
    "pointer/intrusive_ptr.hpp",
    "preprocessor/empty_bases.hpp",
    "preprocessor/stringify.hpp",
    "ranges/array_indices2d.hpp",
    "size.hpp",
    "string/char_util.hpp",
    "template/apply_if.hpp",
    "template/fwd_decl.hpp",
    "template/is_specialization.hpp",
    "template/member_offset.hpp",
    "template/overload.hpp",
    "template/pure_same_types.hpp",
    "template/signature.hpp",
    "template/static_for.hpp",
    "template/string_literal.hpp",
    "template/tagged_identifier.hpp",
    "template/tagged_type_id.hpp",
    "template/types_list.hpp",
    "template/values_list.hpp",
    "template/worst_qualifiers.hpp",
    "template/worst_reference.hpp",
    "threading/batch_thread_pool.hpp",
    "threading/thread_name.hpp",
    "time/measure_time.hpp",
});

// This file lives at modules/edt_tests/code/private/public_headers.cpp, so
// walking four parents lands on modules/. Located through source_location
// because yae exposes no way to pass a compile definition.
std::filesystem::path FindPublicHeadersDir()
{
    const std::filesystem::path self = std::source_location::current().file_name();
    const auto modules_dir = self.parent_path().parent_path().parent_path().parent_path();
    return modules_dir / "edt" / "code" / "public" / "edt";
}
}  // namespace

TEST(PublicHeadersTest, EveryHeaderIsIncludedHere)
{
    const auto root = FindPublicHeadersDir();
    if (!std::filesystem::is_directory(root))
    {
        GTEST_SKIP() << "sources are not available next to the binary: " << root;
    }

    std::set<std::string> on_disk;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".hpp")
        {
            on_disk.insert(entry.path().lexically_relative(root).generic_string());
        }
    }

    const std::set<std::string> included(std::begin(kIncludedHeaders), std::end(kIncludedHeaders));
    EXPECT_EQ(on_disk, included) << "a public header was added or removed without updating public_headers.cpp";
}
