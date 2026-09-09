#include "edt/pointer/intrusive_ptr.hpp"

#include "gtest/gtest.h"

namespace
{
struct Counted
{
    virtual ~Counted() = default;
    int refs = 0;
};

struct Derived : Counted
{
    int payload = 7;
};

struct Other : Counted
{
};

struct Traits
{
    static void AddReference(Counted* p) { ++p->refs; }
    static void ReleaseReference(Counted* p) { --p->refs; }
};

template <typename T>
using Ptr = edt::IntrusivePtr<T, Traits>;

struct OwningTraits
{
    static void AddReference(Counted* p) { ++p->refs; }
    static void ReleaseReference(Counted* p)
    {
        if (--p->refs == 0) delete p;
    }
};

struct OwnedNode : Counted
{
    explicit OwnedNode(int& live) : live_(live) { ++live_; }
    ~OwnedNode() override { --live_; }

    int& live_;
    edt::IntrusivePtr<OwnedNode, OwningTraits> next;
};
}  // namespace

TEST(IntrusivePtrTest, AssignmentFromOwnedMemberKeepsTheChildAlive)
{
    using OwnedPtr = edt::IntrusivePtr<OwnedNode, OwningTraits>;
    for (bool move : {false, true})
    {
        SCOPED_TRACE(move);
        int live = 0;
        {
            auto parent = OwnedPtr::MakeInstance(live);
            parent->next = OwnedPtr::MakeInstance(live);
            OwnedNode* child = parent->next.Get();
            ASSERT_EQ(live, 2);

            if (move)
                parent = std::move(parent->next);
            else
                parent = parent->next;

            ASSERT_EQ(live, 1);
            ASSERT_EQ(parent.Get(), child);
            EXPECT_EQ(parent->refs, 1);
        }
        EXPECT_EQ(live, 0);
    }
}

TEST(IntrusivePtrTest, ConvertingAssignmentFromOwnedMemberKeepsTheChildAlive)
{
    using OwnedPtr = edt::IntrusivePtr<OwnedNode, OwningTraits>;
    for (bool move : {false, true})
    {
        SCOPED_TRACE(move);
        int live = 0;
        {
            edt::IntrusivePtr<Counted, OwningTraits> parent = OwnedPtr::MakeInstance(live);
            auto& node = *static_cast<OwnedNode*>(parent.Get());
            node.next = OwnedPtr::MakeInstance(live);
            OwnedNode* child = node.next.Get();

            if (move)
                parent = std::move(node.next);
            else
                parent = node.next;

            ASSERT_EQ(live, 1);
            ASSERT_EQ(parent.Get(), child);
            EXPECT_EQ(parent->refs, 1);
        }
        EXPECT_EQ(live, 0);
    }
}

TEST(IntrusivePtrTest, CountsReferences)
{
    Derived object;
    {
        Ptr<Derived> a{&object};
        EXPECT_EQ(object.refs, 1);
        {
            Ptr<Derived> b = a;  // NOLINT(performance-unnecessary-copy-initialization)
            EXPECT_EQ(object.refs, 2);
        }
        EXPECT_EQ(object.refs, 1);
    }
    EXPECT_EQ(object.refs, 0);
}

TEST(IntrusivePtrTest, MoveDoesNotChangeTheCount)
{
    Derived object;
    Ptr<Derived> a{&object};
    Ptr<Derived> b = std::move(a);
    EXPECT_EQ(object.refs, 1);
    EXPECT_EQ(b.Get(), &object);

    // The moved-from state is part of the contract: the source gives up its
    // reference rather than adding one.
    // NOLINTBEGIN(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
    EXPECT_EQ(a.Get(), nullptr);
    EXPECT_FALSE(static_cast<bool>(a));
    // NOLINTEND(bugprone-use-after-move, clang-analyzer-cplusplus.Move)
}

TEST(IntrusivePtrTest, SelfAssignmentKeepsThePointerAlive)
{
    Derived object;
    Ptr<Derived> a{&object};
    const auto& alias = a;
    a = alias;
    EXPECT_EQ(object.refs, 1);
    EXPECT_EQ(a.Get(), &object);
}

TEST(IntrusivePtrTest, ComparisonsComeFromSpaceship)
{
    Derived first;
    Derived second;
    Ptr<Derived> a{&first};
    Ptr<Derived> b{&second};
    Ptr<Derived> a2{&first};

    EXPECT_TRUE(a == a2);
    EXPECT_TRUE(a != b);
    EXPECT_EQ(a < b, &first < &second);
    EXPECT_EQ(a > b, &first > &second);
    EXPECT_TRUE(a <= a2);
    EXPECT_TRUE(a >= a2);

    const Ptr<Derived> empty;
    EXPECT_TRUE(empty == nullptr);
    EXPECT_TRUE(a != nullptr);
}

TEST(IntrusivePtrTest, BoolConversionIsExplicit)
{
    static_assert(!std::is_convertible_v<Ptr<Derived>, bool>);
    static_assert(std::is_constructible_v<bool, Ptr<Derived>>);

    Derived object;
    Ptr<Derived> a{&object};
    EXPECT_TRUE(static_cast<bool>(a));
}

TEST(IntrusivePtrTest, UpcastsAndCastsBack)
{
    Derived object;
    Ptr<Derived> derived{&object};

    Ptr<Counted> base = derived;
    EXPECT_EQ(object.refs, 2);

    // Unqualified: these live in edt and are found by ADL.
    const Ptr<Derived> back = dynamic_pointer_cast<Derived>(base);
    ASSERT_NE(back.Get(), nullptr);
    EXPECT_EQ(back->payload, 7);

    EXPECT_EQ(dynamic_pointer_cast<Other>(base).Get(), nullptr);
    EXPECT_EQ(static_pointer_cast<Derived>(base).Get(), &object);
}

TEST(IntrusivePtrTest, SwapAndReset)
{
    Derived first;
    Derived second;
    Ptr<Derived> a{&first};
    Ptr<Derived> b{&second};

    swap(a, b);
    EXPECT_EQ(a.Get(), &second);
    EXPECT_EQ(b.Get(), &first);

    a.reset();
    EXPECT_EQ(a.Get(), nullptr);
    EXPECT_EQ(second.refs, 0);
}

TEST(IntrusivePtrTest, ConstructionFromRawPointerIsExplicit)
{
    static_assert(!std::is_convertible_v<Derived*, Ptr<Derived>>);
    static_assert(std::is_constructible_v<Ptr<Derived>, Derived*>);
}
