#pragma once

#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <utility>

namespace edt
{
template <class T, typename Traits>
class IntrusivePtr
{
public:
    IntrusivePtr() noexcept = default;

    IntrusivePtr(std::nullptr_t) noexcept {}  // NOLINT(google-explicit-constructor): mirrors raw pointer syntax

    explicit IntrusivePtr(T* p) noexcept : p_(p) { AddReference(); }

    IntrusivePtr(const IntrusivePtr& ip) noexcept { CopyFrom<false, false>(ip); }

    IntrusivePtr(IntrusivePtr&& ip) noexcept { MoveFrom<false>(ip); }

    template <typename U>
        requires std::convertible_to<U*, T*>
    IntrusivePtr(const IntrusivePtr<U, Traits>& ip) noexcept  // NOLINT(google-explicit-constructor): upcast
        : p_(ip.p_)
    {
        AddReference();
    }

    template <typename U>
        requires std::convertible_to<U*, T*>
    // NOLINTNEXTLINE(google-explicit-constructor, cppcoreguidelines-rvalue-reference-param-not-moved): upcast; the
    // pointer is stolen below rather than through std::move
    IntrusivePtr(IntrusivePtr<U, Traits>&& ip) noexcept : p_(ip.p_)
    {
        ip.p_ = nullptr;
    }

    ~IntrusivePtr() { ReleaseReference<false>(); }

    void reset() noexcept { ReleaseReference<true>(); }

    void swap(IntrusivePtr& other) noexcept { std::swap(p_, other.p_); }

    friend void swap(IntrusivePtr& a, IntrusivePtr& b) noexcept { a.swap(b); }

    [[nodiscard]] T* Get() const noexcept { return p_; }

    template <typename U>
    [[nodiscard]] IntrusivePtr<U, Traits> StaticCast() const noexcept
    {
        return IntrusivePtr<U, Traits>(*this, typename IntrusivePtr<U, Traits>::StaticCastTag{});
    }

    template <typename U>
    [[nodiscard]] IntrusivePtr<U, Traits> DynamicCast() const noexcept
    {
        return IntrusivePtr<U, Traits>(*this, typename IntrusivePtr<U, Traits>::DynamicCastTag{});
    }

    // NOLINTNEXTLINE(cert-oop54-cpp): CopyFrom<_, true> compares the pointers and returns early on self-assignment
    IntrusivePtr& operator=(const IntrusivePtr& ptr) noexcept
    {
        CopyFrom<true, true>(ptr);
        return *this;
    }

    template <typename U>
        requires std::convertible_to<U*, T*>
    IntrusivePtr& operator=(const IntrusivePtr<U, Traits>& ptr) noexcept
    {
        CopyFrom<true, true>(ptr);
        return *this;
    }

    IntrusivePtr& operator=(IntrusivePtr&& ptr) noexcept
    {
        if (this != &ptr) MoveFrom<true>(ptr);
        return *this;
    }

    template <typename U>
        requires std::convertible_to<U*, T*>
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved): MoveFrom steals the pointer directly
    IntrusivePtr& operator=(IntrusivePtr<U, Traits>&& ptr) noexcept
    {
        MoveFrom<true>(ptr);
        return *this;
    }

    [[nodiscard]] bool operator==(std::nullptr_t) const noexcept { return p_ == nullptr; }

    template <typename U>
    [[nodiscard]] bool operator==(const IntrusivePtr<U, Traits>& p) const noexcept
    {
        return p_ == p.Get();
    }

    template <typename U>
    [[nodiscard]] std::strong_ordering operator<=>(const IntrusivePtr<U, Traits>& p) const noexcept
    {
        return std::compare_three_way{}(p_, p.Get());
    }

    T* operator->() const noexcept
    {
        assert(p_ != nullptr);  // NOLINT(cert-dcl03-c, misc-static-assert): runtime guard
        return p_;
    }

    T& operator*() const noexcept
    {
        assert(p_ != nullptr);  // NOLINT(cert-dcl03-c, misc-static-assert): runtime guard
        return *p_;
    }

    [[nodiscard]] explicit operator bool() const noexcept { return p_ != nullptr; }

    template <typename... Args>
    [[nodiscard]] static IntrusivePtr MakeInstance(Args&&... args)
    {
        return IntrusivePtr(new T(std::forward<Args>(args)...));
    }

protected:
    template <bool releasePrev, bool comparePointers, typename U>
    void CopyFrom(const IntrusivePtr<U, Traits>& that) noexcept
    {
        if constexpr (comparePointers)
        {
            if (p_ == that.p_)
            {
                return;
            }
        }

        T* incoming = that.p_;
        if (incoming != nullptr)
        {
            Traits::AddReference(incoming);
        }
        if constexpr (releasePrev)
        {
            ReleaseReference<false>();
        }
        p_ = incoming;
    }

    template <bool releasePrev, typename U>
    void MoveFrom(IntrusivePtr<U, Traits>& that) noexcept
    {
        T* incoming = std::exchange(that.p_, nullptr);
        if constexpr (releasePrev)
        {
            ReleaseReference<false>();
        }
        p_ = incoming;
    }

    void AddReference() noexcept
    {
        if (p_ != nullptr)
        {
            Traits::AddReference(p_);
        }
    }

    template <bool set_to_null>
    void ReleaseReference() noexcept
    {
        if (p_ != nullptr)
        {
            Traits::ReleaseReference(p_);
            if constexpr (set_to_null)
            {
                p_ = nullptr;
            }
        }
    }

private:
    template <typename, typename>
    friend class IntrusivePtr;

    struct StaticCastTag
    {
    };
    struct DynamicCastTag
    {
    };

    template <typename U>
    IntrusivePtr(const IntrusivePtr<U, Traits>& ip, StaticCastTag) noexcept : p_(static_cast<T*>(ip.p_))
    {
        AddReference();
    }

    template <typename U>
    IntrusivePtr(const IntrusivePtr<U, Traits>& ip, DynamicCastTag) noexcept : p_(dynamic_cast<T*>(ip.p_))
    {
        AddReference();
    }

    T* p_ = nullptr;
};

template <typename U, typename T, typename Traits>
[[nodiscard]] IntrusivePtr<U, Traits> static_pointer_cast(const IntrusivePtr<T, Traits>& ip) noexcept
{
    return ip.template StaticCast<U>();
}

template <typename U, typename T, typename Traits>
[[nodiscard]] IntrusivePtr<U, Traits> dynamic_pointer_cast(const IntrusivePtr<T, Traits>& ip) noexcept
{
    return ip.template DynamicCast<U>();
}
}  // namespace edt
