#pragma once

#include <functional>

#include "math.hpp"

namespace edt
{

// Generates a transformation matrix with 3 rotations applied in fixed order: yaw -> pitch -> roll
class Rotator
{
    template <typename F>
    constexpr Rotator& OpFloat(F f) noexcept
    {
        yaw = f(yaw);
        pitch = f(pitch);
        roll = f(roll);
        return *this;
    }

    template <typename F>
    constexpr Rotator& OpRotator(F f, const Rotator& rhs) noexcept
    {
        yaw = f(yaw, rhs.yaw);
        pitch = f(pitch, rhs.pitch);
        roll = f(roll, rhs.roll);
        return *this;
    }

public:
    [[nodiscard]] constexpr Mat4f ToMatrix() const noexcept
    {
        // This is an inlined varsion of multiplication between three matrices
        // https://en.wikipedia.org/wiki/Rotation_matrix#General_3D_rotations

        float sa{};
        float ca{};
        float sb{};
        float cb{};
        float sg{};
        float cg{};
        Math::SinCos(Math::DegToRad(roll), sa, ca);
        Math::SinCos(Math::DegToRad(pitch), sb, cb);
        Math::SinCos(Math::DegToRad(yaw), sg, cg);

        Mat4f m;

        m.At<0, 0>() = cb * cg;
        m.At<0, 1>() = sa * sb * cg - ca * sg;
        m.At<0, 2>() = ca * sb * cg + sa * sg;

        m.At<1, 0>() = cb * sg;
        m.At<1, 1>() = sa * sb * sg + ca * cg;
        m.At<1, 2>() = ca * sb * sg - sa * cg;

        m.At<2, 0>() = -sb;
        m.At<2, 1>() = sa * cb;
        m.At<2, 2>() = ca * cb;

        m.At<3, 0>() = 0.f;
        m.At<3, 1>() = 0.f;
        m.At<3, 2>() = 0.f;

        m.At<3, 3>() = 1.f;

        return m;
    }

    constexpr Rotator& operator*=(float x) noexcept
    {
        return OpFloat([&](float y) { return x * y; });
    }

    [[nodiscard]] constexpr Rotator operator*(float x) const noexcept
    {
        Rotator copy = *this;
        copy *= x;
        return copy;
    }

    constexpr Rotator& operator/=(float x) noexcept
    {
        return OpFloat([&](float y) { return y / x; });
    }

    [[nodiscard]] constexpr Rotator operator/(float x) const noexcept
    {
        Rotator copy = *this;
        copy /= x;
        return copy;
    }

    constexpr Rotator& operator+=(float x) noexcept
    {
        return OpFloat([&](float y) { return y + x; });
    }

    [[nodiscard]] constexpr Rotator operator+(float x) const noexcept
    {
        Rotator copy = *this;
        copy += x;
        return copy;
    }

    constexpr Rotator& operator-=(float x) noexcept
    {
        return OpFloat([&](float y) { return y - x; });
    }

    [[nodiscard]] constexpr Rotator operator-(float x) const noexcept
    {
        Rotator copy = *this;
        copy -= x;
        return copy;
    }

    constexpr Rotator& operator+=(const Rotator& rhs) noexcept { return OpRotator(std::plus<>{}, rhs); }

    [[nodiscard]] constexpr Rotator operator+(const Rotator& rhs) const noexcept
    {
        Rotator copy = *this;
        copy += rhs;
        return copy;
    }

    constexpr Rotator& operator-=(const Rotator& rhs) noexcept { return OpRotator(std::minus<float>{}, rhs); }

    [[nodiscard]] constexpr Rotator operator-(const Rotator& rhs) const noexcept
    {
        Rotator copy = *this;
        copy -= rhs;
        return copy;
    }

    // Rotation around Z (Up) axis, degrees
    float yaw = 0.f;

    // Rotation around Y (Right) axis, degrees
    float pitch = 0.f;

    // Rotation around X (Forward) axis, degrees
    float roll = 0.f;
};
}  // namespace edt
