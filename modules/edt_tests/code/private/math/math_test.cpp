#include "edt/math/math.hpp"

#include <limits>
#include <numbers>

#include "test_tools.hpp"

namespace edt
{
static_assert(Math::IsFinite(0.f));
static_assert(Math::IsFinite(-1.0));
static_assert(noexcept(Math::IsFinite(0.f)));
static_assert(!Math::IsFinite(std::numeric_limits<float>::infinity()));
static_assert(!Math::IsFinite(std::numeric_limits<double>::quiet_NaN()));
static_assert(Math::IsFinite(Vec3f{1.f, -2.f, 0.f}));
static_assert(noexcept(Math::IsFinite(Vec3f{})));
static_assert(!Math::IsFinite(Vec3f{1.f, std::numeric_limits<float>::infinity(), 0.f}));
static_assert(Math::IsFinite(Matrix<float, 2, 2>{}));
static_assert(!Math::IsFinite(Matrix<float, 2, 2>{{0.f, 1.f, std::numeric_limits<float>::quiet_NaN(), 3.f}}));

static_assert(
    []
    {
        float s{}, c{};
        edt::Math::SinCos(edt::Math::DegToRad(0.f), s, c);
        ExpectEq(s, 0);
        ExpectEq(c, 1);

        edt::Math::SinCos(edt::Math::DegToRad(30.f), s, c);
        ExpectEq(s, 0.5f);
        ExpectEq(c, std::numbers::sqrt3_v<float> / 2);

        edt::Math::SinCos(edt::Math::DegToRad(45.f), s, c);
        ExpectEq(s, 1.f / std::numbers::sqrt2_v<float>);
        ExpectEq(c, 1.f / std::numbers::sqrt2_v<float>);

        edt::Math::SinCos(edt::Math::DegToRad(60.f), s, c);
        ExpectEq(s, std::numbers::sqrt3_v<float> / 2);
        ExpectEq(c, 0.5f);

        edt::Math::SinCos(edt::Math::DegToRad(90.f), s, c);
        ExpectEq(s, 1.f);
        ExpectEq(c, 0.f);

        edt::Math::SinCos(edt::Math::DegToRad(120.f), s, c);
        ExpectEq(s, std::numbers::sqrt3_v<float> / 2);
        ExpectEq(c, -0.5f);

        edt::Math::SinCos(edt::Math::DegToRad(180.f), s, c);
        ExpectEq(s, 0);
        ExpectEq(c, -1);

        edt::Math::SinCos(edt::Math::DegToRad(270.f), s, c);
        ExpectEq(s, -1);
        ExpectEq(c, 0);

        edt::Math::SinCos(edt::Math::DegToRad(360.f), s, c);
        ExpectEq(s, 0);
        ExpectEq(c, 1);

        return true;
    }(),
    "SinCos test");

static_assert(
    []
    {
        auto m = Math::TranslationMatrix(Vec2f{9, 3});
        ExpectEq(m.GetRow(0), {1, 0, 9});
        ExpectEq(m.GetRow(1), {0, 1, 3});
        ExpectEq(m.GetRow(2), {0, 0, 1});
        return true;
    }(),
    "2d translation matrix test");

static_assert(
    []
    {
        auto m = Math::TranslationMatrix(Vec3f{9, 3, 6});
        ExpectEq(m.GetRow(0), {1, 0, 0, 9});
        ExpectEq(m.GetRow(1), {0, 1, 0, 3});
        ExpectEq(m.GetRow(2), {0, 0, 1, 6});
        ExpectEq(m.GetRow(3), {0, 0, 0, 1});
        return true;
    }(),
    "3d translation matrix test");

static_assert(
    []
    {
        auto m = Math::ScaleMatrix(Vec2f{9, 3});
        ExpectEq(m.GetRow(0), {9, 0, 0});
        ExpectEq(m.GetRow(1), {0, 3, 0});
        ExpectEq(m.GetRow(2), {0, 0, 1});
        return true;
    }(),
    "2d scale matrix test");

static_assert(
    []
    {
        auto m = Math::ScaleMatrix(Vec3f{9, 3, 6});
        ExpectEq(m.GetRow(0), {9, 0, 0, 0});
        ExpectEq(m.GetRow(1), {0, 3, 0, 0});
        ExpectEq(m.GetRow(2), {0, 0, 6, 0});
        ExpectEq(m.GetRow(3), {0, 0, 0, 1});
        return true;
    }(),
    "3d scale matrix test");

static_assert(
    []
    {
        auto m = Math::RotationMatrix2d(Math::DegToRad(30.f));
        ExpectEq(m.GetRow(0), {std::numbers::sqrt3_v<float> / 2, -0.5f, 0});
        ExpectEq(m.GetRow(1), {0.5f, std::numbers::sqrt3_v<float> / 2, 0});
        ExpectEq(m.GetRow(2), {0, 0, 1});

        auto v = Math::TransformVector(Math::RotationMatrix2d(Math::DegToRad(45.f)), Vec2f{1.f, 0.f});
        ExpectEq(v, {1 / std::numbers::sqrt2_v<float>, 1 / std::numbers::sqrt2_v<float>});

        return true;
    }(),
    "2d rotation matrix test");

static_assert(
    []
    {
        auto m = Math::RotationMatrix3dX(Math::DegToRad(30.f));
        ExpectEq(m.GetRow(0), {1, 0, 0, 0});
        ExpectEq(m.GetRow(1), {0, std::numbers::sqrt3_v<float> / 2, -0.5f, 0});
        ExpectEq(m.GetRow(2), {0, 0.5f, std::numbers::sqrt3_v<float> / 2, 0});
        ExpectEq(m.GetRow(3), {0, 0, 0, 1});

        auto v = Math::TransformVector(Math::RotationMatrix3dX(Math::DegToRad(45.f)), Vec3f{0, 1, 0});
        ExpectEq(v, {0.f, 1 / std::numbers::sqrt2_v<float>, 1 / std::numbers::sqrt2_v<float>});

        return true;
    }(),
    "3d rotation matrix around axis x");

static_assert(
    []
    {
        Vec2f a{-100.f, 200.f};
        Vec2f b{-200.f, 100.f};

        ExpectEq(edt::Math::Lerp(a, b, 0.5f), {-150.f, 150.f});
        ExpectEq(edt::Math::Lerp(a, b, 0.f), a);
        ExpectEq(edt::Math::Lerp(a, b, 1.f), b);
        ExpectEq(edt::Math::Lerp(a, b, 2.f), {-300, 0});

        return true;
    }(),
    "Vector lerp");

static_assert(
    []
    {
        Vec2f x{-1.f, 2.f};
        Vec2f min{0.f, 0.f};
        Vec2f max{1.f, 1.f};
        ExpectEq(edt::Math::Clamp(x, min, max), {0.f, 1.f});

        return true;
    }(),
    "Vector Clamp");

static_assert(
    []
    {
        Vec2f x{-1.f, 2.f};
        ExpectEq(edt::Math::Clamp(x, 0.f, 1.f), {0.f, 1.f});

        return true;
    }(),
    "Vector clamp with scalar bounds");

}  // namespace edt
