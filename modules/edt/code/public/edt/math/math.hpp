#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

#include "../float_aliases.hpp"
#include "../int_aliases.hpp"
#include "matrix.hpp"

namespace edt
{
class Math
{
public:
    // returns true if begin <= x < end, i.e x in [begin; end)
    template <std::integral T>
    [[nodiscard]] static constexpr bool InRange(const T& x, const T& begin, const T& end)
    {
        return x >= begin && x < end;
    }

    template <typename T>
    [[nodiscard]] static constexpr T Sqr(T x)
    {
        return x * x;
    }

    template <std::floating_point T>
    [[nodiscard]] static constexpr bool IsFinite(T value) noexcept
    {
        return std::isfinite(value);
    }

    template <std::floating_point T, size_t rows, size_t columns>
    [[nodiscard]] static constexpr bool IsFinite(const Matrix<T, rows, columns>& value) noexcept
    {
        return value.IsFinite();
    }

    // Get rainbow colors by time t
    [[nodiscard]] static constexpr Vec3<u8> GetRainbowColors(f32 t)
    {
        return {
            static_cast<u8>(255.0f * Sqr(std::sin(t + 0.00f * 2.0f * std::numbers::pi_v<f32>))),
            static_cast<u8>(255.0f * Sqr(std::sin(t + 0.33f * 2.0f * std::numbers::pi_v<f32>))),
            static_cast<u8>(255.0f * Sqr(std::sin(t + 0.66f * 2.0f * std::numbers::pi_v<f32>)))};
    }

    // Get rainbow colors by time t
    [[nodiscard]] static constexpr Vec4<u8> GetRainbowColorsA(f32 t, u8 alpha = 255)
    {
        return {GetRainbowColors(t), alpha};
    }

    template <typename T, size_t rows, size_t columns>
    [[nodiscard]] static constexpr edt::Matrix<T, rows, columns> Clamp(
        const edt::Matrix<T, rows, columns>& v,
        const edt::Matrix<T, rows, columns>& min,
        const edt::Matrix<T, rows, columns>& max)
    {
        edt::Matrix<T, rows, columns> r;
        for (size_t row_index = 0; row_index != rows; ++row_index)
        {
            for (size_t col_index = 0; col_index != columns; ++col_index)
            {
                r(row_index, col_index) =
                    std::clamp(v(row_index, col_index), min(row_index, col_index), max(row_index, col_index));
            }
        }

        return r;
    }

    template <std::floating_point T>
    [[nodiscard]] static constexpr T DegToRad(T degrees)
    {
        return degrees * std::numbers::pi_v<T> / 180;
    }

    template <std::floating_point T>
    [[nodiscard]] static constexpr T RadToDeg(T radians)
    {
        return T{180} * radians / std::numbers::pi_v<T>;
    }

    template <typename T, size_t rows, size_t columns>
    [[nodiscard]] static constexpr edt::Matrix<T, rows, columns>
    Clamp(const edt::Matrix<T, rows, columns>& v, T min, T max)
    {
        edt::Matrix<T, rows, columns> r;
        for (size_t row_index = 0; row_index != rows; ++row_index)
        {
            for (size_t col_index = 0; col_index != columns; ++col_index)
            {
                r(row_index, col_index) = std::clamp(v(row_index, col_index), min, max);
            }
        }

        return r;
    }

    template <std::floating_point T, size_t rows, size_t columns>
    [[nodiscard]] static constexpr edt::Matrix<T, rows, columns>
    Lerp(const edt::Matrix<T, rows, columns>& a, const edt::Matrix<T, rows, columns>& b, T t)
    {
        edt::Matrix<T, rows, columns> r;
        for (size_t row : r.RowIndices())
        {
            for (size_t column : r.ColumnIndices())
            {
                r(row, column) = std::lerp(a(row, column), b(row, column), t);
            }
        }

        return r;
    }

    static constexpr edt::Mat3f TranslationMatrix(Vec2f translation)
    {
        auto m = edt::Mat3f::Identity();
        m.SetColumn(2, Vec3f{translation, 1});
        return m;
    }

    static constexpr edt::Mat4f TranslationMatrix(Vec3f translation)
    {
        auto m = edt::Mat4f::Identity();
        m.SetColumn(3, Vec4f{translation, 1});
        return m;
    }

    static constexpr edt::Mat3f ScaleMatrix(Vec2f scale)
    {
        auto m = edt::Mat3f::Identity();
        m(0, 0) = scale.x();
        m(1, 1) = scale.y();
        return m;
    }

    static constexpr edt::Mat4f ScaleMatrix(Vec3f scale)
    {
        auto m = edt::Mat4f::Identity();
        m(0, 0) = scale.x();
        m(1, 1) = scale.y();
        m(2, 2) = scale.z();
        return m;
    }

    static constexpr void SinCos(f32 angle, f32& scalar_sin, f32& scalar_cos)
    {
        // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
        f32 quotient = (1.f / std::numbers::pi_v<f32>)*0.5f * angle;
        if (angle >= 0.0f)
        {
            quotient = static_cast<f32>(static_cast<i64>(quotient + 0.5f));
        }
        else
        {
            quotient = static_cast<f32>(static_cast<i64>(quotient - 0.5f));
        }
        f32 y = angle - 2 * std::numbers::pi_v<f32> * quotient;

        // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
        f32 sign;  // NOLINT
        if (y > (std::numbers::pi_v<f32> / 2.f))
        {
            y = std::numbers::pi_v<f32> - y;
            sign = -1.0f;
        }
        else if (y < -(std::numbers::pi_v<f32> / 2.f))
        {
            y = -std::numbers::pi_v<f32> - y;
            sign = -1.0f;
        }
        else
        {
            sign = +1.0f;
        }

        f32 y2 = y * y;

        // 11-degree minimax approximation
        scalar_sin = (((((-2.3889859e-08f * y2 + 2.7525562e-06f) * y2 - 0.00019840874f) * y2 + 0.0083333310f) * y2 -
                       0.16666667f) *
                          y2 +
                      1.0f) *
                     y;

        // 10-degree minimax approximation
        f32 p =
            ((((-2.6051615e-07f * y2 + 2.4760495e-05f) * y2 - 0.0013888378f) * y2 + 0.041666638f) * y2 - 0.5f) * y2 +
            1.0f;
        scalar_cos = sign * p;
    }

    [[nodiscard]] static constexpr Mat3f RotationMatrix2d(f32 angle_radians) noexcept
    {
        f32 s, c;  // NOLINT
        SinCos(angle_radians, s, c);

        Mat3f m{};
        m.SetRow(0, Vec3f{c, -s, 0});
        m.SetRow(1, Vec3f{s, c, 0});
        m.SetRow(2, Vec3f{0, 0, 1});

        return m;
    }

    [[nodiscard]] static constexpr Mat4f RotationMatrix3dX(f32 angle_radians)
    {
        f32 s, c;  // NOLINT
        SinCos(angle_radians, s, c);

        Mat4f m{};
        m.SetRow(0, Vec4f{1, 0, 0, 0});
        m.SetRow(1, Vec4f{0, c, -s, 0});
        m.SetRow(2, Vec4f{0, s, c, 0});
        m.SetRow(3, Vec4f{0, 0, 0, 1});
        return m;
    }

    [[nodiscard]] static constexpr Mat4f RotationMatrix3dY(f32 angle_radians)
    {
        f32 s, c;  // NOLINT
        SinCos(angle_radians, s, c);
        Mat4f m{};
        m(0, 0) = c;
        m(0, 2) = s;
        m(1, 1) = 1.f;
        m(2, 0) = -s;
        m(2, 2) = c;
        m(3, 3) = 1.f;
        return m;
    }

    [[nodiscard]] static constexpr Mat4f RotationMatrix3dZ(f32 angle_radians)
    {
        f32 s, c;  // NOLINT
        SinCos(angle_radians, s, c);
        Mat4f m{};
        m(0, 0) = c;
        m(0, 1) = -s;
        m(1, 0) = s;
        m(1, 1) = c;
        m(2, 2) = 1.f;
        m(3, 3) = 1.f;
        return m;
    }

    static constexpr void
    ToBasisVectors(const Mat4f& m, Vec3f* x = nullptr, Vec3f* y = nullptr, Vec3f* z = nullptr) noexcept
    {
        if (x) *x = TransformVector(m, Vec3f::AxisX());
        if (y) *y = TransformVector(m, Vec3f::AxisY());
        if (z) *z = TransformVector(m, Vec3f::AxisZ());
    }

    [[nodiscard]] static constexpr Vec2f TransformPos(const Mat3f& mat, const Vec2f& pos)
    {
        Vec3f v3 = mat.MatMul(Vec3f{pos.x(), pos.y(), 1.f});
        return Vec2f{v3.x(), v3.y()};
    }

    [[nodiscard]] static constexpr Vec2f TransformVector(const Mat3f& mat, const Vec2f& vec)
    {
        Vec3f v3 = mat.MatMul(Vec3f{vec.x(), vec.y(), 0.f});
        return Vec2f{v3.x(), v3.y()};
    }

    [[nodiscard]] static constexpr Vec3f TransformPos(const Mat4f& mat, const Vec3f& pos)
    {
        Vec4f v4 = mat.MatMul(Vec4f{pos.x(), pos.y(), pos.z(), 1.f});
        return Vec3f{v4.x(), v4.y(), v4.z()};
    }

    [[nodiscard]] static constexpr Vec3f TransformVector(const Mat4f& mat, const Vec3f& vec)
    {
        Vec4f v4 = mat.MatMul(Vec4f{vec.x(), vec.y(), vec.z(), 0.f});
        return Vec3f{v4.x(), v4.y(), v4.z()};
    }

    template <typename A, typename B>
    [[nodiscard]] static constexpr auto MatMul(A&& a, B&& b)
    {
        return std::forward<A>(a).MatMul(std::forward<B>(b));
    }

    template <typename A, typename B, typename C>
    [[nodiscard]] static constexpr auto MatMul(A&& a, B&& b, C&& c)
    {
        return (std::forward<A>(a).MatMul(std::forward<B>(b))).MatMul(std::forward<C>(c));
    }

    [[nodiscard]] static constexpr std::vector<Vec2f>
    GenerateSpiralPoints(size_t num_points, Vec2f size, f32 phase = 0.f) noexcept
    {
        std::vector<Vec2f> points;
        points.reserve(num_points);

        const f32 max_r = size.Max() / 2;
        const f32 nf = static_cast<f32>(num_points);
        f32 delta_angle = 4 * std::numbers::pi_v<f32> / nf;

        for (size_t i = 0; i != num_points; ++i)
        {
            const f32 fi = static_cast<f32>(i);
            const f32 t = fi / (nf - 1);
            f32 x, y;  // NOLINT
            edt::Math::SinCos(delta_angle * fi + phase, y, x);

            f32 r = t * max_r;
            points.emplace_back(Vec2f{x, y} * r);
        }

        return points;
    }

    [[nodiscard]] static constexpr Vec2f
    CatmullRom(const Vec2f& p0, const Vec2f& p1, const Vec2f& p2, const Vec2f& p3, f32 t) noexcept
    {
        f32 t2 = t * t;
        f32 t3 = t2 * t;
        return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                       (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
    }

    [[nodiscard]] static constexpr Vec2f CatmullRom(std::span<const Vec2f, 4> segment, f32 t) noexcept
    {
        return CatmullRom(segment[0], segment[1], segment[2], segment[3], t);
    }

    constexpr void SampleCatmullRomPoints(
        std::span<const Vec2f> control_points,
        std::vector<Vec2f>& curve_points,
        size_t samples_per_segment) noexcept
    {
        curve_points.clear();

        const auto n = control_points.size();
        if (n < 4) return;

        curve_points.reserve((n - 3) * samples_per_segment);

        auto i_end = control_points.end() - 3;
        f32 nf = static_cast<f32>(samples_per_segment);
        for (auto i = control_points.begin(); i < i_end; ++i)
        {
            const std::span<const Vec2f, 4> segment{i, i + 4};
            for (size_t j = 0; j <= samples_per_segment; ++j)
            {
                f32 t = static_cast<f32>(j) / nf;
                curve_points.push_back(CatmullRom(segment, t));
            }
        }
    }

    [[nodiscard]] static constexpr edt::Vec2f ComplexMult(edt::Vec2f a, edt::Vec2f b) noexcept
    {
        return {a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x()};
    }

    [[nodiscard]] static Vec2f ComplexExp(Vec2f z) noexcept
    {
        return std::exp(z.x()) * Vec2f{std::cos(z.y()), std::sin(z.y())};
    }

    [[nodiscard]] static Vec2f ComplexPower(Vec2f base, Vec2f power) noexcept
    {
        f32 r = base.Length();
        f32 base_theta = std::atan2(base.y(), base.x());
        Vec2f log_z = {std::log(r), base_theta};
        Vec2f exponent = ComplexMult(power, log_z);
        return ComplexExp(exponent);
    }

    template <typename Callback>
    static void ComplexPowerN(Vec2f base, Vec2f power, size_t branches, Callback cb) noexcept
    {
        f32 r = base.Length();
        f32 base_theta = std::atan2(base.y(), base.x());
        f32 log_r = std::log(r);

        int half = static_cast<int>(branches / 2);
        for (int n = -half / 2; n <= half / 2; ++n)
        {
            f32 theta_n = base_theta + 2 * std::numbers::pi_v<f32> * static_cast<f32>(n);
            Vec2f log_z = {log_r, theta_n};
            Vec2f exponent = ComplexMult(power, log_z);
            cb(ComplexExp(exponent));
        }
    }

    [[nodiscard]] static std::optional<Vec2f> ComplexPower(Vec2f z, f32 a) noexcept
    {
        // Handle zero base
        if (z.x() == 0 && z.y() == 0)
        {
            if (a > 0)
            {
                return std::optional<Vec2f>{std::in_place, Vec2f{0, 0}};
            }

            return std::nullopt;
        }

        // Convert to polar form
        f32 r = z.Length();
        f32 theta = std::atan2(z.y(), z.x());  // angle

        // Compute r^a and a*theta
        f32 r_pow_a = std::pow(r, a);
        f32 a_theta = a * theta;

        // Convert back to rectangular form
        return std::optional<Vec2f>{std::in_place, Vec2f{r_pow_a * std::cos(a_theta), r_pow_a * std::sin(a_theta)}};
    }
};

}  // namespace edt
