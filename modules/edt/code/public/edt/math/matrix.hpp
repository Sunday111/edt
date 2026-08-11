#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <ranges>

#include "../float_aliases.hpp"
#include "../int_aliases.hpp"

namespace edt
{

template <typename T, size_t num_rows, size_t num_columns>
class Matrix;

template <typename MatrixA, typename MatrixB>
inline constexpr bool kMatricesWithSameSize =
    MatrixA::NumRows() == MatrixB::NumRows() && MatrixA::NumColumns() == MatrixB::NumColumns();

template <typename MatrixA, typename MatrixB>
inline constexpr bool kVectorsWithSameSize =
    MatrixA::IsVector() == MatrixB::IsVector() && MatrixA::Size() == MatrixB::Size();

template <typename T, size_t num_rows, size_t num_columns>
class Matrix
{
public:
    using Component = T;

    [[nodiscard]] static constexpr size_t NumRows() noexcept { return num_rows; }
    [[nodiscard]] static constexpr size_t NumColumns() noexcept { return num_columns; }
    [[nodiscard]] static constexpr size_t Size() noexcept { return NumRows() * NumColumns(); }
    [[nodiscard]] static constexpr bool IsVector() { return NumRows() == 1 or NumColumns() == 1; }

    constexpr Matrix() noexcept = default;
    constexpr Matrix(Matrix&&) noexcept = default;
    constexpr Matrix(const Matrix&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    Matrix& operator=(const Matrix&) noexcept = default;
    constexpr ~Matrix() noexcept = default;

    constexpr Matrix(std::array<T, Size()> data)  // NOLINT
        requires(!IsVector())
        : data_(data)
    {
    }

    constexpr Matrix(T x, T y)
        requires(IsVector() && Size() == 2)
        : data_({x, y})
    {
    }

    constexpr Matrix(T x, T y, T z)
        requires(IsVector() && Size() == 3)
        : data_({x, y, z})
    {
    }

    template <size_t other_rows, size_t other_columns>
    constexpr Matrix(const Matrix<T, other_rows, other_columns> xy, T z)
        requires(
            (other_rows == 1 || other_columns == 1) &&  // xy is a vector
            (other_rows * other_columns == 2) &&        // xy is a 2d vector
            IsVector() && Size() == 3)
        : data_({xy.x(), xy.y(), z})
    {
    }

    constexpr Matrix(T x, T y, T z, T w)
        requires(IsVector() && Size() == 4)
        : data_({x, y, z, w})
    {
    }

    template <size_t other_rows, size_t other_columns>
    constexpr Matrix(const Matrix<T, other_rows, other_columns> xyz, T w)
        requires(
            (other_rows == 1 || other_columns == 1) &&  // xyz is a vector
            (other_rows * other_columns == 3) &&        // xyz is a 3d vector
            IsVector() && Size() == 4)
        : data_({xyz.x(), xyz.y(), xyz.z(), w})
    {
    }

    [[nodiscard]] static constexpr Matrix AxisX() noexcept
        requires(IsVector() && Size() > 0 && Size() <= 4)
    {
        Matrix m;
        m[0] = 1.f;
        return m;
    }

    [[nodiscard]] static constexpr Matrix AxisY() noexcept
        requires(IsVector() && Size() > 1 && Size() <= 4)
    {
        Matrix m;
        m[1] = 1.f;
        return m;
    }

    [[nodiscard]] static constexpr Matrix AxisZ() noexcept
        requires(IsVector() && Size() > 2 && Size() <= 4)
    {
        Matrix m;
        m[2] = 1.f;
        return m;
    }

    // Deducing this covers const/non-const (mainly)
    template <typename Self>
    [[nodiscard]] constexpr auto&& operator[](this Self&& self, size_t row, size_t column)
    {
        return std::forward<Self>(self).data_[row * num_columns + column];
    }

    template <typename Self>
    [[nodiscard]] constexpr auto&& operator()(this Self&& self, size_t row, size_t column)
    {
        return std::forward<Self>(self)[row, column];
    }
    template <typename Self, typename PureSelf = std::decay_t<Self>>
        requires(PureSelf::IsVector() && PureSelf::Size() > 0)
    [[nodiscard]] constexpr auto&& x(this Self&& self)
    {
        return std::forward<Self>(self)[0];
    }

    template <typename Self, typename PureSelf = std::decay_t<Self>>
        requires(PureSelf::IsVector() && PureSelf::Size() > 1)
    [[nodiscard]] constexpr auto&& y(this Self&& self)
    {
        return std::forward<Self>(self)[1];
    }

    template <typename Self, typename PureSelf = std::decay_t<Self>>
        requires(PureSelf::IsVector() && PureSelf::Size() > 2)
    [[nodiscard]] constexpr auto&& z(this Self&& self)
    {
        return std::forward<Self>(self)[2];
    }

    template <typename Self, typename PureSelf = std::decay_t<Self>>
        requires(PureSelf::IsVector() && PureSelf::Size() > 3)
    [[nodiscard]] constexpr auto&& w(this Self&& self)
    {
        return std::forward<Self>(self)[3];
    }

    template <typename Self>
        requires(std::decay_t<Self>::IsVector())
    [[nodiscard]] constexpr auto&& operator[](this Self&& self, const size_t index)
    {
        return std::forward<Self>(self).data_[index];
    }

    template <size_t row, size_t column, typename Self>
        requires(row < num_rows && column < num_columns)
    [[nodiscard]] constexpr auto&& At(this Self&& self)
    {
        return std::forward<Self>(self).data_[row * num_columns + column];
    }

    template <std::convertible_to<T> U>
    [[nodiscard]] constexpr Matrix<U, num_rows, num_columns> Cast() const
    {
        Matrix<U, num_rows, num_columns> result;
        for (size_t index = 0; index != data_.size(); ++index)
        {
            result.data_[index] = static_cast<U>(data_[index]);
        }
        return result;
    }

    [[nodiscard]] static constexpr Matrix Identity() noexcept
    {
        Matrix m{};

        for (size_t k = 0; k != num_rows && k != num_columns; ++k)
        {
            m(k, k) = T{1};
        }

        return m;
    }

    template <size_t other_rows, size_t other_columns>
        requires(kVectorsWithSameSize<Matrix<T, other_rows, other_columns>, Matrix>)
    [[nodiscard]] constexpr T Dot(const Matrix<T, other_rows, other_columns>& other) const
    {
        T sum{0};

        for (size_t i = 0; i != Size(); ++i)
        {
            sum += this->operator[](i) * other[i];
        }

        return sum;
    }

    // https://en.wikipedia.org/wiki/Cross_product
    template <size_t other_rows, size_t other_columns>
        requires(kVectorsWithSameSize<Matrix<T, other_rows, other_columns>, Matrix> && Size() == 3)
    [[nodiscard]] constexpr Matrix Cross(const Matrix<T, other_rows, other_columns>& other) const
    {
        auto& a = *this;
        auto& b = other;
        return Matrix{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
    }

    [[nodiscard]] constexpr T* data() { return data_.data(); }

    [[nodiscard]] const T* data() const { return data_.data(); }

    [[nodiscard]] constexpr bool IsFinite() const
        requires std::floating_point<T>
    {
        return std::ranges::all_of(data_, [](T value) { return std::isfinite(value); });
    }

    // Two-dimensional "cross product". A hack to obtain a magnitude of 3-dimensional cross product
    template <size_t other_rows, size_t other_columns>
        requires(kVectorsWithSameSize<Matrix<T, other_rows, other_columns>, Matrix> && Size() == 2)
    [[nodiscard]] constexpr T Cross(const Matrix<T, other_rows, other_columns>& other) const
    {
        auto& a = *this;
        auto& b = other;
        return a[0] * b[1] - a[1] * b[0];
    }

    [[nodiscard]] constexpr T Magnitude() const
        requires(IsVector())
    {
        return this->Dot(*this);
    }

    [[nodiscard]] constexpr T SquaredLength() const
        requires(IsVector())
    {
        return Magnitude();
    }

    [[nodiscard]] T Length() const
        requires(IsVector())
    {
        return std::sqrt(SquaredLength());
    }

    void Normalize()
        requires(IsVector())
    {
        *this /= Length();
    }

    [[nodiscard]] Matrix Normalized() const
        requires(IsVector())
    {
        auto copy = *this;
        copy.Normalize();
        return copy;
    }

    [[nodiscard]] constexpr auto RowIndices() const { return std::views::iota(size_t{0}, NumRows()); }

    [[nodiscard]] constexpr auto ColumnIndices() const { return std::views::iota(size_t{0}, NumColumns()); }

    [[nodiscard]] constexpr auto Indices() const
        requires(IsVector())
    {
        return std::views::iota(size_t{0}, Size());
    }

    template <size_t other_rows, size_t other_columns>
        requires(NumColumns() == other_rows)
    [[nodiscard]] constexpr Matrix<T, NumRows(), other_columns> MatMul(
        const Matrix<T, other_rows, other_columns>& other) const
    {
        Matrix<T, NumRows(), other_columns> result;

        for (const size_t r : result.RowIndices())
        {
            for (const size_t c : result.ColumnIndices())
            {
                auto& cell = result(r, c);
                for (const size_t k : ColumnIndices())
                {
                    cell += this->operator()(r, k) * other(k, c);
                }
            }
        }

        return result;
    }

    constexpr Matrix& operator+=(const T& value)
    {
        for (T& element : data_) element += value;
        return *this;
    }

    constexpr Matrix operator+(const T& value) const
    {
        auto copy = *this;
        copy += value;
        return copy;
    }

    constexpr Matrix& operator+=(const Matrix& other)
    {
        for (size_t i = 0; i != data_.size(); ++i) data_[i] += other.data_[i];
        return *this;
    }

    constexpr Matrix operator+(const Matrix& other) const
    {
        auto copy = *this;
        copy += other;
        return copy;
    }

    [[nodiscard]] friend constexpr Matrix operator+(const T& value, const Matrix& m) { return m + value; }

    constexpr Matrix& operator-=(const T& value)
    {
        for (T& element : data_) element -= value;
        return *this;
    }

    constexpr Matrix operator-(const T& value) const
    {
        auto copy = *this;
        copy -= value;
        return copy;
    }

    constexpr Matrix& operator-=(const Matrix& other)
    {
        for (size_t i = 0; i != data_.size(); ++i) data_[i] -= other.data_[i];
        return *this;
    }

    constexpr Matrix operator-(const Matrix& value) const
    {
        auto copy = *this;
        copy -= value;
        return copy;
    }

    constexpr Matrix operator-() const
        requires(!std::unsigned_integral<T>)
    {
        Matrix copy = *this;
        for (size_t i = 0; i != data_.size(); ++i) copy.data_[i] = -copy.data_[i];
        return copy;
    }

    [[nodiscard]] friend constexpr Matrix operator-(const T& value, const Matrix& m)
    {
        auto copy = m;
        for (auto& e : copy.data_)
        {
            e = value - e;
        }
        return copy;
    }

    constexpr Matrix& operator*=(const T& value)
    {
        for (T& element : data_) element *= value;
        return *this;
    }

    constexpr Matrix operator*(const T& value) const
    {
        auto copy = *this;
        copy *= value;
        return copy;
    }

    constexpr Matrix& operator*=(const Matrix& other)
    {
        for (size_t i = 0; i != data_.size(); ++i) data_[i] *= other.data_[i];
        return *this;
    }

    constexpr Matrix operator*(const Matrix& value) const
    {
        auto copy = *this;
        copy *= value;
        return copy;
    }

    [[nodiscard]] friend constexpr Matrix operator*(const T& value, const Matrix& m) { return m * value; }

    constexpr Matrix& operator/=(const T& value)
    {
        for (T& element : data_) element /= value;
        return *this;
    }

    constexpr Matrix operator/(const T& value) const
    {
        auto copy = *this;
        copy /= value;
        return copy;
    }

    constexpr Matrix& operator/=(const Matrix& other)
    {
        for (size_t i = 0; i != data_.size(); ++i) data_[i] /= other.data_[i];
        return *this;
    }

    constexpr Matrix operator/(const Matrix& value) const
    {
        auto copy = *this;
        copy /= value;
        return copy;
    }

    [[nodiscard]] friend constexpr Matrix operator/(const T& value, const Matrix& m)
    {
        auto copy = m;
        for (auto& e : copy.data_)
        {
            e = value / e;
        }
        return copy;
    }

    [[nodiscard]] constexpr bool operator==(const Matrix& other) const { return data_ == other.data_; }
    [[nodiscard]] constexpr bool operator!=(const Matrix& other) const { return !(*this == other); }

    [[nodiscard]] constexpr Matrix<T, num_rows, 1> GetColumn(const size_t column_index) const
    {
        Matrix<T, num_rows, 1> column;
        for (const size_t row_index : RowIndices())
        {
            column[row_index] = this->operator()(row_index, column_index);
        }

        return column;
    }

    [[nodiscard]] constexpr Matrix<T, 1, num_columns> GetRow(const size_t row_index) const
    {
        Matrix<T, 1, num_columns> row;
        for (const size_t column_index : ColumnIndices())
        {
            row(0, column_index) = this->operator()(row_index, column_index);
        }

        return row;
    }

    template <size_t nr, size_t nc>
    [[nodiscard]] constexpr Matrix<T, nr, nc> Sub() const noexcept
    {
        Matrix<T, nr, nc> result;
        for (size_t r = 0; r != nr; ++r)
        {
            for (size_t c = 0; c != nc; ++c)
            {
                result(r, c) = (*this)(r, c);
            }
        }
        return result;
    }

    template <typename F, typename R = decltype(std::declval<F>()(std::declval<T>()))>
        requires(std::invocable<F, T>)
    [[nodiscard]] constexpr Matrix<R, num_rows, num_columns> Transform(F&& f) const noexcept
    {
        Matrix<R, num_rows, num_columns> r;
        for (size_t i = 0; i != Size(); ++i)
        {
            r.data_[i] = f(data_[i]);
        }

        return r;
    }

    [[nodiscard]] constexpr Matrix<T, num_columns, num_rows> Transposed() const
    {
        Matrix<T, num_columns, num_rows> r;
        for (const size_t row : RowIndices())
        {
            for (const size_t column : ColumnIndices())
            {
                r(column, row) = (*this)(row, column);
            }
        }

        return r;
    }

    template <size_t other_rows, size_t other_columns>
        requires(Matrix<T, other_rows, other_columns>::IsVector() && (other_rows * other_columns == num_rows))
    constexpr void SetColumn(const size_t column_index, const Matrix<T, other_rows, other_columns>& values)
    {
        for (const size_t row_index : RowIndices())
        {
            (*this)(row_index, column_index) = values[row_index];
        }
    }

    template <size_t other_rows, size_t other_columns>
        requires(Matrix<T, other_rows, other_columns>::IsVector() && (other_rows * other_columns == num_columns))
    constexpr void SetRow(const size_t row_index, const Matrix<T, other_rows, other_columns>& values)
    {
        for (const size_t column_index : ColumnIndices())
        {
            (*this)(row_index, column_index) = values[column_index];
        }
    }

    [[nodiscard]] constexpr T Min() const { return *std::ranges::min_element(data_); }
    [[nodiscard]] constexpr T Max() const { return *std::ranges::max_element(data_); }

    template <typename Self>
        requires(std::is_lvalue_reference_v<Self>)
    [[nodiscard]] constexpr auto RefTuple(this Self&& self)
    {
        return [&]<size_t... indices>(std::index_sequence<indices...>)
        {
            return std::forward_as_tuple(std::forward<Self>(self).data_[indices]...);
        }(std::make_index_sequence<Size()>{});
    }

    template <typename Self>
    [[nodiscard]] constexpr auto Tuple(this Self&& self)
    {
        return [&]<size_t... indices>(std::index_sequence<indices...>)
        {
            return std::make_tuple(std::forward<Self>(self).data_[indices]...);
        }(std::make_index_sequence<Size()>{});
    }

    std::array<T, Size()> data_{};
};

template <typename T>
struct IsMatrixT : std::false_type
{
};

template <typename T, size_t R, size_t C>
struct IsMatrixT<Matrix<T, R, C>> : std::true_type
{
};

template <typename T>
concept IsMatrix = IsMatrixT<T>::value;

namespace lazy_matrix_aliases
{

template <typename T>
using Vec2 = Matrix<T, 2, 1>;

template <typename T>
using Vec3 = Matrix<T, 3, 1>;

template <typename T>
using Vec4 = Matrix<T, 4, 1>;

template <typename T>
using Mat2 = Matrix<T, 2, 2>;

template <typename T>
using Mat3 = Matrix<T, 3, 3>;

template <typename T>
using Mat4 = Matrix<T, 4, 4>;

using Vec2f = Vec2<f32>;
using Vec2i = Vec2<int>;
using Vec3f = Vec3<f32>;
using Vec3i = Vec3<int>;
using Vec2u8 = Vec2<u8>;
using Vec2u32 = Vec2<u32>;
using Vec2i32 = Vec2<i32>;
using Vec3u8 = Vec3<u8>;
using Vec4u8 = Vec4<u8>;
using Vec4f = Vec4<f32>;
using Vec4i = Vec4<int>;
using Mat2f = Mat2<f32>;
using Mat3f = Mat3<f32>;
using Mat4f = Mat4<f32>;

}  // namespace lazy_matrix_aliases

using namespace lazy_matrix_aliases;  // NOLINT

}  // namespace edt
