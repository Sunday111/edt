#include "edt/math/matrix.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace edt
{
namespace
{

template <typename A, typename B>
concept CanDot = requires(const A& a, const B& b) { a.Dot(b); };

template <typename A, typename B>
concept CanCross = requires(const A& a, const B& b) { a.Cross(b); };

static_assert(Matrix<int, 2, 3>::NumRows() == 2);
static_assert(Matrix<int, 2, 3>::NumColumns() == 3);
static_assert(Matrix<int, 2, 3>::Size() == 6);
static_assert(!Matrix<int, 2, 3>::IsVector());
static_assert(Matrix<int, 1, 3>::IsVector());
static_assert(Matrix<int, 3, 1>::IsVector());
static_assert(IsMatrix<Mat4f>);
static_assert(!IsMatrix<int>);
static_assert(std::same_as<Vec2i, Matrix<int, 2, 1>>);
static_assert(std::same_as<Mat3f, Matrix<float, 3, 3>>);

static_assert(
    []
    {
        const Vec2i xy{1, 2};
        const Vec3i xyz{xy, 3};
        const Vec4i xyzw{xyz, 4};
        const Matrix<int, 2, 2> matrix{{1, 2, 3, 4}};
        return xy == Vec2i{1, 2} && xyz == Vec3i{1, 2, 3} && xyzw == Vec4i{1, 2, 3, 4} && matrix(1, 0) == 3;
    }(),
    "Constructors preserve component order");

static_assert(
    []
    {
        Matrix<int, 2, 3> matrix;
        for (const size_t row : matrix.RowIndices())
        {
            for (const size_t column : matrix.ColumnIndices())
            {
                if (matrix(row, column) != 0) return false;
            }
        }
        return true;
    }(),
    "Default construction zero-initializes every cell");

static_assert(
    []
    {
        Vec4i vector{1, 2, 3, 4};
        vector.x() = 5;
        vector.y() = 6;
        vector.z() = 7;
        vector.w() = 8;
        vector[2] = 9;
        return vector == Vec4i{5, 6, 9, 8};
    }(),
    "Named and indexed vector access is mutable");

static_assert(
    []
    {
        Matrix<int, 2, 3> matrix{{0, 1, 2, 3, 4, 5}};
        matrix.At<0, 2>() = 20;
        matrix(1, 0) = 30;
        const auto& const_matrix = matrix;
        return const_matrix.At<0, 2>() == 20 && const_matrix(1, 0) == 30;
    }(),
    "Matrix cell access supports const and mutable matrices");

static_assert(Vec3i::AxisX() == Vec3i{1, 0, 0});
static_assert(Vec3i::AxisY() == Vec3i{0, 1, 0});
static_assert(Vec3i::AxisZ() == Vec3i{0, 0, 1});

static_assert(
    []
    {
        const auto identity = Matrix<int, 3, 5>::Identity();
        for (const size_t row : identity.RowIndices())
        {
            for (const size_t column : identity.ColumnIndices())
            {
                if (identity(row, column) != static_cast<int>(row == column)) return false;
            }
        }
        return true;
    }(),
    "Rectangular identity matrices initialize their full diagonal");

static_assert(Vec3i{1, 2, 3}.Cast<float>() == Vec3f{1.f, 2.f, 3.f});
static_assert(CanDot<Vec3i, Matrix<int, 1, 3>>);
static_assert(!CanDot<Vec2i, Vec3i>);
static_assert(Vec3i{1, 2, 3}.Dot(Matrix<int, 1, 3>{4, 5, 6}) == 32);
static_assert(CanCross<Vec2i, Matrix<int, 1, 2>>);
static_assert(CanCross<Vec3i, Matrix<int, 1, 3>>);
static_assert(!CanCross<Vec4i, Vec4i>);
static_assert(Vec2i{1, 2}.Cross(Matrix<int, 1, 2>{4, 5}) == -3);
static_assert(Vec3i{1, 2, 3}.Cross(Matrix<int, 1, 3>{4, 5, 6}) == Vec3i{-3, 6, -3});
static_assert(Vec3i{1, 2, 3}.SquaredLength() == 14);

static_assert(
    []
    {
        const Matrix<int, 2, 3> lhs{{1, 2, 3, 4, 5, 6}};
        const Matrix<int, 3, 2> rhs{{7, 8, 9, 10, 11, 12}};
        return lhs.MatMul(rhs) == Matrix<int, 2, 2>{{58, 64, 139, 154}};
    }(),
    "Rectangular matrix multiplication");

static_assert(
    []
    {
        const Matrix<int, 2, 3> matrix{{1, 2, 3, 4, 5, 6}};
        const Vec3i vector{7, 8, 9};
        return matrix.MatMul(vector) == Vec2i{50, 122};
    }(),
    "Matrix-vector multiplication");

static_assert(
    []
    {
        Matrix<int, 2, 2> matrix{{1, 2, 3, 4}};
        matrix += 2;
        matrix -= 1;
        matrix *= 3;
        matrix /= 2;
        return matrix == Matrix<int, 2, 2>{{3, 4, 6, 7}};
    }(),
    "Compound scalar arithmetic");

static_assert(
    []
    {
        const Matrix<int, 2, 2> matrix{{1, 2, 4, 5}};
        return matrix + 2 == Matrix<int, 2, 2>{{3, 4, 6, 7}} && 2 + matrix == Matrix<int, 2, 2>{{3, 4, 6, 7}} &&
               matrix - 2 == Matrix<int, 2, 2>{{-1, 0, 2, 3}} && 10 - matrix == Matrix<int, 2, 2>{{9, 8, 6, 5}} &&
               matrix * 2 == Matrix<int, 2, 2>{{2, 4, 8, 10}} && 2 * matrix == Matrix<int, 2, 2>{{2, 4, 8, 10}} &&
               matrix / 2 == Matrix<int, 2, 2>{{0, 1, 2, 2}} && 20 / matrix == Matrix<int, 2, 2>{{20, 10, 5, 4}};
    }(),
    "Binary scalar arithmetic works from both sides");

static_assert(
    []
    {
        Matrix<int, 2, 2> matrix{{2, 4, 6, 8}};
        const Matrix<int, 2, 2> other{{1, 2, 3, 4}};
        matrix += other;
        matrix -= other;
        matrix *= other;
        matrix /= other;
        return matrix == Matrix<int, 2, 2>{{2, 4, 6, 8}};
    }(),
    "Compound component-wise matrix arithmetic");

static_assert(
    []
    {
        const Matrix<int, 2, 2> lhs{{2, 4, 6, 8}};
        const Matrix<int, 2, 2> rhs{{1, 2, 3, 4}};
        return lhs + rhs == Matrix<int, 2, 2>{{3, 6, 9, 12}} && lhs - rhs == Matrix<int, 2, 2>{{1, 2, 3, 4}} &&
               lhs * rhs == Matrix<int, 2, 2>{{2, 8, 18, 32}} && lhs / rhs == Matrix<int, 2, 2>{{2, 2, 2, 2}} &&
               -rhs == Matrix<int, 2, 2>{{-1, -2, -3, -4}};
    }(),
    "Binary component-wise matrix arithmetic and negation");

static_assert(
    []
    {
        Matrix<int, 3, 4> matrix{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
        if (matrix.GetRow(1) != Matrix<int, 1, 4>{4, 5, 6, 7}) return false;
        if (matrix.GetColumn(2) != Vec3i{2, 6, 10}) return false;
        matrix.SetRow(0, Vec4i{20, 21, 22, 23});
        matrix.SetColumn(1, Matrix<int, 1, 3>{30, 31, 32});
        return matrix.GetRow(0) == Matrix<int, 1, 4>{20, 30, 22, 23} && matrix.GetColumn(1) == Vec3i{30, 31, 32};
    }(),
    "Rows and columns can be read and replaced with either vector orientation");

static_assert(
    []
    {
        const Matrix<int, 3, 4> matrix{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
        return matrix.Sub<2, 3>() == Matrix<int, 2, 3>{{0, 1, 2, 4, 5, 6}} &&
               matrix.Transposed() == Matrix<int, 4, 3>{{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}};
    }(),
    "Submatrices and transposition preserve row-major ordering");

static_assert(
    []
    {
        Matrix<int, 2, 3> matrix{{-2, 3, 8, 1, 0, 5}};
        const auto transformed = matrix.Transform([](int value) { return static_cast<float>(value) * 0.5f; });
        return transformed == Matrix<float, 2, 3>{{-1.f, 1.5f, 4.f, 0.5f, 0.f, 2.5f}};
    }(),
    "Transform supports matrices and changing component type");

static_assert(
    []
    {
        const Matrix<int, 2, 3> matrix{{-2, 3, 8, 1, 0, 5}};
        return matrix.Min() == -2 && matrix.Max() == 8 && matrix.Tuple() == std::tuple{-2, 3, 8, 1, 0, 5};
    }(),
    "Reduction and tuple conversion include every component");

static_assert(
    []
    {
        const Vec4i vector{5, 6, 7, 8};
        size_t expected = 0;
        for (const size_t index : vector.Indices())
        {
            if (index != expected++) return false;
        }
        return expected == vector.Size();
    }(),
    "Vector indices span every component");

TEST(MatrixTest, DataProvidesContiguousMutableStorage)
{
    Matrix<int, 2, 3> matrix{{1, 2, 3, 4, 5, 6}};
    int* data = matrix.data();
    EXPECT_EQ(data[0], 1);
    EXPECT_EQ(data[5], 6);
    data[3] = 40;

    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix.data()[3], 40);
    EXPECT_EQ(const_matrix(1, 0), 40);
}

TEST(MatrixTest, FloatingPointLengthAndNormalization)
{
    const Vec2f vector{3.f, 4.f};
    EXPECT_FLOAT_EQ(vector.Length(), 5.f);
    EXPECT_EQ(vector, (Vec2f{3.f, 4.f}));

    const Vec2f normalized = vector.Normalized();
    EXPECT_NEAR(normalized.x(), 0.6f, 0.0001f);
    EXPECT_NEAR(normalized.y(), 0.8f, 0.0001f);
    EXPECT_NEAR(normalized.Length(), 1.f, 0.0001f);

    Vec2f mutable_vector = vector;
    mutable_vector.Normalize();
    EXPECT_NEAR(mutable_vector.x(), 0.6f, 0.0001f);
    EXPECT_NEAR(mutable_vector.y(), 0.8f, 0.0001f);
}

TEST(MatrixTest, CopyAndMovePreserveValues)
{
    const Matrix<int, 2, 2> original{{1, 2, 3, 4}};
    Matrix<int, 2, 2> copied = original;
    // NOLINTNEXTLINE(performance-move-const-arg): a move of a trivially-copyable type is exactly what is tested
    Matrix<int, 2, 2> moved = std::move(copied);
    EXPECT_EQ(moved, original);

    Matrix<int, 2, 2> copy_assigned;
    copy_assigned = original;
    Matrix<int, 2, 2> move_assigned;
    // NOLINTNEXTLINE(performance-move-const-arg): a move of a trivially-copyable type is exactly what is tested
    move_assigned = std::move(copy_assigned);
    EXPECT_EQ(move_assigned, original);
    EXPECT_NE(move_assigned, (Matrix<int, 2, 2>{{4, 3, 2, 1}}));
}

}  // namespace
}  // namespace edt
