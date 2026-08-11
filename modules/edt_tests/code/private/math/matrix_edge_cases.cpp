#include <cmath>
#include <limits>

#include "edt/math/matrix.hpp"
#include "gtest/gtest.h"

using namespace edt::lazy_matrix_aliases;  // NOLINT

// Normalising a zero-length vector divides by zero. For floating point that is not
// a trap - it yields NaN components - so callers must check the length themselves.
// Pinned here because the failure is silent and propagates.
TEST(MatrixEdgeTest, NormalizingAZeroVectorYieldsNaN)  // NOLINT
{
    const Vec3f zero{};
    ASSERT_EQ(zero.Length(), 0.f);

    const Vec3f normalized = zero.Normalized();
    EXPECT_TRUE(std::isnan(normalized.x()));
    EXPECT_TRUE(std::isnan(normalized.y()));
    EXPECT_TRUE(std::isnan(normalized.z()));

    // And the result is not equal to itself, since NaN never is.
    EXPECT_NE(normalized, normalized);
}

// A vector of length one is unchanged, and normalising twice is idempotent.
TEST(MatrixEdgeTest, NormalizationIsIdempotent)  // NOLINT
{
    const Vec3f v{3.f, 0.f, 4.f};
    const Vec3f once = v.Normalized();
    const Vec3f twice = once.Normalized();

    EXPECT_FLOAT_EQ(once.Length(), 1.f);
    for (size_t i = 0; i != 3; ++i) EXPECT_FLOAT_EQ(once[i], twice[i]) << "i = " << i;
}

// SquaredLength avoids the square root, so it must agree with Length exactly on
// values where the root is exact.
TEST(MatrixEdgeTest, SquaredLengthAgreesWithLength)  // NOLINT
{
    const Vec2f v{3.f, 4.f};
    EXPECT_FLOAT_EQ(v.SquaredLength(), 25.f);
    EXPECT_FLOAT_EQ(v.Length(), 5.f);
    EXPECT_FLOAT_EQ(v.Length() * v.Length(), v.SquaredLength());

    // Zero is the only vector with zero length.
    EXPECT_FLOAT_EQ(Vec2f{}.SquaredLength(), 0.f);
}

// An integer vector's Length goes through sqrt and is truncated back to the
// element type, so it is exact only for Pythagorean triples.
TEST(MatrixEdgeTest, IntegerLengthTruncates)  // NOLINT
{
    EXPECT_EQ((Vec2<int>{3, 4}.Length()), 5);
    EXPECT_EQ((Vec2<int>{1, 1}.Length()), 1);  // sqrt(2) truncated
    EXPECT_EQ((Vec2<int>{0, 0}.Length()), 0);
    EXPECT_EQ((Vec2<int>{2, 2}.SquaredLength()), 8);
}

// The defining algebraic properties of the 3d cross product, rather than one
// worked example.
TEST(MatrixEdgeTest, CrossProductProperties)  // NOLINT
{
    const Vec3f a{1.f, 2.f, 3.f};
    const Vec3f b{-4.f, 5.f, 6.f};

    const Vec3f ab = a.Cross(b);
    const Vec3f ba = b.Cross(a);

    // Anti-commutative.
    for (size_t i = 0; i != 3; ++i) EXPECT_FLOAT_EQ(ab[i], -ba[i]) << "i = " << i;

    // Orthogonal to both operands.
    EXPECT_NEAR(ab.Dot(a), 0.f, 1e-4f);
    EXPECT_NEAR(ab.Dot(b), 0.f, 1e-4f);

    // A vector crossed with itself is zero, and so is a cross with a parallel one.
    const Vec3f self = a.Cross(a);
    for (size_t i = 0; i != 3; ++i) EXPECT_FLOAT_EQ(self[i], 0.f) << "i = " << i;

    const Vec3f parallel = a.Cross(a * 3.f);
    for (size_t i = 0; i != 3; ++i) EXPECT_NEAR(parallel[i], 0.f, 1e-4f) << "i = " << i;
}

// The 2d cross is the scalar perp-dot product: zero exactly for parallel vectors,
// and sign-flipping with argument order.
TEST(MatrixEdgeTest, TwoDimensionalCrossIsScalar)  // NOLINT
{
    const Vec2f a{1.f, 0.f};
    const Vec2f b{0.f, 1.f};

    EXPECT_FLOAT_EQ(a.Cross(b), 1.f);
    EXPECT_FLOAT_EQ(b.Cross(a), -1.f);
    EXPECT_FLOAT_EQ(a.Cross(a), 0.f);
    EXPECT_FLOAT_EQ(a.Cross(a * -2.f), 0.f);
}

TEST(MatrixEdgeTest, DotProductProperties)  // NOLINT
{
    const Vec3f a{1.f, 2.f, 3.f};
    const Vec3f b{4.f, -5.f, 6.f};

    EXPECT_FLOAT_EQ(a.Dot(b), b.Dot(a));           // symmetric
    EXPECT_FLOAT_EQ(a.Dot(a), a.SquaredLength());  // with itself, the squared length
    EXPECT_FLOAT_EQ(Vec3f{}.Dot(a), 0.f);          // the zero vector annihilates

    // Orthogonal axes have zero dot product.
    EXPECT_FLOAT_EQ(Vec3f::AxisX().Dot(Vec3f::AxisY()), 0.f);
    EXPECT_FLOAT_EQ(Vec3f::AxisY().Dot(Vec3f::AxisZ()), 0.f);
    EXPECT_FLOAT_EQ(Vec3f::AxisX().Dot(Vec3f::AxisX()), 1.f);
}

// Transposing is its own inverse, including for non-square shapes where the type
// itself changes.
TEST(MatrixEdgeTest, TransposeIsAnInvolution)  // NOLINT
{
    edt::Matrix<int, 2, 3> m{};
    int next = 0;
    for (size_t r = 0; r != 2; ++r)
    {
        for (size_t c = 0; c != 3; ++c) m(r, c) = next++;
    }

    const edt::Matrix<int, 3, 2> t = m.Transposed();
    static_assert(std::is_same_v<decltype(t), const edt::Matrix<int, 3, 2>>);

    for (size_t r = 0; r != 2; ++r)
    {
        for (size_t c = 0; c != 3; ++c) EXPECT_EQ(m(r, c), t(c, r)) << "r = " << r << ", c = " << c;
    }

    EXPECT_EQ(t.Transposed(), m);
}

// Casting follows the element type's own conversion rules: truncation towards
// zero for float to int, and wrapping for the unsigned narrowings.
TEST(MatrixEdgeTest, CastTruncatesLikeTheScalarConversion)  // NOLINT
{
    const Vec3f v{1.9f, -1.9f, 0.5f};
    const auto as_int = v.Cast<int>();

    EXPECT_EQ(as_int.x(), 1);
    EXPECT_EQ(as_int.y(), -1);
    EXPECT_EQ(as_int.z(), 0);

    // Casting to the same type is the identity.
    EXPECT_EQ(v.Cast<float>(), v);

    const auto back = as_int.Cast<float>();
    EXPECT_FLOAT_EQ(back.x(), 1.f);
}

// Element-wise arithmetic must handle a matrix appearing on both sides.
TEST(MatrixEdgeTest, SelfAssigningArithmetic)  // NOLINT
{
    Vec3f v{1.f, 2.f, 3.f};
    v += v;
    EXPECT_EQ(v, (Vec3f{2.f, 4.f, 6.f}));

    v -= v;
    EXPECT_EQ(v, Vec3f{});

    Vec3f w{2.f, 3.f, 4.f};
    w *= w;
    EXPECT_EQ(w, (Vec3f{4.f, 9.f, 16.f}));
}

// Integer element division truncates rather than rounding.
TEST(MatrixEdgeTest, IntegerDivisionTruncates)  // NOLINT
{
    edt::Matrix<int, 1, 3> m{7, -7, 3};
    m /= 2;
    EXPECT_EQ(m[0], 3);
    EXPECT_EQ(m[1], -3);
    EXPECT_EQ(m[2], 1);
}

// Equality is element-wise, so a NaN anywhere makes the matrix unequal to itself.
TEST(MatrixEdgeTest, EqualityIsElementWise)  // NOLINT
{
    const Vec3f a{1.f, 2.f, 3.f};
    Vec3f b = a;
    EXPECT_EQ(a, b);

    b.z() = 3.0001f;
    EXPECT_NE(a, b);

    Vec3f with_nan = a;
    with_nan.y() = std::numeric_limits<float>::quiet_NaN();
    EXPECT_NE(with_nan, with_nan);

    // The two signed zeros compare equal, as the scalars do.
    Vec3f positive_zero{0.f, 0.f, 0.f};
    Vec3f negative_zero{-0.f, -0.f, -0.f};
    EXPECT_EQ(positive_zero, negative_zero);
}

TEST(MatrixEdgeTest, IsFiniteChecksEveryComponent)  // NOLINT
{
    Vec3f value{1.f, -2.f, 0.f};
    EXPECT_TRUE(value.IsFinite());

    value.y() = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(value.IsFinite());

    value = {};
    value.z() = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(value.IsFinite());

    edt::Matrix<float, 2, 2> matrix{};
    EXPECT_TRUE(matrix.IsFinite());

    matrix(1, 0) = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(matrix.IsFinite());
}

// The identity matrix is neutral under multiplication from either side.
TEST(MatrixEdgeTest, IdentityIsNeutralForMatMul)  // NOLINT
{
    const auto identity = edt::Matrix<float, 3, 3>::Identity();
    edt::Matrix<float, 3, 3> m{};
    float next = 1.f;
    for (size_t r = 0; r != 3; ++r)
    {
        for (size_t c = 0; c != 3; ++c) m(r, c) = next++;
    }

    EXPECT_EQ(m.MatMul(identity), m);
    EXPECT_EQ(identity.MatMul(m), m);
    EXPECT_EQ(identity.MatMul(identity), identity);
}

// A one-element matrix is the smallest shape the index arithmetic can get wrong.
// It counts as a vector, so the array constructor is disabled and there is no
// one-argument one: default construction and assignment is the only way in.
TEST(MatrixEdgeTest, SingleElementMatrix)  // NOLINT
{
    edt::Matrix<int, 1, 1> m{};
    EXPECT_EQ(m(0, 0), 0);

    m(0, 0) = 42;
    EXPECT_EQ(m(0, 0), 42);
    EXPECT_EQ(m[0], 42);
    EXPECT_EQ(m.Transposed(), m);
    EXPECT_EQ(m.Cast<float>()(0, 0), 42.f);
}
