#include "ck_tile/core.hpp"

#include <gtest/gtest.h>

namespace {

using namespace ck_tile;

template <typename Rows, typename Cols, typename PaddedRows, typename PaddedCols>
constexpr auto
reordered_padding(Rows rows, Cols cols, PaddedRows padded_rows, PaddedCols padded_cols)
{
    const auto descriptor =
        make_naive_tensor_descriptor(make_tuple(rows, cols), make_tuple(cols, number<1>{}));
    return transform_tensor_descriptor(descriptor,
                                       make_tuple(make_right_pad_transform(cols, padded_cols),
                                                  make_right_pad_transform(rows, padded_rows)),
                                       make_tuple(sequence<1>{}, sequence<0>{}),
                                       make_tuple(sequence<1>{}, sequence<0>{}));
}

constexpr auto square = reordered_padding(number<13>{}, number<29>{}, number<32>{}, number<32>{});
constexpr auto square_lengths = detail::tdm_real_lengths(square);
static_assert(square_lengths[number<0>{}] == 13);
static_assert(square_lengths[number<1>{}] == 29);

TEST(TdmRealLengths, ReorderedSquarePadding)
{
    const auto lengths = detail::tdm_real_lengths(square);
    EXPECT_EQ(lengths[number<0>{}], 13);
    EXPECT_EQ(lengths[number<1>{}], 29);
}

TEST(TdmRealLengths, ReorderedRectangularPadding)
{
    constexpr auto descriptor =
        reordered_padding(number<17>{}, number<43>{}, number<32>{}, number<64>{});
    const auto lengths = detail::tdm_real_lengths(descriptor);
    EXPECT_EQ(lengths[number<0>{}], 17);
    EXPECT_EQ(lengths[number<1>{}], 43);
}

TEST(TdmRealLengths, Unpadded)
{
    constexpr auto descriptor = make_naive_tensor_descriptor(make_tuple(number<17>{}, number<43>{}),
                                                             make_tuple(number<43>{}, number<1>{}));
    const auto lengths        = detail::tdm_real_lengths(descriptor);
    EXPECT_EQ(lengths[number<0>{}], 17);
    EXPECT_EQ(lengths[number<1>{}], 43);
}

TEST(TdmRealLengths, MixedPaddingAndPassThrough)
{
    constexpr auto base = make_naive_tensor_descriptor(make_tuple(number<17>{}, number<43>{}),
                                                       make_tuple(number<43>{}, number<1>{}));
    constexpr auto descriptor =
        transform_tensor_descriptor(base,
                                    make_tuple(make_right_pad_transform(number<43>{}, number<64>{}),
                                               make_pass_through_transform(number<17>{})),
                                    make_tuple(sequence<1>{}, sequence<0>{}),
                                    make_tuple(sequence<1>{}, sequence<0>{}));
    const auto lengths = detail::tdm_real_lengths(descriptor);
    EXPECT_EQ(lengths[number<0>{}], 17);
    EXPECT_EQ(lengths[number<1>{}], 43);
}

TEST(TdmRealLengths, DynamicLengths)
{
    for(index_t rows : {13, 17, 31})
    {
        for(index_t cols : {29, 43, 63})
        {
            const auto descriptor = reordered_padding(rows, cols, index_t{32}, index_t{64});
            const auto lengths    = detail::tdm_real_lengths(descriptor);
            EXPECT_EQ(lengths[number<0>{}], rows);
            EXPECT_EQ(lengths[number<1>{}], cols);
        }
    }
}

} // namespace
