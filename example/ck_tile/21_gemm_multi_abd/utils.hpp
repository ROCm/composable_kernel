// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

struct AddScale
{
    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};
    static constexpr auto I3 = ck::Number<3>{};

    template <typename E, typename C, typename D0>
    CK_TILE_HOST_DEVICE constexpr void operator()(E& a, const C& a0, const D0& a1) const
    {
        a = scale * (a0 + a1);
    }

    float scale = 1.0;
};

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
    {
        const float x0_f = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                           ck_tile::type_convert<float>(d1);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(A0DataType) < sizeof(B0DataType), A0DataType, B0DataType>;

    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(D0DataType), ComputeTypeAB, D0DataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));

    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType, EDataType, EDataType>(kbatch);

    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType, EDataType, EDataType>(
        max_accumulated_value, kbatch);

    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}
