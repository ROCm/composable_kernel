// Copyright (c) Advanced Micro Devices, Inc. or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

enum class ScaleDataType
{
    E8M0, // 0
    E5M3, // 1
    E4M3  // 2
};

template <typename T>
struct ScaleDataTypeToEnum;

template <>
struct ScaleDataTypeToEnum<e8m0_t>
{
    static constexpr index_t value = static_cast<index_t>(ScaleDataType::E8M0);
};

template <>
struct ScaleDataTypeToEnum<e5m3_t>
{
    static constexpr index_t value = static_cast<index_t>(ScaleDataType::E5M3);
};

template <>
struct ScaleDataTypeToEnum<e4m3_t>
{
    static constexpr index_t value = static_cast<index_t>(ScaleDataType::E4M3);
};

template <bool Value>
struct Clamp : bool_constant<Value>
{
};

template <bool Value>
struct ReuseA : bool_constant<Value>
{
};

template <bool Value>
struct ReuseB : bool_constant<Value>
{
};

template <index_t Value>
struct AScaleDataType : number<Value>
{
};

template <index_t Value>
struct BScaleDataType : number<Value>
{
};

// this is used to insert s_nop after mfma instruction
template <bool Value>
struct PostNop : bool_constant<Value>
{
};

template <index_t Value>
struct OpSelA : number<Value>
{
};

template <index_t Value>
struct OpSelB : number<Value>
{
};

// this is used when TransC is true and A/B swapped
template <bool Value>
struct SwapReuse_ : bool_constant<Value>
{
};

struct WarpGemmDefaultParams
{
    using clamp      = bool_constant<false>;
    using reuse_a    = bool_constant<false>;
    using reuse_b    = bool_constant<false>;
    using post_nop   = bool_constant<false>;
    using op_sel_a   = number<0>;
    using op_sel_b   = number<0>;
    using swap_reuse = bool_constant<false>; // internal use only
    using scale_a    = number<0>;
    using scale_b    = number<0>;
};

template <typename T, template <index_t> class Tag>
struct is_number_tag_instance
{
    static constexpr bool value    = false;
    static constexpr index_t param = 0;
};

template <template <index_t> class Tag, index_t V>
struct is_number_tag_instance<Tag<V>, Tag>
{
    static constexpr bool value    = true;
    static constexpr index_t param = V;
};

template <typename... Params>
class WarpGemmParamsParser
{
    private:
    template <template <bool> class Tag,
              typename Default,
              std::enable_if_t<std::is_same_v<Default, bool_constant<Default::value>>, int> = 0>
    static constexpr bool extract()
    {
        constexpr bool DefaultValue = Default::value;
        return ((std::is_base_of_v<Tag<true>, Params>    ? true
                 : std::is_base_of_v<Tag<false>, Params> ? false
                                                         : DefaultValue) ||
                ...);
    }

    template <template <index_t> class Tag,
              typename Default,
              std::enable_if_t<std::is_same_v<Default, number<Default::value>>, int> = 0>
    static constexpr index_t extract()
    {
        index_t result = Default::value;
        (void)((is_number_tag_instance<Params, Tag>::value
                    ? (result = is_number_tag_instance<Params, Tag>::param, true)
                    : false) ||
               ...);
        return result;
    }
    static constexpr bool swap_reuse  = extract<SwapReuse_, WarpGemmDefaultParams::swap_reuse>();
    static constexpr bool raw_reuse_a = extract<ReuseA, WarpGemmDefaultParams::reuse_a>();
    static constexpr bool raw_reuse_b = extract<ReuseB, WarpGemmDefaultParams::reuse_b>();
    static constexpr index_t raw_op_sel_a = extract<OpSelA, WarpGemmDefaultParams::op_sel_a>();
    static constexpr index_t raw_op_sel_b = extract<OpSelB, WarpGemmDefaultParams::op_sel_b>();
    static constexpr index_t raw_scale_a =
        extract<AScaleDataType, WarpGemmDefaultParams::scale_a>();
    static constexpr index_t raw_scale_b =
        extract<BScaleDataType, WarpGemmDefaultParams::scale_b>();

    public:
    static constexpr bool clamp       = extract<Clamp, WarpGemmDefaultParams::clamp>();
    static constexpr bool post_nop    = extract<PostNop, WarpGemmDefaultParams::post_nop>();
    static constexpr bool reuse_a     = swap_reuse ? raw_reuse_b : raw_reuse_a;
    static constexpr bool reuse_b     = swap_reuse ? raw_reuse_a : raw_reuse_b;
    static constexpr index_t op_sel_a = swap_reuse ? raw_op_sel_b : raw_op_sel_a;
    static constexpr index_t op_sel_b = swap_reuse ? raw_op_sel_a : raw_op_sel_b;
    static constexpr index_t scale_a  = swap_reuse ? raw_scale_b : raw_scale_a;
    static constexpr index_t scale_b  = swap_reuse ? raw_scale_a : raw_scale_b;
};

} // namespace ck_tile
