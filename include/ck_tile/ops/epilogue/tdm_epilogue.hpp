// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {
// the Problem will be reused from CShuffleEpilogueProblem
template <typename Problem_>
struct TdmEpilogue
{
    using Problem                          = remove_cvref_t<Problem_>;
    using AsDataType                       = remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType                       = remove_cvref_t<typename Problem::BsDataType>;
    using AccDataType                      = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                        = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType                       = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout                         = remove_cvref_t<typename Problem::DsLayout>;
    using AComputeDataType                 = remove_cvref_t<typename Problem::AComputeDataType>;
    using BComputeDataType                 = remove_cvref_t<typename Problem::BComputeDataType>;
    static constexpr bool ADataTypeIsTuple = is_detected<is_tuple, AsDataType>::value;
    static constexpr bool BDataTypeIsTuple = is_detected<is_tuple, BsDataType>::value;

    using AsDataTypeTuple = std::conditional_t<ADataTypeIsTuple,
                                               remove_cvref_t<AsDataType>,
                                               remove_cvref_t<tuple<AsDataType>>>;

    using BsDataTypeTuple = std::conditional_t<BDataTypeIsTuple,
                                               remove_cvref_t<BsDataType>,
                                               remove_cvref_t<tuple<BsDataType>>>;

    using ADataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataTypeTuple>>;
    using BDataType  = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataTypeTuple>>;
    using ATypeToUse = std::conditional_t<
        std::is_same_v<AComputeDataType, void>,
        std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>,
        AComputeDataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse = std::conditional_t<
        std::is_same_v<BComputeDataType, void>,
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>,
        BComputeDataType>;
    using ELayout       = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise = remove_cvref_t<typename Problem::CDElementwise>;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;
    static constexpr index_t kBlockSize                    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock                    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock                    = Problem::kNPerBlock;

    static constexpr index_t NumDTensor = Problem::NumDTensor;

    // no use of vector store in TDM epilogue
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC() { return 1; }
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> /*index*/)
    {
        return 1;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                make_tuple(number<kNPerBlock>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kNPerBlock>{}, number<kMPerBlock>{}),
                make_tuple(number<kMPerBlock>{}, number<1>{}));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kMPerBlock * kNPerBlock * sizeof(ODataType);
    }

    template <typename LdsTile, typename InLdsWindow>
    CK_TILE_DEVICE void cast_lds_tile(LdsTile& lds_tile, InLdsWindow& in_lds_window)
    {
        const auto c_warptile_in_tensor_casted = cast_tile_pk<ODataType>(lds_tile);

        store_tile(in_lds_window, c_warptile_in_tensor_casted);
    }

    struct EmptyScale
    {
    };

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM = EmptyScale,
              typename ScaleN = EmptyScale>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows&,
                                   void* p_smem,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        ignore = scale_m;
        ignore = scale_n;
        // TODO : add support for NumDTensor > 0 and scale_m/scale_n
        static_assert(NumDTensor == 0, "NumDTensor currently only supports 0");
        // currently just support direct write to lds and store to global memory using tdm
        static_assert(std::is_same_v<ScaleM, EmptyScale> && std::is_same_v<ScaleN, EmptyScale>,
                      "ScaleM and ScaleN now is EmptyScale when TDM");
        static_assert(kBlockSize % get_warp_size() == 0, "BlockSize must be multiple of WarpSize");
        constexpr index_t waveNum      = kBlockSize / get_warp_size();
        constexpr auto outLdsTileDistr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<waveNum, kMPerBlock / waveNum>, sequence<kNPerBlock>>,
                tuple<sequence<1>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<1, 0>>{},
            bool_constant<true>{});

        TDMConfig tdm_config;

        constexpr auto lds_block_desc = MakeLdsBlockDescriptor();

        auto o_lds_block = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(p_smem), lds_block_desc);

        auto in_lds_window =
            make_tile_window(o_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                             {0, 0},
                             o_acc_tile.get_tile_distribution());

        auto out_lds_window =
            make_tile_window(o_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                             {0, 0},
                             outLdsTileDistr);

        s_wait_tensorcnt_barrier<0 /*tensor_cnt*/, 0 /*lgkmcnt*/>();

        cast_lds_tile(o_acc_tile, in_lds_window);
        block_sync_lds();

        store_tile_tdm(tdm_config, out_dram_window, out_lds_window);
    };
};

namespace impl {

template <index_t kBlockSize, index_t kMPerBlock, index_t kNPerBlock>
CK_TILE_DEVICE constexpr auto tdm_wave_linear_distr()
{
    static_assert(kBlockSize % get_warp_size() == 0,
                  "kBlockSize must be a multiple of the wave size");
    constexpr index_t waveNum = kBlockSize / get_warp_size();
    static_assert(kMPerBlock % waveNum == 0, "kMPerBlock must split evenly across waves");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>,
            tuple<sequence<waveNum, kMPerBlock / waveNum>, sequence<kNPerBlock>>,
            tuple<sequence<1>>,
            tuple<sequence<0>>,
            sequence<1, 2>,
            sequence<1, 0>>{},
        bool_constant<true>{});
}

template <typename ODataType, index_t kMPerBlock, index_t kNPerBlock>
CK_TILE_DEVICE auto tdm_row_major_lds_view(void* p)
{
    constexpr auto desc =
        make_naive_tensor_descriptor(make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                     make_tuple(number<kNPerBlock>{}, number<1>{}));
    return make_tensor_view<address_space_enum::lds>(static_cast<ODataType*>(p), desc);
}

template <typename Problem>
constexpr bool is_tdm_epilogue(const TdmEpilogue<Problem>*)
{
    return true;
}
constexpr bool is_tdm_epilogue(const void*) { return false; }

} // namespace impl

/// True for TdmEpilogue and for epilogues derived from it (e.g. TdmMultiDEpilogue).
template <typename Epilogue>
inline constexpr bool is_tdm_epilogue_v =
    impl::is_tdm_epilogue(static_cast<const remove_cvref_t<Epilogue>*>(nullptr));

template <typename ODataType,
          index_t kBlockSize,
          index_t kMPerBlock,
          index_t kNPerBlockA,
          index_t kNPerBlockB,
          typename ADramWindow,
          typename BDramWindow,
          typename AAccTile,
          typename BAccTile>
CK_TILE_DEVICE void tdm_store_2d_pair(ADramWindow& a_dram_window,
                                      const AAccTile& a_acc_tile,
                                      BDramWindow& b_dram_window,
                                      const BAccTile& b_acc_tile,
                                      void* p_smem)
{
    auto* p_a = static_cast<char*>(p_smem);
    auto* p_b = p_a + kMPerBlock * kNPerBlockA * sizeof(ODataType);

    auto a_lds = impl::tdm_row_major_lds_view<ODataType, kMPerBlock, kNPerBlockA>(p_a);
    auto b_lds = impl::tdm_row_major_lds_view<ODataType, kMPerBlock, kNPerBlockB>(p_b);

    auto a_in = make_tile_window(a_lds,
                                 make_tuple(number<kMPerBlock>{}, number<kNPerBlockA>{}),
                                 {0, 0},
                                 a_acc_tile.get_tile_distribution());
    auto b_in = make_tile_window(b_lds,
                                 make_tuple(number<kMPerBlock>{}, number<kNPerBlockB>{}),
                                 {0, 0},
                                 b_acc_tile.get_tile_distribution());

    auto a_out =
        make_tile_window(a_lds,
                         make_tuple(number<kMPerBlock>{}, number<kNPerBlockA>{}),
                         {0, 0},
                         impl::tdm_wave_linear_distr<kBlockSize, kMPerBlock, kNPerBlockA>());
    auto b_out =
        make_tile_window(b_lds,
                         make_tuple(number<kMPerBlock>{}, number<kNPerBlockB>{}),
                         {0, 0},
                         impl::tdm_wave_linear_distr<kBlockSize, kMPerBlock, kNPerBlockB>());

    TDMConfig tdm_config;

    s_wait_tensorcnt_barrier<0 /*tensorcnt*/, 0 /*lgkmcnt*/>();

    store_tile(a_in, cast_tile_pk<ODataType>(a_acc_tile));
    store_tile(b_in, cast_tile_pk<ODataType>(b_acc_tile));
    block_sync_lds();

    store_tile_tdm(tdm_config, a_dram_window, a_out);
    store_tile_tdm(tdm_config, b_dram_window, b_out);
}

} // namespace ck_tile
