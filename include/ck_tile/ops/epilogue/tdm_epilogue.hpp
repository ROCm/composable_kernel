// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

namespace impl {

// Each wave owns a contiguous band of rows, the box of its TDM store.
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

// Dense row-major LDS view (unpadded, unlike CShuffleEpilogue's LDS descriptor).
// The TDM LDS padding fields are documented for loads only (memory to LDS), so a TDM store takes
// its source box from consecutive LDS rows and TDMConfig leaves them disabled here.
// TDM clips against the global tensor extents, so this is unrelated to OOB handling.
// CShuffleEpilogue can pad to avoid LDS bank conflicts because it reads LDS back itself
// through its own descriptor.
template <typename ODataType, index_t kMPerBlock, index_t kNPerBlock>
CK_TILE_DEVICE auto tdm_row_major_lds_view(void* p)
{
    constexpr auto desc =
        make_naive_tensor_descriptor(make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                     make_tuple(number<kNPerBlock>{}, number<1>{}));
    return make_tensor_view<address_space_enum::lds>(static_cast<ODataType*>(p), desc);
}

} // namespace impl

// the Problem will be reused from CShuffleEpilogueProblem
//
// Writes the C tile to LDS, then to global memory with TDM stores. Two modes:
//
// - Full tile (MIterPerStage_ == 0, the default; also TdmEpilogue<Problem> below): all of C goes
//   through LDS at once, with one TDM store per wave. LDS needed: kMPerBlock x kNPerBlock elements.
// - Staged (MIterPerStage_ > 0): C goes through LDS one horizontal slab at a time.
//   A slab is MIterPerStage_ warp tiles tall (MIterPerStage_ * MWave * MPerXdl rows) and the full
//   tile wide.
//   Slabs rotate through NumStageBuffers_ LDS buffers, so LDS needed is NumStageBuffers_ slabs.
//   With two or more buffers, writing one slab overlaps the TDM store of the previous one.
//
// Compared with CShuffleEpilogue: slabs are full width so that each TDM store is one wide box, and
// a buffer is reused only after s_wait_tensorcnt, because the TDM engine reads LDS asynchronously.
template <typename Problem_, index_t MIterPerStage_ = 0, index_t NumStageBuffers_ = 2>
struct TdmEpilogueStaged
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

    /**
     * @brief The TDM store always overwrites its output and ignores the atomic_add memory
     * operation, so this epilogue cannot accumulate split-K partial tiles, whatever the output
     * type. Kernels with a split-K arm read this through epilogue_can_accumulate_v to reject
     * k_batch > 1 on the host and to compile that arm out; an epilogue that does not declare it
     * is assumed able to accumulate. Kernels that do not read it are caught by the trap in
     * operator().
     */
    static constexpr bool SupportsSplitK = false;
    static constexpr bool kIsTdmEpilogue = true;

    // rows that one M iteration covers across the block
    static constexpr index_t kRowsPerMIter = Problem::MWave * Problem::MPerXdl;

    // how many M iterations of each wave (warp-tile rows per wave).
    static constexpr index_t kMIterPerWarp = kMPerBlock / kRowsPerMIter;

    // true when C goes through LDS one slab at a time
    static constexpr bool kStaged = MIterPerStage_ > 0;

    // M iterations per slab
    static constexpr index_t kMIterPerStage = kStaged ? MIterPerStage_ : kMIterPerWarp;

    // Number of slabs
    static constexpr index_t kNumStages = kStaged ? kMIterPerWarp / MIterPerStage_ : 1;

    // Rows of the block tile in one slab
    static constexpr index_t kStageRows = kMIterPerStage * kRowsPerMIter;

    // LDS slab buffers (buffers beyond the number of slabs would never be used)
    static constexpr index_t kNumStageBuffers =
        kStaged ? min(NumStageBuffers_, kNumStages) : index_t{1};

    static_assert(MIterPerStage_ >= 0, "MIterPerStage must be 0 (full tile) or positive");
    static_assert(!kStaged || (kMIterPerWarp > 0 && kMPerBlock % kRowsPerMIter == 0),
                  "TdmEpilogueStaged: kMPerBlock must be a positive multiple of MWave * MPerXdl");
    static_assert(!kStaged || kMIterPerWarp % kMIterPerStage == 0,
                  "MIterPerStage must divide the warp-tile rows per wave");
    static_assert(!kStaged || NumStageBuffers_ >= 1, "NumStageBuffers must be at least 1");

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
        if constexpr(kStaged)
        {
            // Unpadded, unlike CShuffleEpilogue
            return kNumStageBuffers * kStageRows * kNPerBlock * sizeof(ODataType);
        }
        else
        {
            return kMPerBlock * kNPerBlock * sizeof(ODataType);
        }
    }

    // True if slab stage must wait for an earlier slab's TDM store before reusing its buffer.
    CK_TILE_HOST_DEVICE static constexpr bool StageReusesBuffer(index_t stage)
    {
        return stage >= kNumStageBuffers;
    }

    /// Number of workgroup barriers operator() issues. Must stay in sync with operator().
    CK_TILE_HOST_DEVICE static constexpr index_t GetBarrierCount()
    {
        if constexpr(kStaged)
        {
            // 1 barrier at the start
            index_t count = 1;
            for(index_t stage = 0; stage < kNumStages; ++stage)
            {
                // +1 barrier before each slab's TDM store
                // +1 barrier if the slab reuses a buffer
                count += StageReusesBuffer(stage) ? 2 : 1;
            }
            return count;
        }
        else
        {
            // 1 barrier at the start and 1 before the TDM store.
            return 2;
        }
    }

    // Issues the barriers of operator() for wavelet load waves, which hold no C tile.
    CK_TILE_DEVICE static void RunBarrierStub()
    {
        static_for<0, GetBarrierCount(), 1>{}([&](auto) { block_sync_lds(); });
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

        constexpr auto kOutMemOp =
            remove_cvref_t<decltype(out_dram_window.get_bottom_tensor_view())>::DstInMemOp;
        if constexpr(kOutMemOp != memory_operation_enum::set)
        {
            // The TDM store cannot accumulate (see SupportsSplitK): reaching here means a
            // split-K launch the host should have rejected. Abort rather than overwrite C with
            // one split's partial sum -- a trap, not a static_assert, because kernels that
            // dispatch on k_batch at run time still instantiate their atomic_add arm.
            ignore = out_dram_window;
            ignore = o_acc_tile;
            ignore = p_smem;
            __builtin_trap();
        }
        else if constexpr(kStaged)
        {
            run_staged(out_dram_window,
                       o_acc_tile,
                       p_smem,
                       [&](auto& stage_tile, auto& in_lds_window, auto /*stage*/) {
                           cast_lds_tile(stage_tile, in_lds_window);
                       });
        }
        else
        {
            constexpr auto outLdsTileDistr =
                impl::tdm_wave_linear_distr<kBlockSize, kMPerBlock, kNPerBlock>();

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

            // Barriers here must match GetBarrierCount(): RunBarrierStub() mirrors them.
            s_wait_tensorcnt_barrier<0 /*tensor_cnt*/, 0 /*lgkmcnt*/>();

            cast_lds_tile(o_acc_tile, in_lds_window);
            block_sync_lds();

            store_tile_tdm(tdm_config, out_dram_window, out_lds_window);
            // The store reads p_smem asynchronously; retire it before the caller reuses LDS.
            s_wait_tensorcnt();
        }
    }

    private:
    // Tile distribution of one slab: the accumulator's, with kMIterPerStage M iterations per wave.
    template <typename AccEncode>
    CK_TILE_HOST_DEVICE static constexpr auto MakeStageDistribution()
    {
        // The asserts check that each slab is a contiguous band of rows (one 2D TDM box).
        static_assert(AccEncode::NDimX == 2, "the accumulator must be a 2D (M, N) tile");
        // Y0 must be the outermost part of M, so slicing Y0 selects whole row bands.
        static_assert(AccEncode::Ys2RHsMajor::at(number<0>{}) == 1 &&
                          AccEncode::Ys2RHsMinor::at(number<0>{}) == 0,
                      "TdmEpilogueStaged: Y0 of the accumulator must index the outermost M "
                      "component (M iterations of a wave)");
        // That part must have length kMIterPerWarp (the M iterations of a wave).
        constexpr auto hs_m = typename AccEncode::HsLengthss{}.at(number<0>{});
        static_assert(hs_m.at(number<0>{}) == kMIterPerWarp &&
                          hs_m.at(number<1>{}) == Problem::MWave,
                      "TdmEpilogueStaged: the accumulator's M must be (MIterPerWarp, MWave, ...)");
        // No lane or wave index (P dimension) may map to it.
        constexpr bool p_indexes_m_iter = [] {
            bool found = false;
            static_for<0, AccEncode::NDimP, 1>{}([&](auto p) {
                constexpr auto major = typename AccEncode::Ps2RHssMajor{}.at(p);
                constexpr auto minor = typename AccEncode::Ps2RHssMinor{}.at(p);
                static_for<0, major.size(), 1>{}(
                    [&](auto i) { found = found || (major.at(i) == 1 && minor.at(i) == 0); });
            });
            return found;
        }();
        static_assert(!p_indexes_m_iter,
                      "TdmEpilogueStaged: no lane/wave dimension may index the M iterations");

        // Same encoding, with the M iterations cut from kMIterPerWarp to kMIterPerStage.
        using StageHsM =
            remove_cvref_t<decltype(hs_m.modify(number<0>{}, number<kMIterPerStage>{}))>;
        using HsN = remove_cvref_t<decltype(typename AccEncode::HsLengthss{}.at(number<1>{}))>;
        return make_static_tile_distribution(
            tile_distribution_encoding<typename AccEncode::RsLengths,
                                       tuple<StageHsM, HsN>,
                                       typename AccEncode::Ps2RHssMajor,
                                       typename AccEncode::Ps2RHssMinor,
                                       typename AccEncode::Ys2RHsMajor,
                                       typename AccEncode::Ys2RHsMinor>{});
    }

    protected:
    // Staged mode of operator(); its barriers must match GetBarrierCount().
    // write_stage(stage_tile, in_lds_window, stage) writes one slab of the accumulator to LDS.
    template <typename ODramWindow, typename OAccTile, typename WriteStage>
    CK_TILE_DEVICE void run_staged(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   void* p_smem,
                                   const WriteStage& write_stage)
    {
        static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                      "TdmEpilogueStaged supports only row-major E");

        using AccDstr             = remove_cvref_t<decltype(o_acc_tile.get_tile_distribution())>;
        constexpr auto stage_dstr = MakeStageDistribution<typename AccDstr::DstrEncode>();
        constexpr auto out_dstr = impl::tdm_wave_linear_distr<kBlockSize, kStageRows, kNPerBlock>();

        constexpr auto y_lengths = to_sequence(AccDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto stage_y_lengths = y_lengths.modify(number<0>{}, number<kMIterPerStage>{});
        constexpr auto y_origin_zeros  = uniform_sequence_gen_t<AccDstr::NDimY, 0>{};

        const auto dram_origin = out_dram_window.get_window_origin();
        TDMConfig tdm_config;

        // The pipeline's LDS traffic must finish before the slabs overwrite its buffers.
        s_wait_tensorcnt_barrier<0 /*tensor_cnt*/, 0 /*lgkmcnt*/>();

        // Slab s covers rows [s * kStageRows, (s + 1) * kStageRows) of the tile.
        static_for<0, kNumStages, 1>{}([&](auto stage) {
            // Slabs rotate through the LDS buffers.
            constexpr index_t buffer = stage.value % kNumStageBuffers;
            if constexpr(StageReusesBuffer(stage.value))
            {
                // Wait until this wave's store of the slab that last used this buffer is done;
                // the barrier extends that to every wave.
                // With one buffer this waits for all stores.
                // With more, it assumes TENSORcnt retires in issue order, as the partial
                // waits in the TDM pipelines already do.
                s_wait_tensorcnt<kNumStageBuffers - 1>();
                block_sync_lds();
            }

            auto lds_view = impl::tdm_row_major_lds_view<ODataType, kStageRows, kNPerBlock>(
                static_cast<ODataType*>(p_smem) + buffer * kStageRows * kNPerBlock);
            auto in_lds_window =
                make_tile_window(lds_view,
                                 make_tuple(number<kStageRows>{}, number<kNPerBlock>{}),
                                 {0, 0},
                                 stage_dstr);
            auto out_lds_window = make_tile_window(
                lds_view, make_tuple(number<kStageRows>{}, number<kNPerBlock>{}), {0, 0}, out_dstr);

            // Same y-slicing as CShuffleEpilogue::slice_acc_tile, cut along M only
            auto stage_tile =
                make_static_distributed_tensor<typename OAccTile::DataType>(stage_dstr);
            stage_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                y_origin_zeros.modify(number<0>{}, number<stage.value * kMIterPerStage>{}),
                stage_y_lengths);
            write_stage(stage_tile, in_lds_window, stage);
            // Every wave's part of the slab must be in LDS before any wave's TDM store reads it.
            block_sync_lds();

            // A slab-tall window at the slab's rows; TDM clips rows past the end of the tensor.
            auto stage_dram_window =
                make_tile_window(out_dram_window.get_bottom_tensor_view(),
                                 make_tuple(number<kStageRows>{}, number<kNPerBlock>{}),
                                 {dram_origin[0] + stage.value * kStageRows, dram_origin[1]});
            store_tile_tdm(tdm_config, stage_dram_window, out_lds_window);
        });

        // The stores read p_smem asynchronously; retire them before the caller reuses LDS.
        s_wait_tensorcnt();
    }
};

// Full-tile TdmEpilogue
template <typename Problem_>
using TdmEpilogue = TdmEpilogueStaged<Problem_>;

// Number of slab buffers that fit in the LDS the compute pipeline already allocates.
template <typename Problem, index_t MIterPerStage>
CK_TILE_HOST_DEVICE constexpr index_t tdm_stage_buffers_for_lds(index_t pipeline_smem_bytes)
{
    using Stage                   = TdmEpilogueStaged<Problem, MIterPerStage, 1>;
    constexpr index_t stage_bytes = Stage::kStageRows * Problem::kNPerBlock *
                                    static_cast<index_t>(sizeof(typename Problem::ODataType));
    return min(Stage::kNumStages, max(index_t{1}, pipeline_smem_bytes / stage_bytes));
}

// TdmEpilogueStaged with as many slab buffers as fit in GemmPipeline_'s LDS.
// Takes the pipeline type, not its GetSmemSize() value: that value differs between the host and
// device passes, so a buffer count in the template arguments can give them different kernel names.
template <typename Problem_, index_t MIterPerStage_, typename GemmPipeline_>
struct TdmEpilogueStagedForPipeline
    : public TdmEpilogueStaged<Problem_,
                               MIterPerStage_,
                               tdm_stage_buffers_for_lds<Problem_, MIterPerStage_>(
                                   GemmPipeline_::GetSmemSize())>
{
};

namespace impl {

template <typename Epilogue, typename = void>
struct is_tdm_epilogue : false_type
{
};

template <typename Epilogue>
struct is_tdm_epilogue<Epilogue, std::void_t<decltype(Epilogue::kIsTdmEpilogue)>>
    : bool_constant<Epilogue::kIsTdmEpilogue>
{
};

} // namespace impl

/// True for TdmEpilogue and for epilogues derived from it (e.g. TdmMultiDEpilogue).
template <typename Epilogue>
inline constexpr bool is_tdm_epilogue_v = impl::is_tdm_epilogue<remove_cvref_t<Epilogue>>::value;

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
