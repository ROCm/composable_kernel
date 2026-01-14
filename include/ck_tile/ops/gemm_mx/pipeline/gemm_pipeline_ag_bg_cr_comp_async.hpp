// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
//  MX scaling support with OpSel
template <typename Problem>
struct BaseMXGemmPipelineAgBgCrCompAsync
{
    static constexpr index_t PrefetchStages  = 2;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;
    
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop == 1)
        {
            return TailNumber::One;
        }
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::Three;
        }
        else
        {
            return TailNumber::Two;
        }
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        // Handle all the valid cases.
        if(has_hot_loop)
        {
            if(tail_number == TailNumber::Three)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Three>{});
            }
            else if(tail_number == TailNumber::Two)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
        }
        else
        {
            if(tail_number == TailNumber::Three)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Three>{});
            }
            else if(tail_number == TailNumber::Two)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Two>{});
            }
            else
            {
                return (run_func(bool_constant<false>{},
                                 integral_constant<TailNumber, TailNumber::One>{}));
            }
        }
        // If execution reaches here, it's an invalid tail_number because it wasn't handled above.
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_unreachable();
#else
        throw std::logic_error(
            "Invalid TailNumber: Only TailNumber::Three and TailNumber::Two are supported");
#endif
    }
};

/**
 * @brief MX GEMM compute optimized pipeline version async; which is based on V4.
 *
 * This pipeline introduces asynchronous load from global memory to LDS,
 * skipping the intermediate loading into pipeline registers.
 * Supports MX scaling with e8m0 packed values and OpSel.
 */
template <typename Problem, typename Policy = MXGemmPipelineAgBgCrCompAsyncDefaultPolicy>
struct MXGemmPipelineAgBgCrCompAsync : public BaseMXGemmPipelineAgBgCrCompAsync<Problem>
{
    using Base             = BaseMXGemmPipelineAgBgCrCompAsync<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType     = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType     = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using AElementWise = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise = remove_cvref_t<typename Problem::BElementWise>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");
    
    // MX scaling packing constants
    static constexpr int MXdlPack = Policy::MXdlPack;
    static constexpr int NXdlPack = Policy::NXdlPack;
    static constexpr int KXdlPack = Policy::KXdlPack;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "COMPUTE_ASYNC";
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Policy::template IsTransposeC<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0{});
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1{});
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2{});

            constexpr index_t WaveSize = get_warp_size();

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_buffer_load_inst = A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num;
            constexpr auto num_issue            = num_buffer_load_inst;

            static_for<0, num_buffer_load_inst, 1>{}([&](auto i) {
                // TODO: this will likely need to be redesigned after (1) changes to reading from
                // LDS and (2) re-profiling
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0); // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(
                    LLVMSchedGroupMask::DS_READ, 1, 0);                               // DS read : 1
                __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0); // MFMA: 1
                __builtin_amdgcn_sched_group_barrier(
                    LLVMSchedGroupMask::VMEM_READ, 1, 0); // VMEM read :1
                __builtin_amdgcn_sched_group_barrier(
                    LLVMSchedGroupMask::MFMA, C_MFMA_Inst_Num / num_issue - 2, 0); // MFMA : 6
            });
            __builtin_amdgcn_sched_barrier(0);
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename ScaleADramBlockWindowTmp,
                  typename ScaleBDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       const ScaleADramBlockWindowTmp& scale_a_window,
                                       const ScaleBDramBlockWindowTmp& scale_b_window,
                                       index_t num_loop,
                                       void* __restrict__ p_smem_0,
                                       void* __restrict__ p_smem_1) const
        {
            // TODO support multi-ABD
            static_assert(1 == std::tuple_size_v<AsDramBlockWindowTmp>);
            static_assert(1 == std::tuple_size_v<BsDramBlockWindowTmp>);
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;
            // TODO currently fused elementwise are not supported
            ignore = a_element_func;
            ignore = b_element_func;
            static_assert(std::is_same_v<remove_cvref_t<decltype(a_element_func)>,
                                         element_wise::PassThrough>);
            static_assert(std::is_same_v<remove_cvref_t<decltype(b_element_func)>,
                                         element_wise::PassThrough>);
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            ////////////// global window & register /////////////////
            // A DRAM tile window(s) for load
            auto a_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        a_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                        a_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeADramTileDistribution<Problem>());
                },
                number<AsLayout::size()>{});
            // B DRAM window(s) for load
            auto b_tile_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(
                        b_dram_block_window_tmp[number<idx>{}].get_bottom_tensor_view(),
                        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                        b_dram_block_window_tmp[number<idx>{}].get_window_origin(),
                        Policy::template MakeBDramTileDistribution<Problem>());
                },
                number<BsLayout::size()>{});

            ////////////// MX Scale windows /////////////////
            // Get WarpGemm configuration
            using BlockWarps = typename BlockGemmShape::BlockWarps;
            using WarpTile = typename BlockGemmShape::WarpTile;
            constexpr index_t MWarp = BlockWarps::at(I0{});
            constexpr index_t NWarp = BlockWarps::at(I1{});
            constexpr index_t MPerXdl = WarpTile::at(I0{});
            constexpr index_t NPerXdl = WarpTile::at(I1{});
            constexpr index_t KPerXdl = WarpTile::at(I2{});
            
            constexpr index_t ScaleBlockSize = 32;
            
            // Scale A DRAM Window: [MWarp * MPerXdl, kKPerBlock / 32 / KXdlPack]
            auto scale_a_dram_window = make_tile_window(
                scale_a_window.get_bottom_tensor_view(),
                make_tuple(number<MWarp * MPerXdl>{}, number<KPerBlock / ScaleBlockSize / KXdlPack>{}),
                scale_a_window.get_window_origin(),
                Policy::template MakeMX_ScaleA_DramTileDistribution<Problem>());
            
            const auto scale_a_dram_step_m = amd_wave_read_first_lane(
                scale_a_dram_window.get_load_offset(tuple<number<MWarp * MPerXdl>, number<0>>{}));
            const auto scale_a_dram_step_k = amd_wave_read_first_lane(
                scale_a_dram_window.get_load_offset(tuple<number<0>, number<KPerBlock / ScaleBlockSize / KXdlPack>>{}));
            
            // Scale B DRAM Window: [kKPerBlock / 32 / KXdlPack, NWarp * NPerXdl]
            auto scale_b_dram_window = make_tile_window(
                scale_b_window.get_bottom_tensor_view(),
                make_tuple(number<KPerBlock / ScaleBlockSize / KXdlPack>{}, number<NWarp * NPerXdl>{}),
                scale_b_window.get_window_origin(),
                Policy::template MakeMX_ScaleB_DramTileDistribution<Problem>());
            
            const auto scale_b_dram_step_k = amd_wave_read_first_lane(
                scale_b_dram_window.get_load_offset(tuple<number<KPerBlock / ScaleBlockSize / KXdlPack>, number<0>>{}));
            const auto scale_b_dram_step_n = amd_wave_read_first_lane(
                scale_b_dram_window.get_load_offset(tuple<number<0>, number<NWarp * NPerXdl>>{}));

            // this pipeline has a pair of LDS buffers per logical tile
            auto&& [a_lds_block0, b_lds_block0] = Base::GetABLdsTensorViews(p_smem_0);
            auto&& [a_lds_block1, b_lds_block1] = Base::GetABLdsTensorViews(p_smem_1);

            // set up LDS tile shapes
            constexpr auto a_lds_shape = []() {
                if constexpr(is_a_load_tr_v)
                    return make_tuple(number<KPerBlock>{}, number<MPerBlock>{});
                else
                    return make_tuple(number<MPerBlock>{}, number<KPerBlock>{});
            }();

            constexpr auto b_lds_shape = []() {
                if constexpr(is_b_load_tr_v)
                    return make_tuple(number<KPerBlock>{}, number<NPerBlock>{});
                else
                    return make_tuple(number<NPerBlock>{}, number<KPerBlock>{});
            }();

            // LDS tile windows for storing, one per LDS buffer
            auto a_copy_lds_window0 = make_tile_window(a_lds_block0, a_lds_shape, {0, 0});

            auto a_copy_lds_window1 = make_tile_window(a_lds_block1, a_lds_shape, {0, 0});

            auto b_copy_lds_window0 = make_tile_window(b_lds_block0, b_lds_shape, {0, 0});

            auto b_copy_lds_window1 = make_tile_window(b_lds_block1, b_lds_shape, {0, 0});

            // initialize DRAM window steps, used to advance the DRAM windows
            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // read A(0), B(0) from DRAM to LDS window(0)
            // and advance the DRAM windows
            Base::GlobalPrefetchAsync(
                a_copy_lds_window0, a_tile_windows[number<0>{}], a_dram_tile_window_step);
            Base::GlobalPrefetchAsync(
                b_copy_lds_window0, b_tile_windows[number<0>{}], b_dram_tile_window_step);

            // Initialize WarpGemm for MX scaling
            using WarpGemm = typename remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>::WarpGemm;
            using CWarpTensor = typename WarpGemm::CWarpTensor;

            // read A(1), B(1) from DRAM to LDS window(1)
            // and advance the DRAM windows
            Base::GlobalPrefetchAsync(
                a_copy_lds_window1, a_tile_windows[number<0>{}], a_dram_tile_window_step);
            Base::GlobalPrefetchAsync(
                b_copy_lds_window1, b_tile_windows[number<0>{}], b_dram_tile_window_step);

            // tile distribution for the register tiles
            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto BLdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            // register tiles; double buffering -> a register tile corresponds to a LDS tile window
            ALdsTile a_block_tile0, a_block_tile1;
            BLdsTile b_block_tile0, b_block_tile1;

            ////////////// MX Scale register tiles (ping-pong buffers) /////////////////
            // Calculate scale iterations: each scale covers 32 elements in K
            // Each K iteration processes KPerXdl elements
            // Each packed int32 contains KXdlPack scales
            constexpr index_t KIterPerWarp = KPerBlock / KPerXdl;
            constexpr index_t MIterPerWarp = MPerBlock / (MWarp * MPerXdl);
            constexpr index_t NIterPerWarp = NPerBlock / (NWarp * NPerXdl);
            constexpr index_t ScaleKPackedPerIter = (KIterPerWarp * KPerXdl) / (ScaleBlockSize * KXdlPack);
            
            // Load a sample scale tile to get the type
            auto scale_a_sample = load_tile_with_offset(scale_a_dram_window, tuple<number<0>, number<0>>{});
            auto scale_b_sample = load_tile_with_offset(scale_b_dram_window, tuple<number<0>, number<0>>{});
            
            using ScaleTileElementA = remove_cvref_t<decltype(scale_a_sample)>;
            using ScaleTileElementB = remove_cvref_t<decltype(scale_b_sample)>;
            using ScaleATileType = statically_indexed_array<statically_indexed_array<ScaleTileElementA, ScaleKPackedPerIter>, MIterPerWarp>;
            using ScaleBTileType = statically_indexed_array<statically_indexed_array<ScaleTileElementB, ScaleKPackedPerIter>, NIterPerWarp>;
            
            ScaleATileType scale_a_tile_ping, scale_a_tile_pong;
            ScaleBTileType scale_b_tile_ping, scale_b_tile_pong;
            
            // Helper function to load scales
            auto load_scales_ = [&](auto& scale_a, auto& scale_b) {
                // Load scales for each M/N iteration
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    static_for<0, ScaleKPackedPerIter, 1>{}([&](auto kPacked) {
                        scale_a(mIter)(kPacked) = load_tile_with_offset(
                            scale_a_dram_window, mIter * scale_a_dram_step_m + kPacked * scale_a_dram_step_k);
                    });
                });
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    static_for<0, ScaleKPackedPerIter, 1>{}([&](auto kPacked) {
                        // Scale B is [K/32/KXdlPack, N], so K is first dimension
                        scale_b(nIter)(kPacked) = load_tile_with_offset(
                            scale_b_dram_window, kPacked * scale_b_dram_step_k + nIter * scale_b_dram_step_n);
                    });
                });
                move_tile_window(scale_a_dram_window, {0, KPerBlock / ScaleBlockSize / KXdlPack});
                move_tile_window(scale_b_dram_window, {KPerBlock / ScaleBlockSize / KXdlPack, 0});
            };

            constexpr auto a_lds_input_tile_distr = [ALdsTileDistr]() {
                if constexpr(is_a_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(ALdsTileDistr)::DstrEncode,
                            typename Problem::ADataType>::TransposedDstrEncode{});
                else
                    return ALdsTileDistr;
            }();
            constexpr auto b_lds_input_tile_distr = [BLdsTileDistr]() {
                if constexpr(is_b_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(BLdsTileDistr)::DstrEncode,
                            typename Problem::BDataType>::TransposedDstrEncode{});
                else
                    return BLdsTileDistr;
            }();

            // LDS tile windows for reading;
            // they share the data pointer with the LDS windows for storing
            // but also associate with a distribution to produce a register tile when reading
            auto a_lds_ld_window0 =
                make_tile_window(a_lds_block0, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto a_lds_ld_window1 =
                make_tile_window(a_lds_block1, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto b_lds_ld_window0 =
                make_tile_window(b_lds_block0, b_lds_shape, {0, 0}, b_lds_input_tile_distr);
            auto b_lds_ld_window1 =
                make_tile_window(b_lds_block1, b_lds_shape, {0, 0}, b_lds_input_tile_distr);

            static_assert(!(is_tile_window_linear_v<decltype(a_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(a_lds_ld_window1)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window0)>) &&
                              !(is_tile_window_linear_v<decltype(b_lds_ld_window1)>),
                          "LDS windows must not be linear");

            // Create warp-level C tensors (one per M/N iteration)
            statically_indexed_array<statically_indexed_array<CWarpTensor, NIterPerWarp>, MIterPerWarp> c_warp_tensors;
            
            // Initialize C tensors
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    clear_tile(c_warp_tensors(mIter)(nIter));
                });
            });
            
            // Warp GEMM loop with MX scaling
            auto warp_gemm_loop = [&](auto& a_block_tile, auto& b_block_tile, auto& scale_a, auto& scale_b) {
                // Extract A/B values from block tiles to warp iteration structure
                constexpr auto a_warp_y_lengths =
                    to_sequence(typename WarpGemm::AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<WarpGemm::AWarpDstr::NDimY, 0>{};
                constexpr auto b_warp_y_lengths =
                    to_sequence(typename WarpGemm::BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<WarpGemm::BWarpDstr::NDimY, 0>{};

                static_for<0, KIterPerWarp, 1>{}([&](auto k_iter) {
                    // Map k_iter to packed scale index and OpSel
                    constexpr index_t kScalePacked = (k_iter * KPerXdl) / (ScaleBlockSize * KXdlPack);
                    constexpr index_t kScaleInPack = ((k_iter * KPerXdl) / ScaleBlockSize) % KXdlPack;

                    static_for<0, MIterPerWarp, 1>{}([&](auto m_iter) {
                        constexpr auto OpSelA = kScaleInPack;

                        static_for<0, NIterPerWarp, 1>{}([&](auto n_iter) {
                            constexpr auto OpSelB = kScaleInPack;

                            // Extract A/B values for this iteration - create warp tensors
                            typename WarpGemm::AWarpTensor a_warp_tensor{};
                            const auto a_thread_data = a_block_tile.get_y_sliced_thread_data(
                                merge_sequences(sequence<m_iter, k_iter>{}, a_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                            static_for<0, a_warp_tensor.get_thread_buffer_size(), 1>{}([&](auto i) {
                                a_warp_tensor.get_thread_buffer()(i) = a_thread_data[i];
                            });
                            
                            typename WarpGemm::BWarpTensor b_warp_tensor{};
                            const auto b_thread_data = b_block_tile.get_y_sliced_thread_data(
                                merge_sequences(sequence<n_iter, k_iter>{}, b_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                            static_for<0, b_warp_tensor.get_thread_buffer_size(), 1>{}([&](auto i) {
                                b_warp_tensor.get_thread_buffer()(i) = b_thread_data[i];
                            });

                            WarpGemm{}.template operator()<OpSelA, OpSelB>(
                                c_warp_tensors(m_iter)(n_iter),
                                a_warp_tensor,
                                b_warp_tensor,
                                scale_a(m_iter)(number<kScalePacked>{}).get_thread_buffer()[0],
                                scale_b(n_iter)(number<kScalePacked>{}).get_thread_buffer()[0]);
                        });
                    });
                });
            };

            // write to LDS window(0) must complete before the local prefetch
            block_sync_lds_direct_load();
            // read A(0), B(0) from LDS window(0) to pipeline registers(0)
            Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
            Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
            // LDS window(0) contents are overwritten below by global prefetch, need to sync
            block_sync_lds();
            // read A(2), B(2) from DRAM to LDS window(0)
            // and advance the DRAM windows
            Base::GlobalPrefetchAsync(
                a_copy_lds_window0, a_tile_windows[number<0>{}], a_dram_tile_window_step);
            Base::GlobalPrefetchAsync(
                b_copy_lds_window0, b_tile_windows[number<0>{}], b_dram_tile_window_step);

            // Load scales for iteration 0 (ping)
            load_scales_(scale_a_tile_ping, scale_b_tile_ping);
            
            // Load scales for iteration 1 (pong) if needed
            if (num_loop > 1) {
                load_scales_(scale_a_tile_pong, scale_b_tile_pong);
            }

            if(HasHotLoop)
            {
                // we have had 3 global prefetches so far, indexed (0, 1, 2).
                index_t i_global_read = amd_wave_read_first_lane(3);
                // alternate ping: (read to register tile(1), use register tile(0) as gemm input)
                //           pong: (read to register tile(0), use register tile(1) as gemm input)
                do
                {
                    // ping
                    {
                        // read A(i-1), B(i-1) from LDS window(1) to pipeline registers(1)
                        Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                        // LDS window(1) contents are overwritten by global prefetch, need to sync
                        block_sync_lds();
                        // read A(i), B(i) from DRAM to LDS window(1)
                        // and advance the DRAM windows
                        Base::GlobalPrefetchAsync(a_copy_lds_window1,
                                                  a_tile_windows[number<0>{}],
                                                  a_dram_tile_window_step);
                        Base::GlobalPrefetchAsync(b_copy_lds_window1,
                                                  b_tile_windows[number<0>{}],
                                                  b_dram_tile_window_step);
                        // C(i-3) = A(i-3) @ B(i-3) with MX scaling
                        warp_gemm_loop(a_block_tile0, b_block_tile0, scale_a_tile_ping, scale_b_tile_ping);
                        // Load scales for iteration i+2 (ping)
                        if (i_global_read + 2 < num_loop) {
                            load_scales_(scale_a_tile_ping, scale_b_tile_ping);
                        }
                        HotLoopScheduler();
                    }
                    // pong
                    {
                        // write to LDS window(0) must complete before the local prefetch
                        block_sync_lds_direct_load();
                        // read A(i), B(i) from LDS window(0) to pipeline registers(0)
                        Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                        Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                        // LDS window(0) contents are overwritten by global prefetch, need to sync
                        block_sync_lds();
                        // read A(i+1), B(i+1) from DRAM to LDS window(0)
                        // and advance the DRAM windows
                        Base::GlobalPrefetchAsync(a_copy_lds_window0,
                                                  a_tile_windows[number<0>{}],
                                                  a_dram_tile_window_step);
                        Base::GlobalPrefetchAsync(b_copy_lds_window0,
                                                  b_tile_windows[number<0>{}],
                                                  b_dram_tile_window_step);
                        // C(i-2) = A(i-2) @ B(i-2) with MX scaling
                        warp_gemm_loop(a_block_tile1, b_block_tile1, scale_a_tile_pong, scale_b_tile_pong);
                        // Load scales for iteration i+2 (pong)
                        if (i_global_read + 2 < num_loop) {
                            load_scales_(scale_a_tile_pong, scale_b_tile_pong);
                        }
                        HotLoopScheduler();
                    }
                    i_global_read += 2;
                } while(i_global_read < num_loop);
            }

            // 3 block gemms remaining
            if constexpr(TailNum == TailNumber::Three)
            {
                {
                    // read A(num_loop-1), B(num_loop-1) from LDS window(1) to pipeline registers(1)
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    // C(num_loop-2) = A(num_loop-2) @ B(num_loop-2) with MX scaling
                    warp_gemm_loop(a_block_tile0, b_block_tile0, scale_a_tile_ping, scale_b_tile_ping);
                }
                {
                    // write to LDS window(0) must complete before the local prefetch
                    block_sync_lds_direct_load();
                    // read A(num_loop), B(num_loop) from LDS window(0) to pipeline registers(0)
                    Base::LocalPrefetch(a_block_tile0, a_lds_ld_window0, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile0, b_lds_ld_window0, is_b_load_tr_v);
                    // C(num_loop-1) = A(num_loop-1) @ B(num_loop-1) with MX scaling
                    warp_gemm_loop(a_block_tile1, b_block_tile1, scale_a_tile_pong, scale_b_tile_pong);
                }
                {
                    // C(num_loop) = A(num_loop) @ B(num_loop) with MX scaling
                    warp_gemm_loop(a_block_tile0, b_block_tile0, scale_a_tile_ping, scale_b_tile_ping);
                }
            }
            else if(TailNum == TailNumber::Two)
            // 2 block gemms remaining
            {
                {
                    // read A(num_loop), B(num_loop) from LDS window(1) to pipeline registers(1)
                    Base::LocalPrefetch(a_block_tile1, a_lds_ld_window1, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile1, b_lds_ld_window1, is_b_load_tr_v);
                    // C(num_loop-1) = A(num_loop-1) @ B(num_loop-1) with MX scaling
                    warp_gemm_loop(a_block_tile0, b_block_tile0, scale_a_tile_ping, scale_b_tile_ping);
                }
                {
                    // C(num_loop) = A(num_loop) @ B(num_loop) with MX scaling
                    warp_gemm_loop(a_block_tile1, b_block_tile1, scale_a_tile_pong, scale_b_tile_pong);
                }
            }
            else if(TailNum == TailNumber::One)
            {
                block_sync_lds();
                // C(num_loop) = A(num_loop) @ B(num_loop) with MX scaling
                warp_gemm_loop(a_block_tile0, b_block_tile0, scale_a_tile_ping, scale_b_tile_ping);
                __builtin_amdgcn_sched_barrier(0);
            }
            
            // Convert warp-level C tensors to block tile format
            auto c_block_tile = BlockGemm{}.MakeCBlockTile();
            using CWarpDstr = typename WarpGemm::CWarpDstr;
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
            
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    c_block_tile.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensors(mIter)(nIter).get_thread_buffer());
                });
            });
            
            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   const ScaleADramBlockWindowTmp& scale_a_window,
                                   const ScaleBDramBlockWindowTmp& scale_b_window,
                                   index_t num_loop,
                                   void* p_smem_0,
                                   void* p_smem_1) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                scale_a_window,
                scale_b_window,
                num_loop,
                p_smem_0,
                p_smem_1);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    public:
    template <typename ADramBlockWindowTmp, 
              typename BDramBlockWindowTmp,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const ScaleADramBlockWindowTmp& scale_a_window,
                                   const ScaleBDramBlockWindowTmp& scale_b_window,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem_0,
                                   void* __restrict__ p_smem_1) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);

        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                element_wise::PassThrough{},
                b_dram_block_window_tmp,
                element_wise::PassThrough{},
                scale_a_window,
                scale_b_window,
                num_loop,
                p_smem_0,
                p_smem_1);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }
};
} // namespace ck_tile
