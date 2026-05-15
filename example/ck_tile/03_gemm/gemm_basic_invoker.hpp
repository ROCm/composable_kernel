// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "gemm_utils.hpp"

struct BasicInvoker
{
    template <typename GemmConfig,
              typename ADataType_,
              typename BDataType_,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename CLayout,
              bool Persistent,
              typename CDEElementWise>
    static float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
    {
        // ADataTypeCompute: compute type (tf32_t for TF32 mode, used for warp gemm selection)
        // ADataTypeBuf: buffer/storage type (fp32 when tf32)
        using ADataTypeCompute = ADataType_;
        using BDataTypeCompute = BDataType_;
        using ADataTypeBuf = ck_tile::if_select_t<ADataType_, ck_tile::tf32_t, float, ADataType_>;
        using BDataTypeBuf = ck_tile::if_select_t<BDataType_, ck_tile::tf32_t, float, BDataType_>;

        if constexpr(std::is_same_v<ADataTypeCompute, ck_tile::tf32_t>)
        {
            static_assert(std::is_same_v<ADataTypeCompute, BDataTypeCompute>,
                          "ADataTypeCompute and BDataTypeCompute must be the same");
        }

        if constexpr(Persistent)
        {
            std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;
        }

        constexpr bool is_fp32_input   = std::is_same_v<ADataTypeBuf, float>;
        constexpr bool is_tf32_compute = std::is_same_v<ADataTypeCompute, ck_tile::tf32_t>;

        // This part comes from the Codegen
        constexpr ck_tile::index_t M_Tile = is_fp32_input ? 128 : 256;
        constexpr ck_tile::index_t N_Tile = is_fp32_input ? 128 : 256;
        constexpr ck_tile::index_t K_Tile = 64;

#if CK_TILE_USE_WMMA
        constexpr ck_tile::index_t M_Warp = 4;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 16;
        constexpr ck_tile::index_t N_Warp_Tile = 16;
        constexpr ck_tile::index_t K_Warp_Tile =
            ck_tile::get_k_warp_tile<ADataType_, M_Warp_Tile, true>();
        ck_tile::ignore = is_tf32_compute;
#else
        // gfx950: fp32 uses 16x16x16 tile (native MFMA)
        //         tf32 uses 32x32x16 tile (3x bf16 32x32x16 MFMA emulation)
        constexpr ck_tile::index_t M_Warp = (is_fp32_input && !is_tf32_compute) ? 4 : 2;
        constexpr ck_tile::index_t N_Warp = (is_fp32_input && !is_tf32_compute) ? 4 : 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = (is_fp32_input && !is_tf32_compute) ? 16 : 32;
        constexpr ck_tile::index_t N_Warp_Tile = (is_fp32_input && !is_tf32_compute) ? 16 : 32;
        constexpr ck_tile::index_t K_Warp_Tile = 16;
#endif

        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using CodegenGemmTraits = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                                          GemmConfig::kPadN,
                                                          GemmConfig::kPadK,
                                                          ALayout,
                                                          BLayout,
                                                          CLayout>;

        using AComputeDataType = std::
            conditional_t<std::is_same_v<ADataType_, ck_tile::pk_int4_t>, BDataType_, ADataType_>;
        using BComputeDataType =
            std::conditional_t<std::is_same_v<BDataType_, ck_tile::pk_int4_t> ||
                                   std::is_same_v<BDataType_, ck_tile::pk_fp4_raw_t>,
                               ADataType_,
                               BDataType_>;
        using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataTypeBuf,
                                                                    BDataTypeBuf,
                                                                    AccDataType,
                                                                    CodegenGemmShape,
                                                                    CodegenGemmTraits,
                                                                    AComputeDataType,
                                                                    BComputeDataType>;

        using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataTypeCompute,
                                             BDataTypeCompute,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << CodegenGemmShape::GetName() << '\n'
                      << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                      << "pipeline: " << CodegenGemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        // Declare rotating_mem_ptr here so it stays in scope until it is needed
        std::unique_ptr<ck_tile::RotatingMemWrapper<ADataTypeBuf, BDataTypeBuf>> rotating_mem_ptr;
        std::function<void()> preprocess;

        auto clear_gemm_output = [&]() {
            if(args.k_batch > 1)
                hipGetErrorString(hipMemsetAsync(
                    args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
        };

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<ADataTypeBuf> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataTypeBuf> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            rotating_mem_ptr =
                std::make_unique<ck_tile::RotatingMemWrapper<ADataTypeBuf, BDataTypeBuf>>(
                    kargs.as_ptr[0],
                    kargs.bs_ptr[0],
                    s.rotating_count_,
                    size_a_buffer,
                    size_b_buffer);
            rotating_mem_ptr->Print();

            preprocess = [&]() {
                ck_tile::flush_icache();
                rotating_mem_ptr->Next();
                clear_gemm_output();
            };
        }
        else
        {
            preprocess = clear_gemm_output;
        }

        return ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
};
