// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

template <typename Tuple>
class TestCkTileWmmaPipeline : public ::testing::Test
{
    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;
    // TODO: expose tile size through test t-param ?

    template <bool PadM, bool PadN, bool PadK>
    void invoke_gemm(const ck_tile::GemmHostArgs</*NumDTensor = 0*/>& args,
                     const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, Tuple>{};
        constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, Tuple>{};
        constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, Tuple>{};

        constexpr ck_tile::index_t M_Warp_Tile = std::tuple_element_t<10, Tuple>{};
        constexpr ck_tile::index_t N_Warp_Tile = std::tuple_element_t<11, Tuple>{};
        constexpr ck_tile::index_t K_Warp_Tile = std::tuple_element_t<12, Tuple>{};

        constexpr ck_tile::index_t M_Wmma_Tile = std::tuple_element_t<13, Tuple>{};
        constexpr ck_tile::index_t N_Wmma_Tile = std::tuple_element_t<14, Tuple>{};
        constexpr ck_tile::index_t K_Wmma_Tile = std::tuple_element_t<15, Tuple>{};

        constexpr ck_tile::index_t M_BlockWarps = M_Tile / M_Warp_Tile;
        constexpr ck_tile::index_t N_BlockWarps = N_Tile / N_Warp_Tile;
        constexpr ck_tile::index_t K_BlockWarps = K_Tile / K_Warp_Tile;

        constexpr bool kPadM = PadM;
        constexpr bool kPadN = PadN;
        constexpr bool kPadK = PadK;

        constexpr int kBlockPerCu = 1;

        // constexpr bool useTrLdPolicy =
        //     std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor> ||
        //     std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
        //  ===============================================

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_BlockWarps, N_BlockWarps, K_BlockWarps>,
                                   ck_tile::sequence<M_Wmma_Tile, N_Wmma_Tile, K_Wmma_Tile>>;
        using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

        using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        // using GemmEpilogue = ck_tile::GemmCShuffleEpilogue<
        //     ck_tile::GemmShuffleEpilogueProblem<AccDataType, CDataType, GemmShape, Traits>>;
        using GemmEpilogue = ck_tile::Default2DEpilogue<
            ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadM, kPadN>>;

        using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrWmma<
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>>;
        printf("GemmShape::NumWarps is %d \n", GemmShape::NumWarps);
        printf("GemmShape name: %s \n", GemmShape::GetName().c_str());
        const ck_tile::index_t k_grain     = args.k_batch * K_Tile;
        const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            // if trans load policy is used, use GemmPipelineWmmaTrLoadPolicy policy, otherwise use
            // the default Policy
            using GemmPipeline = ck_tile::GemmWmmaPipeline<
                ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      GemmShape,
                                                      Traits,
                                                      ck_tile::GemmPipelineScheduler::Default,
                                                      has_hot_loop_v,
                                                      tail_number_v>>;
            using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args:"
                          << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }

            ck_tile::launch_kernel(
                s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        };

        BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }

    public:
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1}; }

    template <bool PadM = true, bool PadN = true, bool PadK = true>
    void Run(const int M,
             const int N,
             const int K,
             const int StrideA = 0,
             const int StrideB = 0,
             const int StrideC = 0)
    {
        for(auto kb : k_batches_)
        {
            RunSingle<false, false, false>(M, N, K, StrideA, StrideB, StrideC, kb);
        }
    }

    template <bool PadM, bool PadN, bool PadK>
    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch = 1)
    {
        using namespace ck_tile::literals;

        auto f_host_tensor_descriptor = [](std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    // give a chance if stride is zero, return a default packed stride
                    if constexpr(std::is_same_v<decltype(layout),
                                                ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        return col;
                    }
                    else
                    {
                        return row;
                    }
                }
                else
                    return stride;
            };

        std::size_t stride_A = f_get_default_stride(M, K, StrideA, ALayout{});
        std::size_t stride_B = f_get_default_stride(K, N, StrideB, BLayout{});
        std::size_t stride_C = f_get_default_stride(M, N, StrideC, CLayout{});

        ck_tile::HostTensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, stride_A, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, stride_B, BLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());

        ck_tile::GemmHostArgs</*NumDTensor = 0*/> args;
        args.a_ptr    = a_m_k_dev_buf.GetDeviceBuffer();
        args.b_ptr    = b_k_n_dev_buf.GetDeviceBuffer();
        args.e_ptr    = c_m_n_dev_buf.GetDeviceBuffer();
        args.k_batch  = kbatch;
        args.M        = M;
        args.N        = N;
        args.K        = K;
        args.stride_A = stride_A;
        args.stride_B = stride_B;
        args.stride_E = stride_C;

        invoke_gemm<PadM, PadN, PadK>(args, ck_tile::stream_config{nullptr, false});

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        bool pass = true;

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_host_ref);
        EXPECT_TRUE(pass);
    }
};
