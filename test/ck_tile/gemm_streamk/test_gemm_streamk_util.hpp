// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
// After the umbrella headers: as a TU's first ck_tile include it enters numeric.hpp's include
// cycle, which on gfx125 reaches mxfp_convert.hpp before numeric_traits is declared.
#include "ck_tile/core/utility/type_traits.hpp"

enum struct GemmPipelineType
{
    Mem,
    CompV3,
    CompV4,
    // GemmPipelineAgBgCrCompTDMProducerConsumer (gfx1250 only): TDM loads into a ring of LDS
    // slots, filled by producer waves and drained by consumer waves.
    CompTDMProducerConsumer
};

enum struct StreamKEpilogueType
{
    CShuffle,
    // TdmEpilogue: stores C through the Tensor Data Mover (gfx1250 only). It overwrites C, so it
    // pairs with linear and tree reduction only.
    Tdm
};

// PCPolicy is the GemmPipelineAgBgCrCompTDMProducerConsumerPolicy; other pipelines ignore it.
template <GemmPipelineType PT, typename Problem, typename PCPolicy>
struct GemmPipelineTypeSelector;

template <typename Problem, typename PCPolicy>
struct GemmPipelineTypeSelector<GemmPipelineType::Mem, Problem, PCPolicy>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrMem<Problem>;
};

template <typename Problem, typename PCPolicy>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV3, Problem, PCPolicy>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
};

template <typename Problem, typename PCPolicy>
struct GemmPipelineTypeSelector<GemmPipelineType::CompV4, Problem, PCPolicy>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompV4<Problem>;
};

template <typename Problem, typename PCPolicy>
struct GemmPipelineTypeSelector<GemmPipelineType::CompTDMProducerConsumer, Problem, PCPolicy>
{
    using pipeline = ck_tile::GemmPipelineAgBgCrCompTDMProducerConsumer<Problem, PCPolicy>;
};

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));

    // The logic below may need to become more advanced once bugs in Stream-K Tile Partitioner are
    // resolved. Because the number of WGs contributing to a macro tile in C may not be the same for
    // all macro tiles in C.

    // Calculate error due to more than 1 WG contributing to the same macro tile in C
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

ck_tile::index_t get_cu_count();

template <typename Tuple>
class TestCkTileStreamK : public ::testing::Test
{
    protected:
    using ALayout                                 = std::tuple_element_t<0, Tuple>;
    using BLayout                                 = std::tuple_element_t<1, Tuple>;
    using CLayout                                 = std::tuple_element_t<2, Tuple>;
    using ADataType                               = std::tuple_element_t<3, Tuple>;
    using BDataType                               = std::tuple_element_t<4, Tuple>;
    using AccDataType                             = std::tuple_element_t<5, Tuple>;
    using CDataType                               = std::tuple_element_t<6, Tuple>;
    using DsLayout                                = ck_tile::tuple<>;
    using DsDataType                              = ck_tile::tuple<>;
    static constexpr ck_tile::index_t M_Tile      = std::tuple_element_t<7, Tuple>::value;
    static constexpr ck_tile::index_t N_Tile      = std::tuple_element_t<8, Tuple>::value;
    static constexpr ck_tile::index_t K_Tile      = std::tuple_element_t<9, Tuple>::value;
    static constexpr ck_tile::index_t M_Warp_Tile = std::tuple_element_t<10, Tuple>::value;
    static constexpr ck_tile::index_t N_Warp_Tile = std::tuple_element_t<11, Tuple>::value;
    static constexpr ck_tile::index_t K_Warp_Tile = std::tuple_element_t<12, Tuple>::value;

    static constexpr bool Persistent        = std::tuple_element_t<13, Tuple>::value;
    static constexpr auto PipelineType      = std::tuple_element_t<14, Tuple>::value;
    static constexpr auto ReductionStrategy = std::tuple_element_t<15, Tuple>::value;
    // Optional trailing parameters, so existing type lists need not spell them out.
    static constexpr auto EpilogueType = ck_tile::tuple_element_or_default_t<
        Tuple,
        16,
        ck_tile::integral_constant<StreamKEpilogueType, StreamKEpilogueType::CShuffle>>::value;
    using PCPolicy = ck_tile::tuple_element_or_default_t<
        Tuple,
        17,
        ck_tile::GemmPipelineAgBgCrCompTDMProducerConsumerPolicy<>>;

    // TDM pipelines clip their loads to the real extents and reject padding.
    static constexpr bool kIsTdmPipeline =
        PipelineType == GemmPipelineType::CompTDMProducerConsumer;

    // The kernel type for a given pad/preshuffle combination, kept separate from invoke_streamk
    // so host-side argument-validation tests can reach Kernel::IsSupportedArgument without
    // allocating device memory or launching.
    template <bool PadM, bool PadN, bool PadK, bool Preshuffle, bool TransposeC>
    struct KernelFor
    {
        static constexpr ck_tile::index_t M_Warp = 2;
        static constexpr ck_tile::index_t N_Warp = 2;
        static constexpr ck_tile::index_t K_Warp = 1;

        // CompTDMProducerConsumer sizes LDS from its own slot ring and ignores this flag; it is
        // set to match what the tile engine emits for it.
        static constexpr bool DoubleSmemBuffer =
            PipelineType == GemmPipelineType::CompV4 || kIsTdmPipeline;
        static constexpr bool StructuredSparsity       = false;
        static constexpr ck_tile::index_t NumWaveGroup = 1;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::StreamKTilePartitioner<GemmShape, ReductionStrategy, Persistent>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<PadM,
                                                                     PadN,
                                                                     PadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     Preshuffle>;

        static constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        // We create the GEMM pipeline without specifying has_hot_loop or tail_num.
        // This is because num_loop can vary (a) per WG and (b) per iteration of the Stream-K
        // while loop. Instead, the pipeline determines has_hot_loop and tail_num at run time
        // from num_loop. This is a similar pattern used by grouped GEMM.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = typename GemmPipelineTypeSelector<PipelineType,
                                                               UniversalGemmProblem,
                                                               PCPolicy>::pipeline;

        using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                 BDataType,
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
                                                                 UniversalGemmProblem::TransposeC>;

        using GemmEpilogue = std::conditional_t<EpilogueType == StreamKEpilogueType::Tdm,
                                                ck_tile::TdmEpilogue<EpilogueProblem>,
                                                ck_tile::CShuffleEpilogue<EpilogueProblem>>;

        using Kernel = ck_tile::StreamKKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    };

    template <bool PadM       = !kIsTdmPipeline,
              bool PadN       = !kIsTdmPipeline,
              bool PadK       = !kIsTdmPipeline,
              bool Preshuffle = false,
              bool TransposeC = false>
    using KernelType = typename KernelFor<PadM, PadN, PadK, Preshuffle, TransposeC>::Kernel;

    // Host-side argument validation without any allocation or launch. The pointers are never
    // dereferenced.
    static bool IsArgSupported(const ck_tile::StreamKHostArgs& args, int num_cu)
    {
        using Kernel = KernelType<>;
        return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args, num_cu, 1));
    }

    // Checks a device result against reference_gemm_gpu on the same device inputs. With integer
    // inputs whose fp32 sums stay below 2^24, every summation order gives the same fp32 value,
    // so @p exact demands equality, provided the results are integers CDataType holds exactly.
    static void ExpectMatchesReference(const ck_tile::HostTensor<CDataType>& c_result,
                                       void* a_dev,
                                       void* b_dev,
                                       ck_tile::index_t M,
                                       ck_tile::index_t N,
                                       ck_tile::index_t K,
                                       ck_tile::index_t stride_A,
                                       ck_tile::index_t stride_B,
                                       ck_tile::index_t stride_C,
                                       ck_tile::index_t num_accumulations_per_tile,
                                       bool exact = false)
    {
        ck_tile::HostTensor<CDataType> c_ref(c_result.mDesc);
        ck_tile::DeviceMem c_ref_buf(c_ref.get_element_space_size_in_bytes());
        c_ref_buf.SetZero();
        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(static_cast<ADataType*>(a_dev),
                                             static_cast<BDataType*>(b_dev),
                                             static_cast<CDataType*>(c_ref_buf.GetDeviceBuffer()),
                                             M,
                                             N,
                                             K,
                                             stride_A,
                                             stride_B,
                                             stride_C);
        c_ref_buf.FromDevice(c_ref.data());

        // The tolerance must bound the magnitude the atomic adds accumulate, whatever its sign.
        float max_accumulated_value = 0.f;
        for(const auto& v : c_ref.mData)
        {
            max_accumulated_value =
                std::max(max_accumulated_value, std::fabs(ck_tile::type_convert<float>(v)));
        }

        if(exact)
        {
            // Every integer up to 2^(mant + 1) is exact in CDataType, so both sides convert the
            // same fp32 sum to the same value whatever rounding mode each conversion uses.
            ASSERT_LE(max_accumulated_value,
                      std::ldexp(1.f, ck_tile::numeric_traits<CDataType>::mant + 1))
                << "an exact comparison needs results CDataType represents exactly";
        }

        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, num_accumulations_per_tile, max_accumulated_value);

        EXPECT_TRUE(ck_tile::check_err(c_result,
                                       c_ref,
                                       "Error: Incorrect results!",
                                       exact ? 0. : rtol_atol.at(ck_tile::number<0>{}),
                                       exact ? 0. : rtol_atol.at(ck_tile::number<1>{})));
    }

    template <bool PadM       = !kIsTdmPipeline,
              bool PadN       = !kIsTdmPipeline,
              bool PadK       = !kIsTdmPipeline,
              bool Preshuffle = false,
              bool TransposeC = false>
    ck_tile::index_t invoke_streamk(const ck_tile::StreamKHostArgs& args,
                                    const ck_tile::stream_config& s,
                                    int num_cu         = 0,
                                    bool require_split = false)
    {
        constexpr int kBlockPerCu = 1;

        using Kernel = KernelType<PadM, PadN, PadK, Preshuffle, TransposeC>;

        auto kargs =
            num_cu > 0 ? Kernel::MakeKernelArgs(args, num_cu, 1) : Kernel::MakeKernelArgs(args);
        if(require_split)
        {
            const auto& partitioner = kargs.tile_partitioner;
            int split_boundaries    = 0;
            for(int cta = 1; cta < partitioner.get_sk_ctas(); ++cta)
            {
                const int start = cta * partitioner.get_iters_per_sk_cta() +
                                  std::min(cta, partitioner.get_extra_iters());
                split_boundaries += start % partitioner.get_iters_per_tile() != 0;
            }
            // A successful data-parallel launch cannot validate cross-workgroup reduction.
            EXPECT_GT(split_boundaries, 0);
        }
        const auto workspace_size = Kernel::GetWorkSpaceSize(kargs);
        ck_tile::DeviceMem workspace_data(workspace_size);
        workspace_data.SetZero();
        kargs.workspace_ptr = workspace_data.GetDeviceBuffer();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            // Since IsSupportedArgument only logs with an enviroment variable set, it's best to
            // throw when we hit an unsupported case.
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        dim3 grid_dims  = Kernel::GridSize(kargs.tile_partitioner);
        dim3 block_dims = Kernel::BlockSize();

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid_dims, block_dims, 0, kargs));

        return kargs.tile_partitioner.estimate_num_wgs_per_tile();
    }

    public:
    void Run(ck_tile::index_t M,
             ck_tile::index_t N,
             ck_tile::index_t K,
             ck_tile::index_t stride_A = 0,
             ck_tile::index_t stride_B = 0,
             ck_tile::index_t stride_C = 0,
             int num_cu                = 0,
             bool require_split        = false)
    {
        // Since M, N, and K will vary depending on the number of CUs, we print it here to
        // facilitate test output readability.
        std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

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

        stride_A = f_get_default_stride(M, K, stride_A, ALayout{});
        stride_B = f_get_default_stride(K, N, stride_B, BLayout{});
        stride_C = f_get_default_stride(M, N, stride_C, CLayout{});

        ck_tile::HostTensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, stride_A, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, stride_B, BLayout{}));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, /*seed*/ 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, /*seed*/ 11940}(b_k_n);

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::StreamKHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
                                      b_k_n_dev_buf.GetDeviceBuffer(),
                                      c_m_n_dev_buf.GetDeviceBuffer(),
                                      M,
                                      N,
                                      K,
                                      stride_A,
                                      stride_B,
                                      stride_C};

        ck_tile::index_t num_accumulations_per_tile = invoke_streamk<>(
            args, ck_tile::stream_config{nullptr, false, 0, 0, 1}, num_cu, require_split);

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        ExpectMatchesReference(c_m_n_dev_result,
                               a_m_k_dev_buf.GetDeviceBuffer(),
                               b_k_n_dev_buf.GetDeviceBuffer(),
                               M,
                               N,
                               K,
                               stride_A,
                               stride_B,
                               stride_C,
                               num_accumulations_per_tile);
    };
};
