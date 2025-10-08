// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "test_gemm_pipeline_smoke_util.hpp"
#include "test_gemm_pipeline_smoke_run_test.inc"

struct GemmConfig_Mfma : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

struct GemmConfig_Wmma : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

template <typename GemmConfig,
          typename ADataType,
          typename BDataType = ADataType,
          typename CDataType = ADataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool run_gemm_test_with_layouts(ck_tile::ArgParser& arg_parser,
                                const ALayout a_layout                  = ALayout{},
                                const BLayout b_layout                  = BLayout{},
                                [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    using AccDataType = typename GemmTypeConfig<ADataType, BDataType, CDataType>::AccDataType;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool persistent              = arg_parser.get_int("persistent");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{1.f, 1.f}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }

    if(GemmConfig::UseStructuredSparsity)
    {
        ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    static_assert(!GemmConfig::PermuteA, "Not implemented");
    if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
    {
        // Permute vector pk_i4x4 data for device implementation
        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        if constexpr(GemmConfig::PermuteB)
        {
            permute_tensor_b<GemmConfig,
                             decltype(b_k_n_dev),
                             ADataType,
                             BDataType,
                             AccDataType,
                             CDataType,
                             ALayout,
                             BLayout,
                             CLayout>(b_k_n_dev);
        }
        permute_vectors_i4x4_b(b_k_n_dev);
        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
    }
    else
    {
        if constexpr(GemmConfig::PermuteB)
        {
            std::cout << "Permute for this DataType is not implemented." << std::endl;
            return false;
        }
        b_k_n_dev_buf.ToDevice(b_k_n.data());
    }

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    invoke_gemm<GemmConfig,
                ADataType,
                BDataType,
                ck_tile::tuple<>,
                AccDataType,
                CDataType,
                ALayout,
                BLayout,
                ck_tile::tuple<>,
                CLayout>(a_m_k_dev_buf,
                         b_k_n_dev_buf,
                         c_m_n_dev_buf,
                         M,
                         N,
                         K,
                         stride_A,
                         stride_B,
                         stride_C,
                         kbatch,
                         n_warmup,
                         n_repeat,
                         persistent);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Restore input for B for gpu reference
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        // memory on host to store gpu reference result
        ck_tile::HostTensor<CDataType> c_m_n_gpu_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        // memory on device to store gpu reference result
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_gpu_ref.get_element_space_size_in_bytes());

        c_m_n_gpu_ref.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buf_ref.GetDeviceBuffer());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_gpu_ref.data());

        const float max_accumulated_value =
            *std::max_element(c_m_n_gpu_ref.mData.begin(), c_m_n_gpu_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                  c_m_n_gpu_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));
        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The GPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          bool Persistent,
          typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)

{
    if constexpr(Persistent)
        std::cout << "WARNING: Ignoring persistent kernel option for basic gemm." << std::endl;
    // The kPadM, kPadN, kPadK & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadM = false;
    constexpr bool kPadN = false;
    constexpr bool kPadK = false;

    constexpr int kBlockPerCu = 1;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
    constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
    constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

    using CodegenGemmTraits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using CodegenPipelineProblem = ck_tile::
        GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenGemmShape, CodegenGemmTraits>;

    using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto memory_operation = memory_operation_.value;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
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
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::GemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw ArgumentsNotSupportedException(
                "Wrong! Arguments not supported! Skipping gemm!\n");
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

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    };

    if(args.k_batch == 1)
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::atomic_add>{});
    }
}

template <typename DateType>
bool run_load_tile()
{
    // Define possible values for each parameter
    std::vector<std::string> m_values = {"32"};
    std::vector<std::string> k_values = {"256"};

}
