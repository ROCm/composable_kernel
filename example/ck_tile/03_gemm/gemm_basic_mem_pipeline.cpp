
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_basic.hpp"
#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("m", "1024", "m dimension")
        .insert("n", "2048", "n dimension")
        .insert("k", "64", "k dimension")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("e", "1e-5", "Absolute error tolerance")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "10", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename LayoutA, typename LayoutB, typename LayoutC>
float gemm_calc(const gemm_basic_args& args, const ck_tile::stream_config& s)
{
    // ToDo: This will be modified by the codegen code later.
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 128;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadA = true;
    constexpr bool kPadB = true;
    constexpr bool kPadC = true;

    constexpr int kBlockPerCu = 1;

    // ===============================================

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmTilePartitioner<GemmShape>;
    using PipelineProblem =
        ck_tile::BlockGemmUniversalPipelineProblem<ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   GemmShape,
                                                   kPadA,
                                                   kPadB,
                                                   kPadC,
                                                   ck_tile::BlockGemmPipelineScheduler::Intrawave>;

    // The GemmPipeline should also come from the Codegen.
    using GemmPipeline = ck_tile::BlockGemmPipelineAgBgCrMem<PipelineProblem>;
    using GemmEpilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadA, kPadB>>;
    using Kernel =
        ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, LayoutA, LayoutB, LayoutC>;

    auto kargs = Kernel::MakeKargs(args.p_a,
                                   args.p_b,
                                   args.p_c,
                                   args.M,
                                   args.N,
                                   args.K,
                                   args.stride_A,
                                   args.stride_B,
                                   args.stride_C);

    const dim3 grids      = Kernel::GridSize(args.M, args.N, args.kbatch);
    constexpr dim3 blocks = Kernel::BlockSize();

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

template <typename DataType, typename LayoutA, typename LayoutB, typename LayoutC>
float invoke_gemm(ck_tile::DeviceMem& a_buf,
                  ck_tile::DeviceMem& b_buf,
                  ck_tile::DeviceMem& c_buf,
                  const ck_tile::ArgParser& arg_parser)
{

    std::string data_type = arg_parser.get_str("prec");

    if(data_type != DataTypeTraits<DataType>::name)
    {
        std::cerr << "Data type mismatch: expected " << DataTypeTraits<DataType>::name << ", got "
                  << data_type << std::endl;
        return -1; // Or handle the error appropriately
    }

    float epsilon               = arg_parser.get_float("e");
    ck_tile::index_t batch_size = arg_parser.get_int("b");
    ck_tile::index_t M          = arg_parser.get_int("m");
    ck_tile::index_t N          = arg_parser.get_int("n");
    ck_tile::index_t K          = arg_parser.get_int("k");

    ck_tile::index_t stride_a = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_b = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_c = arg_parser.get_int("stride_c");

    gemm_basic_args args;
    args.p_a     = a_buf.GetDeviceBuffer();
    args.p_b     = b_buf.GetDeviceBuffer();
    args.p_c     = c_buf.GetDeviceBuffer();
    args.epsilon = epsilon;
    args.kbatch  = batch_size;
    args.M       = M;
    args.N       = N;
    args.K       = K;

    // Only set stride_M and stride_N if they are non-zero and not equal to K.
    if(stride_a != 0)
    {
        args.stride_A = stride_a;
    }
    else
    {
        if constexpr(std::is_same_v<LayoutA, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            args.stride_A = K;
        }
        else
        {
            args.stride_A = M;
        }
    }

    if(stride_b != 0)
    {
        args.stride_B = stride_b;
    }
    else
    {
        if constexpr(std::is_same_v<LayoutB, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            args.stride_B = N;
        }
        else
        {
            args.stride_B = K;
        }
    }

    if(stride_c != 0)
    {
        args.stride_C = stride_c;
    }
    else
    {
        if constexpr(std::is_same_v<LayoutC, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            args.stride_C = N;
        }
        else
        {
            args.stride_C = M;
        }
    }

    float ave_time =
        gemm_calc<LayoutA, LayoutB, LayoutC>(args, ck_tile::stream_config{nullptr, true});
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "The overall perfomance of the GEMM with "
              << "[" << data_type << "]"
              << "batch size: " << batch_size << ". m:" << M << ",n:" << N << ", k:" << K
              << "is: \n";
    std::cout << "Running time :" << ave_time << "ms, Throughput" << gb_per_sec << "GB/s \n"
              << std::flush;

    return ave_time;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    // The Matrix Multiplication goes with Matrix A (M, K), Matrix B (N, K) = Matrix C (M, N).
    using matrix_a_layout = ck_tile::tensor_layout::gemm::RowMajor;
    using matrix_b_layout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using matrix_c_layout = ck_tile::tensor_layout::gemm::RowMajor;

    // host verify
    std::vector<int> a_dimensions =
        (std::is_same_v<matrix_a_layout, ck_tile::tensor_layout::gemm::RowMajor>)
            ? std::vector<int>{M, K}
            : std::vector<int>{K, M};
    std::vector<int> b_dimensions =
        (std::is_same_v<matrix_b_layout, ck_tile::tensor_layout::gemm::ColumnMajor>)
            ? std::vector<int>{N, K}
            : std::vector<int>{K, N};
    std::vector<int> c_dimensions =
        (std::is_same_v<matrix_c_layout, ck_tile::tensor_layout::gemm::RowMajor>)
            ? std::vector<int>{M, N}
            : std::vector<int>{N, M};

    ck_tile::HostTensor<ADataType> a_host(a_dimensions);
    ck_tile::HostTensor<BDataType> b_host(b_dimensions);

    ck_tile::HostTensor<CDataType> c_host_ref(c_dimensions);
    ck_tile::HostTensor<CDataType> c_host_dev(c_dimensions);

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_host);

    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_buf(c_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.data());
    b_buf.ToDevice(b_host.data());

    invoke_gemm<ck_tile::half_t, matrix_a_layout, matrix_b_layout, matrix_c_layout>(
        a_buf, b_buf, c_buf, arg_parser);

    c_buf.FromDevice(c_host_dev.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        // ToDo: Will Add the Element Op (bias) verification in the future.
        ck_tile::reference_gemm<ADataType,
                                BDataType,
                                AccDataType,
                                CDataType,
                                matrix_a_layout,
                                matrix_b_layout,
                                matrix_c_layout>(a_host, b_host, c_host_ref);

        pass = ck_tile::check_err(c_host_dev, c_host_ref);

        std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::flush;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        ck_tile::index_t stride_a = arg_parser.get_int("stride_a");
        ck_tile::index_t stride_b = arg_parser.get_int("stride_b");
        ck_tile::index_t stride_c = arg_parser.get_int("stride_c");

        if(stride_a == 0)
        {
            if constexpr(std::is_same_v<matrix_a_layout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                stride_a = K;
            }
            else
            {
                stride_a = M;
            }
        }

        if(stride_b == 0)
        {
            if constexpr(std::is_same_v<matrix_b_layout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                stride_b = N;
            }
            else
            {
                stride_b = K;
            }
        }

        if(stride_c == 0)
        {
            if constexpr(std::is_same_v<matrix_c_layout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                stride_c = N;
            }
            else
            {
                stride_c = M;
            }
        }

        ck_tile::HostTensor<CDataType> c_host_gpu_ref(c_dimensions);
        ck_tile::DeviceMem c_gpu_buf(c_host_gpu_ref.get_element_space_size_in_bytes());

        ck_tile::reference_gemm_gpu<ADataType, BDataType, AccDataType, CDataType>(
            a_buf, b_buf, c_gpu_buf, M, N, K, stride_a, stride_b, stride_c);

        c_buf.FromDevice(c_host_gpu_ref.data());

        pass = ck_tile::check_err(c_host_dev, c_host_gpu_ref);

        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::flush;
    }

    std::cout << std::endl << std::flush;

    return pass;
}
