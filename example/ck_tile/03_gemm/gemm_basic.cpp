
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "gemm_basic.hpp"

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
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "10", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename PipelineProblem,
          typename GemmPipeline,
          typename GemmShape>
float gemm_calc(const gemm_basic_args& args, const ck_tile::stream_config& s)
{
    // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadA = true;
    constexpr bool kPadB = true;

    constexpr int kBlockPerCu = 1;

    using TilePartitioner = ck_tile::GemmTilePartitioner<GemmShape>;
    using GemmEpilogue    = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadA, kPadB>>;
    // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
    // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
    using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

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

template <typename DataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC,
          typename PipelineProblem,
          typename GemmPipeline,
          typename GemmShape>
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

    ck_tile::index_t batch_size = arg_parser.get_int("b");
    ck_tile::index_t M          = arg_parser.get_int("m");
    ck_tile::index_t N          = arg_parser.get_int("n");
    ck_tile::index_t K          = arg_parser.get_int("k");

    ck_tile::index_t stride_a = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_b = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_c = arg_parser.get_int("stride_c");

    gemm_basic_args args;
    args.p_a    = a_buf.GetDeviceBuffer();
    args.p_b    = b_buf.GetDeviceBuffer();
    args.p_c    = c_buf.GetDeviceBuffer();
    args.kbatch = batch_size;
    args.M      = M;
    args.N      = N;
    args.K      = K;

    auto f_get_default_stride = [](std::size_t row,
                                   std::size_t col,
                                   std::size_t stride,
                                   auto layout) {
        if(stride == 0)
        {
            // give a chance if stride is zero, return a default packed stride
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
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

    args.stride_A = f_get_default_stride(M, K, stride_a, LayoutA{});
    args.stride_B = f_get_default_stride(K, N, stride_b, LayoutB{});
    args.stride_C = f_get_default_stride(M, N, stride_c, LayoutC{});

    float ave_time = gemm_calc<LayoutA, LayoutB, LayoutC, PipelineProblem, GemmPipeline, GemmShape>(
        args, ck_tile::stream_config{nullptr, true});
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "The overall perfomance of the GEMM with "
              << "[" << data_type << "]"
              << "batch size: " << batch_size << ". m:" << M << ", n:" << N << ", k:" << K
              << " is: \n";
    std::cout << "Running time: " << ave_time << "ms, Throughput " << gb_per_sec << "GB/s \n"
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

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using namespace ck_tile::literals;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride = [](std::size_t row,
                                   std::size_t col,
                                   std::size_t stride,
                                   auto layout) {
        if(stride == 0)
        {
            // give a chance if stride is zero, return a default packed stride
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
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

    ck_tile::HostTensor<ADataType> a_host(f_host_tensor_descriptor(M, K, stride_A, ALayout{}));
    ck_tile::HostTensor<BDataType> b_host(f_host_tensor_descriptor(K, N, stride_B, BLayout{}));

    ck_tile::HostTensor<CDataType> c_host_ref(f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
    ck_tile::HostTensor<CDataType> c_host_dev(f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_host);

    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_buf(c_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.data());
    b_buf.ToDevice(b_host.data());

    // The kPadA, kPadB, kPadC & kBlockPerCu should also come from the Codegen part.
    constexpr bool kPadA = true;
    constexpr bool kPadB = true;
    constexpr bool kPadC = true;

    // This part comes from the Codegen
    constexpr ck_tile::index_t M_Tile = 128;
    constexpr ck_tile::index_t N_Tile = 128;
    constexpr ck_tile::index_t K_Tile = 32;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 2;
    constexpr ck_tile::index_t K_Warp = 1;

    constexpr ck_tile::index_t M_Warp_Tile = 32;
    constexpr ck_tile::index_t N_Warp_Tile = 32;
    constexpr ck_tile::index_t K_Warp_Tile = 8;

    using CodegenGemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                               ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                               ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using CodegenPipelineProblem = ck_tile::BlockGemmPipelineProblem<ADataType,
                                                                     BDataType,
                                                                     AccDataType,
                                                                     CodegenGemmShape,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     kPadA,
                                                                     kPadB,
                                                                     kPadC>;

    using CodegenGemmPipeline = ck_tile::BlockGemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

    invoke_gemm<ck_tile::half_t,
                ALayout,
                BLayout,
                CLayout,
                CodegenPipelineProblem,
                CodegenGemmPipeline,
                CodegenGemmShape>(a_buf, b_buf, c_buf, arg_parser);

    c_buf.FromDevice(c_host_dev.data());

    bool pass_cpu = true;

    if(arg_parser.get_int("v") == 1)
    {
        // ToDo: Will Add the Element Op (bias) verification in the future.
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_host, b_host, c_host_ref);

        pass_cpu = ck_tile::check_err(c_host_dev, c_host_ref);

        std::cout << "The CPU verification result is:" << (pass_cpu ? "correct" : "fail")
                  << std::flush;
    }

    bool pass_gpu = true;

    if(arg_parser.get_int("v") == 2)
    {
        ck_tile::HostTensor<CDataType> c_host_gpu_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        ck_tile::DeviceMem c_gpu_buf(c_host_gpu_ref.get_element_space_size_in_bytes());
        c_gpu_buf.SetZero();

        ck_tile::reference_gemm_gpu<ADataType, BDataType, AccDataType, CDataType>(
            a_buf, b_buf, c_gpu_buf, M, N, K, stride_A, stride_B, stride_C);

        c_gpu_buf.FromDevice(c_host_gpu_ref.data());

        pass_gpu = ck_tile::check_err(c_host_dev, c_host_gpu_ref);

        std::cout << "The GPU verification result is: " << (pass_gpu ? "correct" : "fail")
                  << std::flush;
    }

    std::cout << std::endl << std::flush;

    return !pass_gpu;
}
