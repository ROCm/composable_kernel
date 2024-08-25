
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_basic.hpp"
#include "ck_tile/host.hpp"

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

/*
create_args is a function
*/
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("b", "1", "batch size")
        .insert("m", "1024", "m dimension")
        .insert("n", "2048", "n dimension")
        .insert("k", "32", "k dimension")
        .insert("stride_a", "0", "stride on apply the m,k A block")
        .insert("stride_b", "0", "stride on apply the n,k B block")
        .insert("stride_c", "0", "stride on apply the m,n C block")
        .insert("grouped", "0", "bool condition on whether it is a grouped gemm")
        .insert(
            "grouped_dimension_m", "0", "Fill in the desired dimension when enable grouped gemm")
        .insert(
            "grouped_dimension_n", "0", "Fill in the desired dimension when enable grouped gemm")
        .insert(
            "grouped_dimension_k", "0", "Fill in the desired dimension when enable grouped gemm")
        .insert("v", "1", "cpu validation or not")
        .insert("e", "1e-5", "epsilon")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("following_op", "no", "combined_op. bias/relu/gelu...")
        .insert("warmup", "10", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename Layouts>
float gemm_calc(gemm_basic_args& args, const ck_tile::stream_config& s)
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
    constexpr bool kPadC = false;

    constexpr ck_tile::index_t kBlockPerCu = 1;

    // ===============================================

    using Shape = ck_tile::TileGemmShapeNewGemm<
        ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
        ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
        ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmTilePartitioner<Shape>;
    using PipelineProblem = ck_tile::
        BlockGemmPipelineProblem<XDataType, YDataType, AccDataType, Shape, kPadA, kPadB, kPadC>;
    // The GemmPipeline should also come from the Codegen.
    using GemmPipeline = ck_tile::BlockGemmPipelineAGmemBGmemCRegV1<PipelineProblem>;
    using GemmEpilogue = ck_tile::Default2DEpilogue<
        ck_tile::Default2DEpilogueProblem<AccDataType, ODataType, kPadA, kPadB>>;
    // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
    // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
    using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, Layouts>;

    auto kargs = Kernel::MakeKargs(args.p_x,
                                   args.p_y,
                                   args.p_z,
                                   args.batch_size,
                                   args.epsilon,
                                   args.M,
                                   args.N,
                                   args.K,
                                   args.stride_A,
                                   args.stride_B,
                                   args.stride_C);

    const dim3 grids      = Kernel::GridSize(args.M, args.N, args.batch_size);
    constexpr dim3 blocks = Kernel::BlockSize();

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    return ave_time;
}

template <typename DataType, typename Layouts>
float OperatorExecution(ck_tile::DeviceMem& x_buf,
                        ck_tile::DeviceMem& y_buf,
                        ck_tile::DeviceMem& z_buf,
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
    args.p_x        = x_buf.GetDeviceBuffer();
    args.p_y        = y_buf.GetDeviceBuffer();
    args.p_z        = z_buf.GetDeviceBuffer();
    args.epsilon    = epsilon;
    args.batch_size = batch_size;
    args.M          = M;
    args.N          = N;
    args.K          = K;

    // Only set stride_M and stride_N if they are non-zero and not equal to K.
    if(stride_a != 0)
    {
        args.stride_A = stride_a;
    }
    else
    {
        args.stride_A = [&]() {
            if constexpr(Layouts::LayoutA == ck_tile::MatrixALayout::KM)
            {
                return M;
            }
            else
            {
                return K;
            }
        }();
    }

    if(stride_b != 0)
    {
        args.stride_B = stride_b;
    }
    else
    {
        args.stride_B = [&]() {
            if constexpr(Layouts::LayoutB == ck_tile::MatrixBLayout::KN)
            {
                return N;
            }
            else
            {
                return K;
            }
        }();
    }

    if(stride_c != 0)
    {
        args.stride_C = stride_c;
    }
    else
    {
        args.stride_C = [&]() {
            if constexpr(Layouts::LayoutC == ck_tile::MatrixCLayout::NM)
            {
                return M;
            }
            else
            {
                return N;
            }
        }();
    }

    float ave_time = gemm_calc<Layouts>(args, ck_tile::stream_config{nullptr, true});
    std::size_t num_byte =
        sizeof(XDataType) * M * K + sizeof(YDataType) * N * K + sizeof(ODataType) * M * N;
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

    bool grouped_enable             = arg_parser.get_bool("grouped");
    std::string following_op_descrp = arg_parser.get_str("following_op");
    ck_tile::index_t M              = arg_parser.get_int("m");
    ck_tile::index_t N              = arg_parser.get_int("n");
    ck_tile::index_t K              = arg_parser.get_int("k");

    constexpr ck_tile::MatrixALayout matrix_a_layout = ck_tile::MatrixALayout::MK;
    constexpr ck_tile::MatrixBLayout matrix_b_layout = ck_tile::MatrixBLayout::NK;
    constexpr ck_tile::MatrixCLayout matrix_c_layout = ck_tile::MatrixCLayout::MN;

    using Layouts = LayoutConfig<matrix_a_layout, matrix_b_layout, matrix_c_layout>;
    // host verify
    std::vector<int> x_dimensions = (matrix_a_layout == ck_tile::MatrixALayout::MK)
                                        ? std::vector<int>{M, K}
                                        : std::vector<int>{K, M};
    std::vector<int> y_dimensions = (matrix_b_layout == ck_tile::MatrixBLayout::NK)
                                        ? std::vector<int>{N, K}
                                        : std::vector<int>{K, N};
    std::vector<int> z_dimensions = (matrix_c_layout == ck_tile::MatrixCLayout::MN)
                                        ? std::vector<int>{M, N}
                                        : std::vector<int>{N, M};

    ck_tile::HostTensor<XDataType> x_host(x_dimensions);
    ck_tile::HostTensor<YDataType> y_host(y_dimensions);

    ck_tile::HostTensor<ODataType> z_host_ref(z_dimensions);
    ck_tile::HostTensor<ODataType> z_host_dev(z_dimensions);

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    ck_tile::FillUniformDistribution<YDataType>{-5.f, 5.f}(y_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem z_buf(z_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    y_buf.ToDevice(y_host.data());

    if(grouped_enable || following_op_descrp != "no")
    {
        std::cerr << "Other category of the GEMM is unsupported for now!" << std::endl;
        return -1;
    }

    OperatorExecution<ck_tile::half_t, Layouts>(x_buf, y_buf, z_buf, arg_parser);

    bool pass = true;

    if(arg_parser.get_bool("v"))
    {
        // ToDo: Will Add the Element Op (bias) verification in the future.
        ck_tile::reference_gemm<XDataType, YDataType, AccDataType, ODataType>(
            x_host, y_host, z_host_ref, matrix_a_layout);

        z_buf.FromDevice(z_host_dev.data());

        pass = ck_tile::check_err(z_host_dev, z_host_ref);

        std::cout << "The veification result is:" << (pass ? "correct" : "fail") << std::flush;
    }

    std::cout << std::endl << std::flush;

    return !pass;
}
