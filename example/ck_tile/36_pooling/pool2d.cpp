// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/pool.hpp"
#include "ck_tile/host/reference/reference_pool.hpp"
#include <cstring>

// Parse command-line arguments for 2D pooling example
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("N", "128", "N dimension")
        .insert("H", "71", "H dimension")
        .insert("W", "71", "W dimension")
        .insert("C", "192", "C dimension")
        .insert("X", "3", "X dimension")
        .insert("Y", "3", "Y dimension")
        .insert("Sy", "2", "window stride h")
        .insert("Sx", "2", "window stride w")
        .insert("Dy", "1", "window dilation h")
        .insert("Dx", "1", "window dilation w")
        .insert("LeftPy", "1", "left padding h")
        .insert("LeftPx", "1", "left padding w")
        .insert("RightPy", "1", "right padding h")
        .insert("RightPx", "1", "right padding w")
        .insert("v", "1", "cpu validation or not")
        .insert("warmup", "0", "cold iter")
        .insert("repeat", "1", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InDataType, typename OutDataType, typename ComputeDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{

    const ck_tile::index_t N = arg_parser.get_int("N");
    const ck_tile::index_t H = arg_parser.get_int("H");
    const ck_tile::index_t W = arg_parser.get_int("W");
    const ck_tile::index_t C = arg_parser.get_int("C");

    const ck_tile::index_t Y = arg_parser.get_int("Y");
    const ck_tile::index_t X = arg_parser.get_int("X");

    const ck_tile::index_t Sy = arg_parser.get_int("Sy");
    const ck_tile::index_t Sx = arg_parser.get_int("Sx");

    const ck_tile::index_t Dy = arg_parser.get_int("Dy");
    const ck_tile::index_t Dx = arg_parser.get_int("Dx");

    const ck_tile::index_t LeftPy  = arg_parser.get_int("LeftPy");
    const ck_tile::index_t LeftPx  = arg_parser.get_int("LeftPx");
    const ck_tile::index_t RightPy = arg_parser.get_int("RightPy");
    const ck_tile::index_t RightPx = arg_parser.get_int("RightPx");

    // Calculate effective window size with dilation
    const ck_tile::index_t Ys = (Y - 1) * Dy + 1;
    const ck_tile::index_t Xs = (X - 1) * Dx + 1;

    // Calculate output spatial dimensions
    const ck_tile::index_t Ho = (H + LeftPy + RightPy - Ys) / Sy + 1;
    const ck_tile::index_t Wo = (W + LeftPx + RightPx - Xs) / Sx + 1;

    printf("Input parameters:\n");
    printf("N: %d, H: %d, W: %d, C: %d\n", N, H, W, C);
    printf("Window Y: %d, X: %d, Stride Y: %d, X: %d\n", Y, X, Sy, Sx);
    printf("Output Ho: %d, Wo: %d\n", Ho, Wo);

    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    // Shapes / strides / parameters (NHWC)
    const auto input_shape            = ck_tile::make_tuple(N, H, W, C);
    const auto output_shape           = ck_tile::make_tuple(N, Ho, Wo, C);
    const auto input_strides          = ck_tile::make_tuple(H * W * C, W * C, C, 1);
    const auto output_strides         = ck_tile::make_tuple(Ho * Wo * C, Wo * C, C, 1);
    const auto window_spatial_lengths = ck_tile::make_tuple(Y, X);
    const auto window_strides         = ck_tile::make_tuple(Sy, Sx);
    const auto window_dilations       = ck_tile::make_tuple(Dy, Dx);
    const auto input_left_pads        = ck_tile::make_tuple(LeftPy, LeftPx);
    const auto input_right_pads       = ck_tile::make_tuple(RightPy, RightPx);

    ck_tile::HostTensor<InDataType> in({N, H, W, C}, {H * W * C, W * C, C, 1});
    ck_tile::HostTensor<OutDataType> out({N, Ho, Wo, C}, {Ho * Wo * C, Wo * C, C, 1});
    ck_tile::HostTensor<OutDataType> out_ref({N, Ho, Wo, C}, {Ho * Wo * C, Wo * C, C, 1});

    ck_tile::FillUniformDistribution<InDataType>{-5.f, 5.f}(in);

    ck_tile::DeviceMem in_buf(in.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_buf(out.get_element_space_size_in_bytes());

    in_buf.ToDevice(in.data());

    using ReduceOp   = ck_tile::ReduceOp::Max;
    using BlockWarps = ck_tile::sequence<4, 1>;
    using BlockTile  = ck_tile::sequence<128, 128>;
    using WarpTile   = ck_tile::sequence<32, 128>;
    using ThreadTile     = ck_tile::sequence<8, 8>;

    using Shape   = ck_tile::PoolShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;
    using Problem = ck_tile::PoolProblem<InDataType,
                                         OutDataType,
                                         ComputeDataType,
                                         OutDataType,
                                         ReduceOp,
                                         false,
                                         false,
                                         Shape>;
    using Kernel  = ck_tile::Pool<Problem>;

    constexpr ck_tile::index_t kBlockPerCu = 1;
    constexpr ck_tile::index_t kBlockSize  = Kernel::kBlockSize;

    // M dimension = merged (N,Ho,Wo,C)
    const ck_tile::index_t M = N * Ho * Wo * C;
    const ck_tile::index_t kGridSize =
        (M + BlockTile::at(ck_tile::number<0>{}) - 1) / BlockTile::at(ck_tile::number<0>{});
    std::cout << "grid size " << kGridSize << std::endl;

    // Package arguments for kernel execution
    auto host_args = ck_tile::PoolHostArgs<decltype(input_shape), decltype(window_spatial_lengths)>{
        static_cast<InDataType*>(in_buf.GetDeviceBuffer()),
        static_cast<OutDataType*>(out_buf.GetDeviceBuffer()),
        input_shape,
        output_shape,
        input_strides,
        output_strides,
        window_spatial_lengths,
        window_strides,
        window_dilations,
        input_left_pads,
        input_right_pads};

    auto kernel_args = Kernel::MakeKernelArgs(host_args);

    // Validate kernel can handle the given configuration
    if(!Kernel::IsSupportedArgument(kernel_args))
    {
        std::cout << "ERROR: Kernel arguments are not supported!" << std::endl;
        return false;
    }
    std::cout << "Kernel argument validation: PASSED" << std::endl;

    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{}, kGridSize, kBlockSize, 0, kernel_args));

    // Calculate bandwidth: input + output data volume
    std::size_t num_btype = sizeof(InDataType) * N * C * H * W + sizeof(OutDataType) * M;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // Compare against CPU reference implementation
        ck_tile::reference_pool2d<InDataType, ComputeDataType, OutDataType>(in,
                                                                            out_ref,
                                                                            input_shape,
                                                                            output_shape,
                                                                            input_strides,
                                                                            output_strides,
                                                                            window_spatial_lengths,
                                                                            window_strides,
                                                                            window_dilations,
                                                                            input_left_pads,
                                                                            input_right_pads,
                                                                            ReduceOp{});
        out_buf.FromDevice(out.mData.data());
        pass = ck_tile::check_err(out, out_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    return run<ck_tile::half_t, ck_tile::half_t, float>(arg_parser) ? 0 : -2;
}
