// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/reference/reference_elementwise.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "10", "cold iter")
        .insert("repeat", "50", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M      = arg_parser.get_int("m");
    ck_tile::index_t N      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = N;
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    assert(stride >= N);

    using XDataType             = DataType;
    using YDataType             = DataType;
    using ComputeDataType       = float;
    using XElementwiseOperation = ck_tile::element_wise::UnarySquare;

    // 1. Initialize the input data on the host
    ck_tile::HostTensor<XDataType> x_host_a({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_validation({M, N}, {stride, 1});

    std::vector<ck_tile::index_t> shape = {M, N};

    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);

    // 2. Create device memory buffers and copy input data from host to device
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());
    x_buf_a.ToDevice(x_host_a.data());

    // 3. Create the kernel

    // Dividing the problem into blocktile, warptile, and vector
    using BlockTile = ck_tile::sequence<2048>; // Size of the block tile (Entire problem is divided
                                               // into blocks of this size)
    using BlockWarps = ck_tile::sequence<8>; // How many concurrent warps are in a block (Each warp
                                             // will cover some part of blockTile)
    using WarpTile = ck_tile::sequence<64>;  // How many elements are covered by a warp

    using Shape   = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, ComputeDataType>;
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        XDataType, // ComputeDataType is same as
                                                                   // XDataType in the unary case
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    // Compute flattened size
    ck_tile::index_t total_elements = 1;
    for(auto d : shape)
        total_elements *= d;

    constexpr ck_tile::index_t kBlockSize =
        ck_tile::get_warp_size() * BlockWarps::at(ck_tile::number<0>{});
    constexpr ck_tile::index_t kBlockPerCu = 1;

    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

    std::cout << "grid size = " << kGridSize << std::endl;
    std::cout << "Total elements = " << total_elements << std::endl;

    auto input_tensors = ck_tile::make_tuple(static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()));
    auto input_size    = ck_tile::make_tuple(M, N);

    // Check if the kernel configuration is supported
    if(!Kernel::IsSupportedArgument(input_size))
    {
        throw std::runtime_error(
            "The kernel configuration is not supported for the given input size.");
    }

    // 4. Run the kernel
    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       input_size,
                                       ck_tile::make_tuple(N, 1), // Input Stride
                                       ck_tile::make_tuple(N, 1), // Output Stride
                                       input_tensors,
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer())));

    std::cout << "Average time: " << ave_time << " ms" << std::endl;

    // 5. Verify the output
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data());

        auto op = [](const auto& v0) { return v0 * v0; };

        ck_tile::reference_unary_elementwise<XDataType, YDataType, YDataType>(x_host_a, y_host, op);

        pass = ck_tile::check_err(
            y_validation, y_host, "Elementwise Add Error: Incorrect results!", 0.01, 0.01);
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
