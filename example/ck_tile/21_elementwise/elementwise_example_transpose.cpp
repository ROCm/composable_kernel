// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/reference/reference_transpose.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "m dimension of input")
        .insert("n", "1024", "n dimension of input")
        .insert("stride_in", "-1", "stride for input M dim, if -1 then equal to n")
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
    ck_tile::index_t M         = arg_parser.get_int("m");
    ck_tile::index_t N         = arg_parser.get_int("n");
    ck_tile::index_t stride_in = arg_parser.get_int("stride_in");

    if(stride_in < 0)
        stride_in = N; // Dense input: stride for M dim is N
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    if(stride_in < N)
    {
        throw std::runtime_error("stride_in must be >= N");
    }

    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;
    // Use PassThrough operation for transposition (data is moved, not changed)
    using XElementwiseOperation = ck_tile::element_wise::PassThrough;

    // 1. Initialize the input data on the host (CPU).
    // Input x_host_a: M x N
    // Output y_host: N x M (transposed)
    ck_tile::HostTensor<XDataType> x_host_a({M, N}, {stride_in, 1});
    // Output tensor y_host will have dimensions N x M.
    // Assuming dense output, its stride for the N dimension will be M.
    ck_tile::index_t stride_out_dim0 = M;
    ck_tile::HostTensor<YDataType> y_host({N, M}, {stride_out_dim0, 1});
    ck_tile::HostTensor<YDataType> y_validation({N, M}, {stride_out_dim0, 1});

    // The logical shape for the element-wise operation kernel is based on the input tensor's
    // elements.
    std::vector<ck_tile::index_t> op_shape_vec = {M, N};
    auto op_lengths                            = ck_tile::make_tuple(M, N); // Lens for the kernel

    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);

    // 2. Create device memory buffers
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes()); // y_host is N x M

    x_buf_a.ToDevice(x_host_a.data());

    // 3. Configure the kernel execution parameters.
    using BlockTile  = ck_tile::sequence<1024>;
    using BlockWarps = ck_tile::sequence<8>;
    using WarpTile   = ck_tile::sequence<64>;

    using Shape = ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, ComputeDataType>;

    // Problem definition for a single input tensor
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    ck_tile::index_t total_elements = M * N;

    constexpr ck_tile::index_t kBlockSize         = 64 * BlockWarps::at(ck_tile::number<0>{});
    constexpr ck_tile::index_t kBlockPerCu        = 1;
    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

    std::cout << "Input M=" << M << ", N=" << N << ", StrideIn=" << stride_in << std::endl;
    std::cout << "Output N=" << N << ", M=" << M << ", StrideOut=" << stride_out_dim0 << std::endl;
    std::cout << "Grid size = " << kGridSize << ", BlockSize = " << kBlockSize << std::endl;
    std::cout << "Total elements = " << total_elements << std::endl;

    // Input tensors tuple (single input)
    auto input_tensors = ck_tile::make_tuple(static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()));
    // Input strides tuple (tuple of tuples, one for each input)
    auto input_strides = ck_tile::make_tuple(stride_in, 1);
    // Output strides (for N x M tensor, dense)
    auto output_strides = ck_tile::make_tuple(1, stride_out_dim0);

    // Check if the kernel configuration is supported
    if(!Kernel::IsSupportedArgument(op_lengths))
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
                                       0,             // Shared memory
                                       op_lengths,    // Logical dimensions for the operation (M, N)
                                       input_strides, // Strides for input tensor(s)
                                       output_strides, // Strides for output tensor (N, M)
                                       input_tensors,
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer())));

    std::cout << "Average time: " << ave_time << " ms" << std::endl;

    // 5. Verify the output
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data()); // Copy result from device to y_validation
        ck_tile::reference_transpose_elementwise<XDataType, YDataType>(
            x_host_a, y_host); // Compute reference on host
        pass = ck_tile::check_err(
            y_validation, y_host, "Transpose Error: Incorrect results!", 0.01, 0.01);
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

    std::cerr << "Unsupported data type: " << data_type << std::endl;
    return -3;
}
