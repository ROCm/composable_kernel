// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include <cstring>

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("n", "32", "n dimension")
        .insert("h", "19", "h dimension")
        .insert("w", "7", "w dimension")
        .insert("c", "512", "c dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        // .insert("warmup", "5", "cold iter")
        // .insert("repeat", "20", "hot iter")
        .insert("warmup", "0", "cold iter")
        .insert("repeat", "1", "hot iter")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "multi_reduce_multiblock.json", "json file name to dump results");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;

    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t H = arg_parser.get_int("h");
    ck_tile::index_t W = arg_parser.get_int("w");
    ck_tile::index_t C = arg_parser.get_int("c");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    std::vector<ck_tile::index_t> problem_shape = {N, H, W, C};
    std::vector<ck_tile::index_t> strides(4);
    strides[0] = H * W * C;
    strides[1] = W * C;
    strides[2] = C;
    strides[3] = 1;

    // Define reduction specification:
    constexpr auto kept_dim    = ck_tile::sequence<0, 3>{}; // Which dimension to keep
    constexpr auto reduce_dims = ck_tile::sequence<1, 2>{}; // Which dimensions to reduce

    ck_tile::HostTensor<XDataType> x_host(problem_shape, strides);
    ck_tile::HostTensor<YDataType> y_host_add_ref({N, C}, {C, 1});
    ck_tile::HostTensor<YDataType> y_host_max_ref({N, C}, {C, 1});
    auto y_host_ref_tuple = ck_tile::make_tuple(y_host_add_ref, y_host_max_ref);
    using YRefTuple       = decltype(y_host_ref_tuple);

    ck_tile::HostTensor<YDataType> y_host_add_dev({N, C}, {C, 1});
    ck_tile::HostTensor<YDataType> y_host_max_dev({N, C}, {C, 1});
    auto y_host_dev_tuple = ck_tile::make_tuple(y_host_add_dev, y_host_max_dev);

    const auto number_operations = y_host_dev_tuple.size();

    std::vector<YDataType> h(number_operations * N * C);
    std::fill(h.begin(), h.end(), static_cast<YDataType>(0.0f)); // TODO: Fill it in with the identify element

    auto y_buf_size = number_operations *
                      y_host_dev_tuple.at(ck_tile::number<0>{}).get_element_space_size_in_bytes();
    ck_tile::DeviceMem y_buf(y_buf_size);

    const auto output_tensor_offset = N * C;

    // This needs to be atomic operations as multiple blocks may write to the same output element
    // auto blockwise_acc_ops = ck_tile::tuple<decltype(atomicAdd), decltype(atomicAdd)>{}; // TODO: make this work
    // size_t cluster_size = 3; // TODO: calculate automatically
    // size_t cluster_size_m = 3;
    // size_t cluster_size_k = 3;

    // TODO:
    // [] Atomic ADD function or a tuple of it, for each operations
    // [X] Provide number of cluster (possible calculate it based on the other inputs sizes)

    // ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    ck_tile::FillUniformDistribution<XDataType>{1.f, 1.f}(x_host);
    // ck_tile::FillUniformDistribution<XDataType>{1.f, 2.f}(x_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    y_buf.ToDevice(h.data()); // Initialize the output buffer with operations' identity element

    using ReduceOps  = ck_tile::tuple<ck_tile::ReduceOp::Add, ck_tile::ReduceOp::Add>;
    using BlockWarps = ck_tile::sequence<4, 1>;
    using BlockTile  = ck_tile::sequence<128, 128>;
    using WarpTile   = ck_tile::sequence<32, 128>;
    using Vector     = ck_tile::sequence<8, 8>;

    constexpr ck_tile::index_t kBlockPerCu = 1;
    ck_tile::index_t kept_dim_len_prod     = N * C;

    using Shape = ck_tile::Reduce2dShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Problem =
        ck_tile::Reduce2dMultiBlockProblem<XDataType, ComputeDataType, YDataType, Shape, ReduceOps>;

    using Kernel                      = ck_tile::MultiReduceMultiblock<Problem>;

    // Determine block group size for multi-block reduction
    // TODO: it should be in a helper function somewhere else
    ck_tile::index_t reduce_total_length = H * W; // Hardcoded for now
    int K_BlockTileSize = BlockTile::at(1);
    int num_block_tile_iterations = std::max(1, static_cast<int>(std::ceil((reduce_total_length - 1.0) / (127.0 * K_BlockTileSize)))); // Ensure at most 128 blocks in a group
    int block_group_size = (reduce_total_length + (K_BlockTileSize * num_block_tile_iterations) - 1) /
                    (K_BlockTileSize * num_block_tile_iterations);

    const ck_tile::index_t kBlockSize = Kernel::BlockSize();
    ck_tile::index_t kGridSize = (kept_dim_len_prod + BlockTile::at(ck_tile::number<0>{}) - 1) /
                                 BlockTile::at(ck_tile::number<0>{})*block_group_size;
    std::cout << "Block group size: " << block_group_size << ", Num block tile iterations: " << num_block_tile_iterations << ", Reduce total length: " << reduce_total_length << std::endl;
    std::cout << "grid size " << kGridSize << ", block size " << kBlockSize << std::endl;

    // Create input tensor shape and strides
    auto input_shape =
        ck_tile::make_tuple(problem_shape[0], problem_shape[1], problem_shape[2], problem_shape[3]);
    auto input_strides = ck_tile::make_tuple(strides[0], strides[1], strides[2], strides[3]);

    if(!Kernel::IsSupportedArgument(
           C, input_strides)) // output tensor's continuous dimension and input strides
    {
        throw std::runtime_error("Wrong! Arguments not supported!\n");
    }


    // TODO: clearn the output buffer, use launch_kernel_time_mask see the splitk example
    // auto clear_gemm_output = [&]() {
    //             if(args.k_batch > 1)
    //                 hipGetErrorString(hipMemsetAsync(
    //                     ws_args.c_ptr, 0, args.M * args.N * sizeof(WorkspaceType), s.stream_id_));
    // };

    auto noop = [](auto& element) { element = 2*element;}; // TODO: check for the passthrough function

    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{},
                                          kGridSize,
                                          kBlockSize,
                                          0,
                                          static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                          static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                          input_shape,
                                          input_strides,
                                          kept_dim,
                                          reduce_dims,
                                          output_tensor_offset,
                                          block_group_size,
                                          num_block_tile_iterations,
                                          noop
                                        ));
                                        //   blockwise_acc_ops));

    std::size_t num_btype = sizeof(XDataType) * N * C * H * W + sizeof(YDataType) * N * C;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_multiple_reduce<XDataType, ComputeDataType, YDataType>(
            x_host, y_host_ref_tuple, ReduceOps{}, kept_dim, reduce_dims, noop);
        std::cout << "Read " << y_buf_size / 10 << " Bytes from the device" << std::endl;

        // Transfer data from device and check error for each operation
        y_buf.FromDevice(h.data());
        ck_tile::static_for<0, number_operations, 1>{}([&](auto i) {
            std::memcpy(y_host_dev_tuple.get(ck_tile::number<i>{}).data(),
                        h.data() + i * output_tensor_offset,
                        output_tensor_offset * sizeof(YDataType));
                        
            // std::cout << y_host_dev_tuple.get(ck_tile::number<i>{}) << std::endl;
            pass &= ck_tile::check_err(y_host_dev_tuple.get(ck_tile::number<i>{}),
                                       y_host_ref_tuple.get(ck_tile::number<i>{}));
        });

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
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
    } /*else {
        return run<float>(arg_parser) ? 0 : -2;
    }*/
}
