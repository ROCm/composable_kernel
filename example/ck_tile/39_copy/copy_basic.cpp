// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include <cstring>
#include "copy_basic.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "128", "m dimension")
        .insert("n", "8", "n dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision(fp16 or fp32)")
        .insert("warmup", "50", "cold iter")
        .insert("repeat", "100", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType = DataType;
    using YDataType = DataType;

    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    // Create host tensors
    ck_tile::HostTensor<XDataType> x_host({m, n});     // input matrix
    ck_tile::HostTensor<YDataType> y_host_ref({m, n}); // reference output matrix
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}); // device output matrix

    // Initialize input data with increasing values
    ck_tile::half_t value = 1;
    for(int i = 0; i < m; i++)
    {
        value = 1;
        for(int j = 0; j < n; j++)
        {
            x_host(i, j) = value++;
        }
    }

    // Allocate device memory
    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    // Define tile configuration
    using Vector     = ck_tile::sequence<1, 4>;   // vector size along M and N dimension
    using WaveTile   = ck_tile::sequence<32, 8>;  // warp size along M and N dimension
    using BlockWaves = ck_tile::sequence<4, 1>;   // number of warps along M dimension
    using BlockTile  = ck_tile::sequence<512, 8>; // block size along M and N dimension

    // Calculate grid size
    ck_tile::index_t kGridSize = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "grid size (number of blocks per grid) " << kGridSize << std::endl;

    // Define kernel types
    using Shape   = ck_tile::TileCopyShape<BlockWaves, BlockTile, WaveTile, Vector>;
    using Problem = ck_tile::TileCopyProblem<XDataType, Shape>;
    using Policy  = ck_tile::TileCopyPolicy<Problem>;
    using Kernel  = ck_tile::TileCopyKernel<Problem, Policy>;
    // using LDSKernel = ck_tile::TileCopyKernel_LDS<Problem, Policy>;

    constexpr ck_tile::index_t kBlockSize = Shape::BlockSize;

    // Print configuration information
    std::cout << "block size (number of threads per block) " << kBlockSize << std::endl;
    std::cout << "warp size (number of threads per warp) " << ck_tile::get_warp_size() << std::endl;
    std::cout << "block waves (number of warps per block) " << BlockWaves::at(ck_tile::number<0>{})
              << " " << BlockWaves::at(ck_tile::number<1>{}) << std::endl;
    std::cout << "block tile (number of elements per block) " << BlockTile::at(ck_tile::number<0>{})
              << " " << BlockTile::at(ck_tile::number<1>{}) << std::endl;
    std::cout << "wave tile (number of elements per wave) " << WaveTile::at(ck_tile::number<0>{})
              << " " << WaveTile::at(ck_tile::number<1>{}) << std::endl;
    std::cout << "vector (number of elements per thread) " << Vector::at(ck_tile::number<0>{})
              << " " << Vector::at(ck_tile::number<1>{}) << std::endl;
    std::cout << "WarpRepetitionPerBlock_M =  " << Shape::WarpRepetitionPerBlock_M << " --> ("
              << Shape::Block_Tile_M << "/" << Shape::Warps_Per_Block_M << "*" << Shape::Warp_Tile_M
              << ")" << std::endl;
    std::cout << "WarpRepetitionPerBlock_N =  " << Shape::WarpRepetitionPerBlock_N << " --> ("
              << Shape::Block_Tile_N << "/" << Shape::Warps_Per_Block_N << "*" << Shape::Warp_Tile_N
              << ")" << std::endl;

    // Launch kernel
    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, warmup, repeat, 1},
        ck_tile::make_kernel<kBlockSize, 1>(Kernel{},
                                            kGridSize,
                                            kBlockSize,
                                            0,
                                            static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                            static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                            m,
                                            n));

    // Calculate and print performance metrics
    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m;
    float gb_per_sec      = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // Copy results back to host
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, x_host);
        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    // // Print results for debugging
    // std::cout << "Input matrix (x_host): " << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     for(int j = 0; j < n; j++)
    //     {
    //         std::cout << static_cast<float>(x_host(i, j)) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Output matrix (y_host_dev): " << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     for(int j = 0; j < n; j++)
    //     {
    //         std::cout << static_cast<float>(y_host_dev(i, j)) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    if(arg_parser.get_str("prec") == "fp16")
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    else
        return run<float>(arg_parser) ? 0 : -2;
}
