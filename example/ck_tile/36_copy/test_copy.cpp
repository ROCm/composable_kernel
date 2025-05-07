// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include <cstring>
#include "test_copy.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "64", "m dimension")
        .insert("n", "8", "n dimension")
        .insert("id", "0", "warp to use")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
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

    ck_tile::index_t m       = arg_parser.get_int("m");
    ck_tile::index_t n       = arg_parser.get_int("n");
    ck_tile::index_t warp_id = arg_parser.get_int("id");
    int do_validation        = arg_parser.get_int("v");
    int warmup               = arg_parser.get_int("warmup");
    int repeat               = arg_parser.get_int("repeat");

    ck_tile::HostTensor<XDataType> x_host({m, n});
    ck_tile::HostTensor<YDataType> y_host_ref({m, n});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n});

    // ck_tile::FillConstant<XDataType>{1.f}(x_host);
    ck_tile::half_t value = 1;
    for(int i = 0; i < m; i++)
    {
        value = 1;
        for(int j = 0; j < n; j++)
        {
            x_host(i, j) = value++;
        }
    }

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    using BlockWaves = ck_tile::sequence<2, 1>;
    using BlockTile  = ck_tile::sequence<64, 8>;
    using WaveTile   = ck_tile::sequence<64, 8>;
    using Vector     = ck_tile::sequence<1, 4>;

    ck_tile::index_t kGridSize = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape   = ck_tile::TileCopyShape<BlockWaves, BlockTile, WaveTile, Vector>;
    using Problem = ck_tile::TileCopyProblem<XDataType, Shape>;
    using Kernel  = ck_tile::TileCopy<Problem>;

    constexpr ck_tile::index_t kBlockSize  = 128;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    std::cout << "block size " << kBlockSize << std::endl;
    std::cout << "warp SIze " << ck_tile::get_warp_size() << std::endl;
    std::cout << "warps per block _M " << Shape::WarpPerBlock_M << " " << Shape::WarpPerBlock_N
              << std::endl;
    std::cout << "Block waves: " << BlockWaves::at(ck_tile::number<0>{}) << " "
              << BlockWaves::at(ck_tile::number<1>{}) << std::endl;
    std::cout << " Wave Groups: " << Shape::WaveGroups << std::endl;

    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m,
                                       n,
                                       warp_id));

    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m;

    float gb_per_sec = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // reference
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, x_host);

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
    return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
}
