#include "ck_tile/host.hpp"
#include "reference_add.hpp"
#include "add.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "10240", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;

    ck_tile::index_t m = arg_parser.get_int("m");
    ck_tile::index_t n = arg_parser.get_int("n");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    ck_tile::HostTensor<XDataType> x_host_a({m, n});
    ck_tile::HostTensor<XDataType> x_host_b({m, n});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n});

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_b);

    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());

    // 27xx -> 32xx, 1D block to 2D block
    using BlockWarps = ck_tile::sequence<2, 2>;
    using BlockTile  = ck_tile::sequence<2, 1024>;
    using WarpTile   = ck_tile::sequence<1, 512>;
    using Vector     = ck_tile::sequence<1, 4>;
    
    constexpr ck_tile::index_t kBlockSize  = 256;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    ck_tile::index_t kGridSize             = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape = ck_tile::AddShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Porblem =
        ck_tile::AddProblem<XDataType, ComputeDataType, YDataType, Shape>;

    using Kernel = ck_tile::Add<Porblem>;

    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                       static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m,
                                       n));

    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_add<XDataType, YDataType>(
           x_host_a, x_host_b, y_host_ref);
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

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
    }
}
