#include "ck_tile/host.hpp"
#include "reference_vector_add.hpp" 
#include "vector_add.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256000000", "m dimension")
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
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    ck_tile::HostTensor<XDataType> x_host_a({m});
    ck_tile::HostTensor<XDataType> x_host_b({m});

    ck_tile::HostTensor<YDataType> y_host_ref({m});
    ck_tile::HostTensor<YDataType> y_host_dev({m});

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_b);

    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());

    // using BlockTile = ck_tile::sequence<8192>;
    // using BlockWarps = ck_tile::sequence<4>;
    // using WarpTile = ck_tile::sequence<512>; // 8 * 64 = 512
    // using Vector  = ck_tile::sequence<8>; // 8 * 16 = 128 bytes


    // constexpr ck_tile::index_t kBlockSize  = 256;
    // constexpr ck_tile::index_t kBlockPerCu = 1;


    using BlockTile  = ck_tile::sequence<8192>;
    using BlockWarps = ck_tile::sequence<8>;
    using WarpTile   = ck_tile::sequence<64>;
    using Vector    = ck_tile::sequence<1>; 
    
    constexpr ck_tile::index_t kBlockSize  = 512;
    constexpr ck_tile::index_t kBlockPerCu = 1;


    ck_tile::index_t kGridSize             = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "grid size " << kGridSize << std::endl;

    

    using Shape = ck_tile::MultiplyVector<BlockWarps, BlockTile, WarpTile, Vector>;
    std::cout << "Problem Shape:: M = " << m << std::endl;
    std::cout << "BlockTile: " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "Number of Blocks in Grid: " << m / BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "BlockWarps: " << BlockWarps::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "WarpTile: " << WarpTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "Vector: " << Vector::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "Repeat: " << Shape::Repeat_M << std::endl;
    std::cout << "Threads per Block: " << kBlockSize << std::endl;
    std::cout << "ThreadBlocks per CU: " << kBlockPerCu << std::endl;
    using Problem =
        ck_tile::MultiplyVectorProblem<XDataType, ComputeDataType, YDataType, Shape>;

    using Kernel = ck_tile::MultiplyVectorKernel<Problem>;

    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                       static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m));

    std::size_t num_btype = sizeof(XDataType) * m + sizeof(YDataType) * m;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_vector_add<XDataType, YDataType>(
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
