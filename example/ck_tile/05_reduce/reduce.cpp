#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "2048", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("k", "2", "k dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "0", "cold iter")
        .insert("repeat", "1", "hot iter");

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
    ck_tile::index_t k = arg_parser.get_int("k");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    std::vector<ck_tile::index_t> problem_shape = {m, n, k};
    std::vector<ck_tile::index_t> strides(3);
    strides[0] = n * k;
    strides[1] = k;
    strides[2] = 1;

    // Define reduction specification:
    // dimension 0 is kept, dimensions 1,2 are reduced
    constexpr auto kept_dim    = ck_tile::sequence<0>{}; // Which dimension to keep (pass-through)
    constexpr auto reduce_dims = ck_tile::sequence<1, 2>{}; // Which dimensions to reduce (merge)

    ck_tile::HostTensor<XDataType> x_host(problem_shape, strides);
    ck_tile::HostTensor<YDataType> y_host_ref({problem_shape[kept_dim.at(0)]}, {1});
    ck_tile::HostTensor<YDataType> y_host_dev({problem_shape[kept_dim.at(0)]}, {1});

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    using ReduceOp   = ck_tile::ReduceOp::Add;
    using BlockWarps = ck_tile::sequence<4, 1>;
    using BlockTile  = ck_tile::sequence<128, 128>;
    using WarpTile   = ck_tile::sequence<32, 128>;
    using Vector     = ck_tile::sequence<8, 8>;

    // cross warp-reduce
    // using BlockWarps = ck_tile::sequence<2, 2>;
    // using BlockTile  = ck_tile::sequence<2, 1024>;
    // using WarpTile   = ck_tile::sequence<1, 512>;
    // using Vector = ck_tile::sequence<1, 8>;

    constexpr ck_tile::index_t kBlockSize  = 256;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    ck_tile::index_t kGridSize =
        (problem_shape[kept_dim.at(0)] + BlockTile::at(ck_tile::number<0>{}) - 1) /
        BlockTile::at(ck_tile::number<0>{});
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape = ck_tile::Reduce2dShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Porblem =
        ck_tile::Reduce2dProblem<XDataType, ComputeDataType, YDataType, Shape, ReduceOp>;

    using Kernel = ck_tile::Reduce<Porblem>;

    // Create input tensor shape and strides
    auto input_shape   = ck_tile::make_tuple(problem_shape[0], problem_shape[1], problem_shape[2]);
    auto input_strides = ck_tile::make_tuple(strides[0], strides[1], strides[2]);

    if(!Kernel::IsSupportedArgument(arg_parser))
    {
        throw std::runtime_error("Wrong! Arguments not supported!\n");
    }

    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       input_shape,
                                       input_strides,
                                       kept_dim,
                                       reduce_dims));

    std::size_t num_btype =
        sizeof(XDataType) * m * n * k + sizeof(YDataType) * problem_shape[kept_dim.at(0)];

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_reduce<XDataType, ComputeDataType, YDataType>(
            x_host, y_host_ref, ReduceOp{}, kept_dim, reduce_dims);
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
    // else if(data_type == "bf16")
    // {
    //     return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    // }
}
