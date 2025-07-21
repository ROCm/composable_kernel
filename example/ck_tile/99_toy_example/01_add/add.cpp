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
        .insert("warmup", "200", "cold iter")
        .insert("repeat", "1000", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType, int GfxId>
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

    // --- Device Properties Query (for logging and warpSize) ---
    int deviceId;
    HIP_CHECK_ERROR(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, deviceId));

    std::cout << "Running on GPU: " << props.name << " (Architecture: " << props.gcnArchName << ")"
              << std::endl;
    std::cout << "GfxId instantiated: " << GfxId << std::endl;

    // These will hold the *values* for the ck_tile::sequence types
    // They are initialized based on the GfxId
    constexpr ck_tile::index_t selected_warp_tile = (GfxId == 1200)  ? Gfx120x::WarpTile
                                                    : (GfxId == 900) ? Gfx90x::WarpTile
                                                                     :
                                                                     /* else */ Generic::WarpTile;

    // Use if constexpr to select the compile-time constants for the current GfxId
    bool fail = false;
    if constexpr(GfxId == 1200)
    {
        std::cout << "Using gfx120x-optimized parameters (template specialization)." << std::endl;
    }
    else if constexpr(GfxId == 900)
    {
        std::cout << "Using gfx90x-optimized parameters (template specialization)." << std::endl;
    }
    else
    { // Fallback for GfxId == 0 or unknown
        std::cerr << "WARNING: No specific parameters for GfxId " << GfxId
                  << ". Using generic parameters." << std::endl;
        return fail;
    }

    using BlockWarps =
        ck_tile::sequence<1, 8>; // number of concurrent warps in one block (if 8 warps * 64 threads
                                 // per warp, 512 threads in one block are NEEDED)
    using BlockTile =
        ck_tile::sequence<1, 4096>; // shape of one blockTile (elements covered by one block)
    using WarpTile = ck_tile::sequence<1, 8 * selected_warp_tile>; // shape of one warpTile
                                                                   // (elements covered by one warp
                                                                   // (32/64 threads))
    using Vector = ck_tile::sequence<1, 8>; // shape of one vector (elements covered by one thread)

    constexpr ck_tile::index_t kBlockSize =
        512; // number of blockWarps * number of threads per warp
    constexpr ck_tile::index_t kBlockPerCu = 1;
    ck_tile::index_t kGridSize             = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape   = ck_tile::AddShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Porblem = ck_tile::AddProblem<XDataType, ComputeDataType, YDataType, Shape>;

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

    std::size_t num_btype = 2 * sizeof(XDataType) * m * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_add<XDataType, YDataType>(x_host_a, x_host_b, y_host_ref);
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

    int deviceId;
    HIP_CHECK_ERROR(hipGetDevice(&deviceId));
    hipDeviceProp_t props;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, deviceId));
    std::string arch_name = props.gcnArchName;

    if(data_type == "fp16" && (arch_name.find("gfx12") != std::string::npos))
        return run<ck_tile::half_t, 1200>(arg_parser) ? 0 : -2;
    else if(data_type == "fp16" && (arch_name.find("gfx908") != std::string::npos))
        return run<ck_tile::half_t, 900>(arg_parser) ? 0 : -2;
    else
    {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
        return -1;
    }
}
