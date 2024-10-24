#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/rmsnorm2d.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("e", "1e-5", "epsilon")
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
    ck_tile::index_t m      = arg_parser.get_int("m");
    ck_tile::index_t n      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = n;
    float epsilon         = arg_parser.get_float("e");
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    assert(stride >= n);

    using XDataType      = DataType;
    using YDataType      = DataType;
    using GammaDataType  = DataType;
    using InvRmsDataType = ck_tile::null_type;

    using ComputeDataType = float;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({m, n}, {stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}, {stride, 1});

    ck_tile::HostTensor<InvRmsDataType> invRms_host_ref({m});

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());

    constexpr bool kTwoPass = true;

    using BlockWarps = ck_tile::sequence<2, 2>;
    using BlockTile  = ck_tile::sequence<2, 256>;
    using WarpTile   = ck_tile::sequence<1, 64>;
    using Vector     = ck_tile::sequence<1, 1>;

    using Shape   = ck_tile::Rmsnorm2dShape<BlockTile, BlockWarps, WarpTile, Vector>;
    using Problem = ck_tile::Rmsnorm2dFwdPipelineProblem<XDataType,
                                                         GammaDataType,
                                                         ComputeDataType,
                                                         YDataType,
                                                         InvRmsDataType,
                                                         Shape,
                                                         true,  // kPadN
                                                         false, // kSaveInvRms
                                                         kTwoPass>;

    using OnePassPipeline = ck_tile::Rmsnorm2dFwdPipelineOnePass<Problem>;
    using TwoPassPipeline = ck_tile::Rmsnorm2dFwdPipelineTwoPass<Problem>;
    using Pipeline        = std::conditional_t<kTwoPass, TwoPassPipeline, OnePassPipeline>;
    using Kernel          = ck_tile::Rmsnorm2dFwd<Pipeline>;

    ck_tile::Rmsnorm2dFwdHostArgs args{x_buf.GetDeviceBuffer(),
                                       gamma_buf.GetDeviceBuffer(),
                                       y_buf.GetDeviceBuffer(),
                                       nullptr,
                                       epsilon,
                                       m,
                                       n,
                                       stride};

    auto kargs = Kernel::MakeKargs(args);

    const dim3 grids                       = Kernel::GridSize(args);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;
    auto s = ck_tile::stream_config{nullptr, true, 0, warmup, repeat};

    ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_rmsnorm2d_fwd<XDataType,
                                         GammaDataType,
                                         ComputeDataType,
                                         YDataType,
                                         InvRmsDataType>(
            x_host, gamma_host, y_host_ref, invRms_host_ref, epsilon);

        y_buf.FromDevice(y_host_dev.data());

        auto [rtol, atol] = ck_tile::make_tuple(1e-3, 1e-3);
        if(stride == n)
        {
            pass = ck_tile::check_err(
                y_host_dev, y_host_ref, std::string("OUT Error: Incorrect results!"), rtol, atol);
        }
        else
        {
            for(int i_r = 0; i_r < m; i_r++)
            {
                std::vector<YDataType> y_host_dev_row(y_host_dev.begin() + i_r * stride,
                                                      y_host_dev.begin() + i_r * stride + n);
                std::vector<YDataType> y_host_ref_row(y_host_ref.begin() + i_r * stride,
                                                      y_host_ref.begin() + i_r * stride + n);
                pass &= ck_tile::check_err(y_host_dev_row,
                                           y_host_ref_row,
                                           std::string("OUT[") + std::to_string(i_r) +
                                               std::string("] Error: Incorrect results!"),
                                           rtol,
                                           atol);
            }
        }

        std::cout << "[" << data_type << "]"
                  << " m:" << m << ", n:" << n << ", stride:" << stride
                  << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
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
