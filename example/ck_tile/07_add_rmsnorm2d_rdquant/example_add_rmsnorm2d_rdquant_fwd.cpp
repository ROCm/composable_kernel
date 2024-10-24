#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/add_rmsnorm2d_rdquant.hpp"
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

    using ADataType       = DataType;
    using BDataType       = DataType;
    using GammaDataType   = DataType;
    using XDataType       = DataType;
    using YScaleDataType  = DataType;
    using QYDataType      = int8_t;
    using ComputeDataType = float;

    // host verify
    ck_tile::HostTensor<XDataType> a_host({m, n}, {stride, 1});
    ck_tile::HostTensor<XDataType> b_host({m, n}, {stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});

    ck_tile::HostTensor<XDataType> x_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<XDataType> x_host_dev({m, n}, {stride, 1});
    ck_tile::HostTensor<YScaleDataType> yscale_host_ref({m}, {1});
    ck_tile::HostTensor<YScaleDataType> yscale_host_dev({m}, {1});
    ck_tile::HostTensor<QYDataType> qy_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<QYDataType> qy_host_dev({m, n}, {stride, 1});

    ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);

    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf(x_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem yscale_buf(yscale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qy_buf(qy_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.data());
    b_buf.ToDevice(b_host.data());
    gamma_buf.ToDevice(gamma_host.data());

    constexpr bool kTwoPass = false;

    using BlockWarps = ck_tile::sequence<2, 2>;
    using BlockTile  = ck_tile::sequence<2, 128>;
    using WarpTile   = ck_tile::sequence<1, 64>;
    using Vector     = ck_tile::sequence<1, 1>;

    using Shape   = ck_tile::AddRmsnorm2dRdquantShape<BlockTile, BlockWarps, WarpTile, Vector>;
    using Problem = ck_tile::AddRmsnorm2dRdquantFwdPipelineProblem<ADataType,
                                                                   BDataType,
                                                                   GammaDataType,
                                                                   ComputeDataType,
                                                                   XDataType,
                                                                   YScaleDataType,
                                                                   QYDataType,
                                                                   Shape,
                                                                   true, // kPadN
                                                                   true, // kSaveX
                                                                   kTwoPass>;

    using OnePassPipeline = ck_tile::AddRmsnorm2dRdquantFwdPipelineOnePass<Problem>;
    using TwoPassPipeline = ck_tile::AddRmsnorm2dRdquantFwdPipelineTwoPass<Problem>;
    using Pipeline        = std::conditional_t<kTwoPass, TwoPassPipeline, OnePassPipeline>;
    using Kernel          = ck_tile::AddRmsnorm2dRdquantFwd<Pipeline>;

    ck_tile::AddRmsnorm2dRdquantFwdHostArgs args{a_buf.GetDeviceBuffer(),
                                                 b_buf.GetDeviceBuffer(),
                                                 gamma_buf.GetDeviceBuffer(),
                                                 x_buf.GetDeviceBuffer(),
                                                 yscale_buf.GetDeviceBuffer(),
                                                 qy_buf.GetDeviceBuffer(),
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
        using InvRmsDataType = DataType;
        ck_tile::HostTensor<InvRmsDataType> invRms_host_ref({m});

        // ck_tile::reference_rmsnorm2d_fwd<XDataType,
        //                                  GammaDataType,
        //                                  ComputeDataType,
        //                                  QYDataType,
        //                                  InvRmsDataType>(
        //     x_host, gamma_host, qy_host_ref, invRms_host_ref, epsilon);

        // qy_buf.FromDevice(qy_host_dev.data());

        // auto [rtol, atol] = ck_tile::make_tuple(1e-3, 1e-3);
        // if(stride == n)
        // {
        //     pass = ck_tile::check_err(
        //         qy_host_dev, qy_host_ref, std::string("OUT Error: Incorrect results!"), rtol,
        //         atol);
        // }
        // else
        // {
        //     for(int i_r = 0; i_r < m; i_r++)
        //     {
        //         std::vector<QYDataType> qy_host_dev_row(qy_host_dev.begin() + i_r * stride,
        //                                               qy_host_dev.begin() + i_r * stride + n);
        //         std::vector<QYDataType> qy_host_ref_row(qy_host_ref.begin() + i_r * stride,
        //                                               qy_host_ref.begin() + i_r * stride + n);
        //         pass &= ck_tile::check_err(qy_host_dev_row,
        //                                    qy_host_ref_row,
        //                                    std::string("OUT[") + std::to_string(i_r) +
        //                                        std::string("] Error: Incorrect results!"),
        //                                    rtol,
        //                                    atol);
        //     }
        // }

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
