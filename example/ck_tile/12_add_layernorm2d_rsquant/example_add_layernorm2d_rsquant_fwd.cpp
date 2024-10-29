#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/add_rmsnorm2d_rdquant.hpp"
#include <cstring>

// different threshold for different dtype
template <typename DataType>
auto get_elimit()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::int8_t>()
{
    // due to rounding, int8 quantization might have 1 abs error
    double rtol = 1;
    double atol = 1;
    return ck_tile::make_tuple(rtol, atol);
}

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
    using BetaDataType    = DataType;
    using XDataType       = DataType;
    using YScaleDataType  = DataType;
    using QYDataType      = ck_tile::int8_t;
    using ComputeDataType = float;

    // input
    ck_tile::HostTensor<ADataType> a_host({m, n}, {stride, 1});
    ck_tile::HostTensor<BDataType> b_host({m, n}, {stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});
    ck_tile::HostTensor<BetaDataType> beta_host({n});
    ck_tile::HostTensor<YScaleDataType> yscale_host({m}, {1});

    // output
    ck_tile::HostTensor<XDataType> x_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<XDataType> x_host_dev({m, n}, {stride, 1});
    ck_tile::HostTensor<QYDataType> qy_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<QYDataType> qy_host_dev({m, n}, {stride, 1});

    ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f}(b_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);
    ck_tile::FillUniformDistribution<BetaDataType>{-.5f, .5f}(beta_host);
    ck_tile::FillUniformDistribution<YScaleDataType>{1e-3, .1f}(yscale_host);

    ck_tile::DeviceMem a_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf(x_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem yscale_buf(yscale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qy_buf(qy_host_dev.get_element_space_size_in_bytes());

    a_buf.ToDevice(a_host.data());
    b_buf.ToDevice(b_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());
    yscale_buf.ToDevice(yscale_host.data());

    constexpr bool kThreePass = false;

    using BlockWarps = ck_tile::sequence<2, 2>;
    using BlockTile  = ck_tile::sequence<2, 128>;
    using WarpTile   = ck_tile::sequence<1, 64>;
    using Vector     = ck_tile::sequence<1, 1>;

    // using Shape   = ck_tile::AddLayernorm2dRsquantShape<BlockTile, BlockWarps, WarpTile, Vector>;
    // using Problem = ck_tile::AddLayernorm2dRsquantFwdPipelineProblem<ADataType,
    //                                                                  BDataType,
    //                                                                  GammaDataType,
    //                                                                  BetaDataType,
    //                                                                  YScaleDataType,
    //                                                                  ComputeDataType,
    //                                                                  XDataType,
    //                                                                  QYDataType,
    //                                                                  Shape,
    //                                                                  true, // kPadN
    //                                                                  true, // kSaveX
    //                                                                  kThreePass>;

    // using OnePassPipeline   = ck_tile::AddLayernorm2dRsquantFwdPipelineOnePass<Problem>;
    // using ThreePassPipeline = ck_tile::AddLayernorm2dRsquantFwdPipelineThreePass<Problem>;
    // using Pipeline          = std::conditional_t<kThreePass, ThreePassPipeline, OnePassPipeline>;
    // using Kernel            = ck_tile::AddLayernorm2dRsquantFwd<Pipeline>;

    // ck_tile::AddLayernorm2dRsquantFwdHostArgs args{a_buf.GetDeviceBuffer(),
    //                                                b_buf.GetDeviceBuffer(),
    //                                                gamma_buf.GetDeviceBuffer(),
    //                                                beta_buf.GetDeviceBuffer(),
    //                                                yscale_buf.GetDeviceBuffer(),
    //                                                x_buf.GetDeviceBuffer(),
    //                                                qy_buf.GetDeviceBuffer(),
    //                                                epsilon,
    //                                                m,
    //                                                n,
    //                                                stride};

    // auto kargs = Kernel::MakeKargs(args);

    // const dim3 grids                       = Kernel::GridSize(args);
    // constexpr dim3 blocks                  = Kernel::BlockSize();
    // constexpr ck_tile::index_t kBlockPerCu = 1;
    // auto s = ck_tile::stream_config{nullptr, true, 0, warmup, repeat};

    // ck_tile::launch_kernel(
    //     s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    bool pass = true;

    if(do_validation)
    {
        using YDataType      = ComputeDataType;
        using MeanDataType   = DataType;
        using InvStdDataType = DataType;

        // Add
        {
            auto op = [](const auto& v0, const auto& v1) { return v0 + v1; };
            ck_tile::reference_binary_elementwise<ADataType, BDataType, XDataType, ComputeDataType>(
                a_host, b_host, x_host_ref, op);

            x_buf.FromDevice(x_host_dev.data());

            auto [rtol, atol] = get_elimit<XDataType>();
            if(stride == n)
            {
                pass = ck_tile::check_err(
                    x_host_dev, x_host_ref, std::string("x Error: Incorrect results!"), rtol, atol);
            }
            else
            {
                for(int i_r = 0; i_r < m; i_r++)
                {
                    std::vector<QYDataType> x_host_dev_row(x_host_dev.begin() + i_r * stride,
                                                           x_host_dev.begin() + i_r * stride + n);
                    std::vector<QYDataType> x_host_ref_row(x_host_ref.begin() + i_r * stride,
                                                           x_host_ref.begin() + i_r * stride + n);
                    pass &= ck_tile::check_err(x_host_dev_row,
                                               x_host_ref_row,
                                               std::string("x[") + std::to_string(i_r) +
                                                   std::string("] Error: Incorrect results!"),
                                               rtol,
                                               atol);
                }
            }
        }

        ck_tile::HostTensor<YDataType> y_host({m, n});
        // Layernorm2d
        {
            ck_tile::HostTensor<MeanDataType> mean_host_ref({m});
            ck_tile::HostTensor<InvStdDataType> invStd_host_ref({m});

            // CAUSION: kernel use ComputeDataType version of x, but we use XDataType here for
            // simplicity
            ck_tile::reference_layernorm2d_fwd<XDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               ComputeDataType,
                                               YDataType,
                                               MeanDataType,
                                               InvStdDataType>(
                x_host_ref, gamma_host, beta_host, y_host, mean_host_ref, invStd_host_ref, epsilon);
        }

        // rowwise quantization
        {
            ck_tile::reference_rowwise_quantization2d<YDataType, YScaleDataType, QYDataType>(
                y_host, yscale_host, qy_host_ref);

            qy_buf.FromDevice(qy_host_dev.data());
            auto [rtol, atol] = get_elimit<QYDataType>();

            if(stride == n)
            {
                pass = ck_tile::check_err(qy_host_dev,
                                          qy_host_ref,
                                          std::string("qy Error: Incorrect results!"),
                                          rtol,
                                          atol);
            }
            else
            {
                for(int i_r = 0; i_r < m; i_r++)
                {
                    std::vector<QYDataType> qy_host_dev_row(qy_host_dev.begin() + i_r * stride,
                                                            qy_host_dev.begin() + i_r * stride + n);
                    std::vector<QYDataType> qy_host_ref_row(qy_host_ref.begin() + i_r * stride,
                                                            qy_host_ref.begin() + i_r * stride + n);
                    pass &= ck_tile::check_err(qy_host_dev_row,
                                               qy_host_ref_row,
                                               std::string("qy[") + std::to_string(i_r) +
                                                   std::string("] Error: Incorrect results!"),
                                               rtol,
                                               atol);
                }
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
