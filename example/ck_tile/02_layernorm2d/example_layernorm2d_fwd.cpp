#include "ck_tile/host.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include "ck_tile/ops/layernorm2d.hpp"
#include <algorithm>
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

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("x_stride", "-1", "x row_stride, if -1 then equal to n")
        .insert("xr_stride", "-1", "x residule row_stride, if -1 then equal to n")
        .insert("y_stride", "-1", "y row_stride, if -1 then equal to n")
        .insert("yr_stride", "-1", "y residule row_stride, if -1 then equal to n")
        .insert("e", "1e-5", "epsilon")
        .insert("v", "1", "cpu validation or not")
        .insert("prec_i", "fp16", "input precision")
        .insert("prec_o", "auto", "output precision, set auto will be the same as input")
        .insert("prec_sx",
                "auto",
                "output quant scale type, set auto will use fp32. used when fquant=1")
        .insert("prec_sy",
                "auto",
                "output quant scale type, set auto will use fp32. used when fquant=1 or 2")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InDataType,
          typename OutDataType,
          typename XScaleDataType,
          typename YScaleDataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m        = arg_parser.get_int("m");
    ck_tile::index_t n        = arg_parser.get_int("n");
    ck_tile::index_t x_stride = arg_parser.get_int("x_stride");
    if(x_stride < 0)
        x_stride = n;
    ck_tile::index_t xr_stride = arg_parser.get_int("xr_stride");
    if(xr_stride < 0)
        xr_stride = n;
    ck_tile::index_t y_stride = arg_parser.get_int("y_stride");
    if(y_stride < 0)
        y_stride = n;
    ck_tile::index_t yr_stride = arg_parser.get_int("yr_stride");
    if(yr_stride < 0)
        yr_stride = n;
    float epsilon       = arg_parser.get_float("e");
    std::string prec_i  = arg_parser.get_str("prec_i");
    std::string prec_o  = arg_parser.get_str("prec_o");
    std::string prec_sx = arg_parser.get_str("prec_sx");
    std::string prec_sy = arg_parser.get_str("prec_sy");
    if(prec_o == "auto")
    {
        prec_o = prec_i;
    }
    if(prec_sx == "auto")
    {
        prec_sx = "fp32";
    }
    if(prec_sy == "auto")
    {
        prec_sy = "fp32";
    }

    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");

    assert(x_stride >= n);

    using XDataType         = InDataType;
    using YDataType         = InDataType;
    using GammaDataType     = InDataType;
    using BetaDataType      = InDataType;
    using XResidualDataType = InDataType;
    using YResidualDataType = InDataType;

    using ComputeDataType = float;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({m, n}, {x_stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});
    ck_tile::HostTensor<BetaDataType> beta_host({n});

    ck_tile::HostTensor<XResidualDataType> x_residual_host({m, n}, {xr_stride, 1});
    ck_tile::HostTensor<YResidualDataType> y_residual_host({m, n}, {yr_stride, 1});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n}, {y_stride, 1});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}, {y_stride, 1});

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<XResidualDataType>{-.5f, .5f}(x_residual_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);
    ck_tile::FillUniformDistribution<BetaDataType>{-.5f, .5f}(beta_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    ck_tile::DeviceMem x_residual_buf(x_residual_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_residual_buf(y_residual_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());
    x_residual_buf.ToDevice(x_residual_host.data());

    constexpr bool kTwoPass   = false;
    constexpr auto kFuseAdd   = ck_tile::Layernorm2dFusedAddEnum::PRE_ADD_STORE;
    constexpr auto kFuseQuant = ck_tile::Layernorm2dFusedQuantEnum::NO_SWEEP;

    using BlockWarps = ck_tile::sequence<1, 4>;
    using BlockTile  = ck_tile::sequence<1, 8192>;
    using WarpTile   = ck_tile::sequence<1, 512>;
    using Vector     = ck_tile::sequence<1, 8>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
    using Trait = ck_tile::Layernorm2dFwdTraits<true, false, true, kTwoPass, kFuseAdd, kFuseQuant>;
    using PipelineProblem = ck_tile::Layernorm2dFwdPipelineProblem<XDataType,
                                                                   GammaDataType,
                                                                   BetaDataType,
                                                                   ComputeDataType,
                                                                   YDataType,
                                                                   ck_tile::null_type,
                                                                   ck_tile::null_type,
                                                                   ComputeDataType,
                                                                   ComputeDataType,
                                                                   Shape,
                                                                   Trait>;

    using OnePassPipeline = ck_tile::Layernorm2dFwdPipelineOnePass<PipelineProblem>;
    using TwoPassPipeline = ck_tile::Layernorm2dFwdPipelineTwoPass<PipelineProblem>;
    using Pipeline        = std::conditional_t<kTwoPass, TwoPassPipeline, OnePassPipeline>;

    using EpilogueProblem =
        ck_tile::Default2DEpilogueProblem<ComputeDataType, YDataType, false, true, false>;
    using Epilogue = ck_tile::Default2DEpilogue<EpilogueProblem>;
    using Kernel   = ck_tile::Layernorm2dFwd<Pipeline, Epilogue>;

    ck_tile::Layernorm2dFwdHostArgs args{x_buf.GetDeviceBuffer(),
                                         x_residual_buf.GetDeviceBuffer(),
                                         nullptr, // x_scale for quant
                                         gamma_buf.GetDeviceBuffer(),
                                         beta_buf.GetDeviceBuffer(),
                                         y_buf.GetDeviceBuffer(),
                                         y_residual_buf.GetDeviceBuffer(),
                                         nullptr, // y_scale for quant
                                         nullptr, // p_mean, unsupported yet
                                         nullptr, // p_invStd, unsupported yet
                                         epsilon,
                                         m,
                                         n,
                                         x_stride,   // x row_stride
                                         xr_stride,  // x residule row stride
                                         y_stride,   // y row stride
                                         yr_stride}; // y residule row stride

    auto kargs = Kernel::MakeKargs(args);

    const dim3 grids                       = Kernel::GridSize(args);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;
    auto s = ck_tile::stream_config{nullptr, true, 1, warmup, repeat};

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(GammaDataType) * n +
                           sizeof(BetaDataType) * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << Kernel::GetName() << std::endl;
    std::cout << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        std::transform(x_host.mData.cbegin(),
                       x_host.mData.cend(),
                       x_residual_host.mData.cbegin(),
                       x_host.mData.begin(),
                       [](auto x_, auto r_) {
                           auto o_ = ck_tile::type_convert<ComputeDataType>(x_) +
                                     ck_tile::type_convert<ComputeDataType>(r_);
                           return ck_tile::type_convert<XDataType>(o_);
                       });

        ck_tile::HostTensor<ck_tile::null_type> dummy({m});

        ck_tile::reference_layernorm2d_fwd<XDataType,
                                           GammaDataType,
                                           BetaDataType,
                                           ComputeDataType,
                                           YDataType,
                                           ck_tile::null_type,
                                           ck_tile::null_type>(
            x_host, gamma_host, beta_host, y_host_ref, dummy, dummy, epsilon);

        y_buf.FromDevice(y_host_dev.data());

        ck_tile::HostTensor<YResidualDataType> y_residual_host_dev({m, n}, {yr_stride, 1});
        y_residual_buf.FromDevice(y_residual_host_dev.data());

        auto [rtol, atol] = get_elimit<InDataType>();

        if(x_stride == n)
        {
            pass &= ck_tile::check_err(y_residual_host_dev,
                                       x_host,
                                       std::string(" ADD Error: Incorrect results!"),
                                       rtol,
                                       atol);

            pass &= ck_tile::check_err(
                y_host_dev, y_host_ref, std::string(" OUT Error: Incorrect results!"), rtol, atol);
        }

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string prec_i  = arg_parser.get_str("prec_i");
    std::string prec_o  = arg_parser.get_str("prec_o");
    std::string prec_sx = arg_parser.get_str("prec_sx");
    std::string prec_sy = arg_parser.get_str("prec_sy");

    if(prec_o == "auto")
    {
        prec_o = prec_i;
    }
    if(prec_sx == "auto")
    {
        prec_sx = "fp32";
    }
    if(prec_sy == "auto")
    {
        prec_sy = "fp32";
    }

    // no dynamic quant case
    /*if(prec_i == "fp16" && prec_o == "fp16" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::half_t, ck_tile::half_t, float, float>(arg_parser) ? 0 : -2;
    }
    else */
    if(prec_i == "bf16" && prec_o == "bf16" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, float, float>(arg_parser) ? 0 : -2;
    }

    return -3;
}
