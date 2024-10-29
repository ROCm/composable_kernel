#include "ck_tile/host.hpp"
#include "layernorm2d_fwd.hpp"
#include <cstring>
#include <algorithm>

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
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("e", "1e-5", "epsilon")
        .insert("save_mv", "0", "save mean/variance(invstd) or not. set to 1 in training case")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec_i", "fp16", "input precision")
        .insert("prec_o", "auto", "output precision, set auto will be the same as input")
        .insert("prec_s",
                "auto",
                "output quant scale type, set auto will be the same as input. used when fsweep=1")
        .insert("fadd", "0", "fused-add, 0:no fused add, 1:preadd+store, 2:preadd only")
        .insert("fsweep", "0", "fused-sweep, 0:no, 1:fused-dynamic-quant")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InDataType, typename OutDataType, typename ScaleDataType, bool SaveMeanVar>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m      = arg_parser.get_int("m");
    ck_tile::index_t n      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = n;
    float epsilon      = arg_parser.get_float("e");
    std::string prec_i = arg_parser.get_str("prec_i");
    std::string prec_o = arg_parser.get_str("prec_o");
    std::string prec_s = arg_parser.get_str("prec_s");
    if(prec_o == "auto")
    {
        prec_o = prec_i;
    }
    if(prec_s == "auto")
    {
        prec_s = prec_i;
    }

    int kname         = arg_parser.get_int("kname");
    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");
    int fused_add     = arg_parser.get_int("fadd");
    int fused_sweep   = arg_parser.get_int("fsweep");
    if(fused_sweep == 1 && prec_o != "int8")
    {
        std::cout << "if fused_sweep is 1, only support \"-prec_o=int8\" case" << std::endl;
        return false;
    }

    assert(stride >= n);

    using TypeConfig = LayerNormTypeConfig<InDataType, OutDataType, ScaleDataType>;

    using XDataType     = typename TypeConfig::XDataType;
    using YDataType     = typename TypeConfig::YDataType;
    using GammaDataType = typename TypeConfig::GammaDataType;
    using BetaDataType  = typename TypeConfig::BetaDataType;
    using SXDataType    = XDataType;
    using SYDataType    = XDataType;

    using MeanDataType =
        std::conditional_t<SaveMeanVar, typename TypeConfig::MeanDataType, ck_tile::null_type>;
    using InvStdDataType =
        std::conditional_t<SaveMeanVar, typename TypeConfig::InvStdDataType, ck_tile::null_type>;

    using ComputeDataType = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({m, n}, {stride, 1});
    ck_tile::HostTensor<GammaDataType> gamma_host({n});
    ck_tile::HostTensor<BetaDataType> beta_host({n});

    ck_tile::HostTensor<SXDataType> sx_host({m, n}, {stride, 1});
    ck_tile::HostTensor<SYDataType> sy_host({m, n}, {stride, 1});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}, {stride, 1});

    ck_tile::HostTensor<MeanDataType> mean_host_ref({m});
    ck_tile::HostTensor<InvStdDataType> invStd_host_ref({m});
    ck_tile::HostTensor<ScaleDataType> y_scale_host_ref({m});
    ck_tile::HostTensor<ScaleDataType> y_scale_host_dev({m});

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);
    ck_tile::FillUniformDistribution<BetaDataType>{-.5f, .5f}(beta_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_scale_buf(y_scale_host_dev.get_element_space_size_in_bytes());

    ck_tile::DeviceMem sx_buf(sx_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem sy_buf(sy_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());
    sx_buf.ToDevice(sx_host.data());

    auto prec_str = [&]() {
        auto base_str = prec_i;
        if(prec_i != prec_o)
        {
            base_str += "|" + prec_o;
        }
        if(fused_sweep == 1)
        {
            base_str += std::string("(") + prec_s + ")";
        }
        return base_str;
    }();

    std::cout << "[" << prec_str << "]"
              << " m:" << m << ", n:" << n << ", stride:" << stride << std::flush;

    layernorm2d_fwd_traits traits{prec_i, prec_o, prec_s, SaveMeanVar, fused_add, fused_sweep};

    layernorm2d_fwd_args args{x_buf.GetDeviceBuffer(),
                              fused_add != 0 ? sx_buf.GetDeviceBuffer() : nullptr,
                              gamma_buf.GetDeviceBuffer(),
                              beta_buf.GetDeviceBuffer(),
                              y_buf.GetDeviceBuffer(),
                              fused_add == 1 ? sy_buf.GetDeviceBuffer() : nullptr,
                              nullptr,
                              nullptr,
                              fused_sweep == 1 ? y_scale_buf.GetDeviceBuffer() : nullptr,
                              epsilon,
                              m,
                              n,
                              stride};

    float ave_time = layernorm2d_fwd(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    if(ave_time < 0)
    {
        std::cout << " not supported!" << std::endl << std::flush;
        return false;
    }

    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(GammaDataType) * n +
                           sizeof(BetaDataType) * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        // reference
        if(fused_add != 0)
        {
            // fused pre_add/pre_add_store
            // TODO we accumulate directly to x_host for simplcity here...

            std::transform(x_host.mData.cbegin(),
                           x_host.mData.cend(),
                           sx_host.mData.cbegin(),
                           x_host.mData.begin(),
                           std::plus<XDataType>{});
        }
        ck_tile::reference_layernorm2d_fwd<XDataType,
                                           GammaDataType,
                                           BetaDataType,
                                           ComputeDataType,
                                           YDataType,
                                           MeanDataType,
                                           InvStdDataType>(
            x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);

        if(fused_sweep == 1)
        {
            auto dquant_functor = [&](int m_, auto o_, auto acc_) {
                int N_                 = acc_.mDesc.get_lengths()[1];
                ComputeDataType absmax = 0;
                for(int n_ = 0; n_ < N_; n_++)
                {
                    const auto a = abs(acc_(m_, n_));
                    absmax       = a > absmax ? a : absmax;
                }
                y_scale_host_ref(m_) = absmax / 127.0;
                for(int n_ = 0; n_ < N_; n_++)
                {
                    o_(m_, n_) = static_cast<YDataType>(acc_(m_, n_) / y_scale_host_ref(m_));
                }
            };

            ck_tile::reference_layernorm2d_fwd<XDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               ComputeDataType,
                                               YDataType,
                                               MeanDataType,
                                               InvStdDataType>(x_host,
                                                               gamma_host,
                                                               beta_host,
                                                               y_host_ref,
                                                               mean_host_ref,
                                                               invStd_host_ref,
                                                               epsilon,
                                                               dquant_functor);
        }
        else
        {
            ck_tile::reference_layernorm2d_fwd<XDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               ComputeDataType,
                                               YDataType,
                                               MeanDataType,
                                               InvStdDataType>(
                x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);
        }

        y_buf.FromDevice(y_host_dev.data());

        ck_tile::HostTensor<SYDataType> sy_host_dev({m, n}, {stride, 1});
        if(fused_add == 1)
        {
            sy_buf.FromDevice(sy_host_dev.data());
        }

        auto [rtol, atol] = get_elimit<InDataType>();

        if(stride == n)
        {
            pass = ck_tile::check_err(
                y_host_dev, y_host_ref, std::string("OUT Error: Incorrect results!"), rtol, atol);
            if(fused_add == 1)
            {
                pass &= ck_tile::check_err(
                    sy_host_dev, x_host, std::string("ADD Error: Incorrect results!"), rtol, atol);
            }
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
                if(fused_add == 1)
                {
                    std::vector<SYDataType> sy_host_dev_row(sy_host_dev.begin() + i_r * stride,
                                                            sy_host_dev.begin() + i_r * stride + n);
                    std::vector<SYDataType> sy_host_ref_row(x_host.begin() + i_r * stride,
                                                            x_host.begin() + i_r * stride + n);
                    pass &= ck_tile::check_err(sy_host_dev_row,
                                               sy_host_ref_row,
                                               std::string("ADD[") + std::to_string(i_r) +
                                                   std::string("] Error: Incorrect results!"),
                                               rtol,
                                               atol);
                }
            }
        }
        if(fused_sweep == 1)
        {
            y_scale_buf.FromDevice(y_scale_host_dev.data());
            pass &= ck_tile::check_err(y_scale_host_dev,
                                       y_scale_host_ref,
                                       std::string("SCALE Error: Incorrect results!"),
                                       rtol,
                                       atol);
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

    std::string prec_i = arg_parser.get_str("prec_i");
    std::string prec_o = arg_parser.get_str("prec_o");
    std::string prec_s = arg_parser.get_str("prec_s");

    if(prec_o == "auto")
    {
        prec_o = prec_i;
    }
    if(prec_s == "auto")
    {
        prec_s = prec_i;
    }
    int save_mv = arg_parser.get_int("save_mv");

    // no dynamic quant case
    if(prec_i == "fp16" && prec_o == "fp16" && prec_s == "fp16" && save_mv)
    {
        return run<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, true>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "fp16" && prec_o == "fp16" && prec_s == "fp16" && !save_mv)
    {
        return run<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_s == "bf16" && save_mv)
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, true>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_s == "bf16" && !save_mv)
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, true>(arg_parser) ? 0 : -2;
    }

    // dynamic quant case, only in inference
    else if(prec_i == "fp16" && prec_o == "int8" && prec_s == "fp16" && !save_mv)
    {
        return run<ck_tile::half_t, ck_tile::int8_t, ck_tile::half_t, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "int8" && prec_s == "bf16" && !save_mv)
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t, ck_tile::bf16_t, false>(arg_parser) ? 0 : -2;
    }

    return -3;
}
