#include "ck_tile/host.hpp"
#include "layernorm2d_fwd.hpp"
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


// mfma_type, 0:32x32, 1:16x16
template<typename H>
auto shuffle_moe_weight(const H& t, std::string mfma_dtype, int mfma_type = 0)
{
    static_assert(t.get_lengths().size() == 3);
    int b_ = t.get_lengths()[0];
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[2];
    if ((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 0) {
        std::vector<ck_tile::index_t> new_lens {b_, n_/32, 32, k_/16, 2, 8};
}
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("t", "128", "num input tokens")
        .insert("e", "32", "num of experts")
        .insert("k", "5", "topk")
        .insert("h", "8192", "hidden_size of this model")
        .insert("i", "8192", "intermediate_size between 2 gemms of FFN")
        .insert("stride", "-1", "stride per row, if -1 then equal to hidden_size")
        .insert("bm", "32", "blocking factor for sorted tokens")
        .insert("tp", "8", "tensor parallel size")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec_i", "bf16", "input precision")
        .insert("prec_w", "bf16", "weight precision")
        .insert("prec_o", "bf16", "output precision")
        .insert("prec_st", "auto", "token scale data type. auto will set to fp32")
        .insert("prec_sw", "auto", "weight scale data type. auto will set to fp32")
        .insert("prec_sq", "auto", "(dynamic) smooth quant data type. auto will set to fp32")
        .insert("prec_kw", "auto", "topk-weight data type. auto will set to fp32")
        .insert("fquant", "0", "fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant")
        .insert("gonly", "0", "w0(gate/up) style, 0:gate+up will double interm size, 1:only gate")
        .insert("balance", "1", "if set to 1, will try balance the expert in topk-ids(convenient for testing)")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// I:input-type, W:weight-type, O:output-type, ST:toke-scale-tpye, SW:weight-scale-type, SQ:smooth-quant-type, KW:topk-weight-type
template <typename I, typename W, typename O, typename ST, typename SW, typename SQ, typename KW>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t tokens   = arg_parser.get_int("t");
    ck_tile::index_t experts  = arg_parser.get_int("e");
    ck_tile::index_t topk    = arg_parser.get_int("k");
    ck_tile::index_t hidden_size    = arg_parser.get_int("h");
    ck_tile::index_t intermediate_size    = arg_parser.get_int("i");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    ck_tile::index_t block_m = arg_parser.get_int("bm");
    if(stride < 0)
        stride = hidden_size;
    std::string prec_i  = arg_parser.get_str("prec_i");
    std::string prec_w = arg_parser.get_str("prec_w");
    std::string prec_o  = arg_parser.get_str("prec_o");
    std::string prec_st  = arg_parser.get_str("prec_st");
    std::string prec_sw  = arg_parser.get_str("prec_sw");
    std::string prec_sq  = arg_parser.get_str("prec_sq");
    std::string prec_kw = arg_parser.get_str("prec_kw");
    prec_st = (prec_st == "auto") ? "fp32" : prec_st;
    prec_sw = (prec_sw == "auto") ? "fp32" : prec_sw;
    prec_sq = (prec_sq == "auto") ? "fp32" : prec_sq;
    prec_kw = (prec_kw == "auto") ? "fp32" : prec_kw;
    int kname         = arg_parser.get_int("kname");
    int do_validation = arg_parser.get_int("v");
    int warmup        = arg_parser.get_int("warmup");
    int repeat        = arg_parser.get_int("repeat");
    int fused_quant   = arg_parser.get_int("fquant");
    int gonly = arg_parser.get_int("gonly");
    int balance = arg_parser.get_int("balance");
    int tp = arg_parser.get_int("tp");
    ck_tile::index_t shared_intermediate_size = intermediate_size * (gonly ? 1 : 2) / tp;


    using TypeConfig = FusedMoeGemmTypeConfig<I, W, O, ST, SW, SQ, KW>;
    using ADataType             = typename TypeConfig::ADataType           ;
    using GDataType             = typename TypeConfig::GDataType           ;
    using DDataType             = typename TypeConfig::DDataType           ;
    using AccDataType           = typename TypeConfig::AccDataType         ;
    using ODataType             = typename TypeConfig::ODataType           ;
    using AScaleDataType        = typename TypeConfig::AScaleDataType      ;
    using W0ScaleDataType       = typename TypeConfig::W0ScaleDataType     ;
    using W1ScaleDataType       = typename TypeConfig::W1ScaleDataType     ;
    using YSmoothScaleDataType  = typename TypeConfig::YSmoothScaleDataType;
    using TopkWeightDataType    = typename TypeConfig::TopkWeightDataType  ;
    using IndexDataType         = typename TypeConfig::IndexDataType       ;

    // host verify
    ck_tile::HostTensor<ADataType> a_host({tokens, hidden_size}, {stride, 1});
    ck_tile::HostTensor<ADataType> g_host({e, shared_intermediate_size, hidden_size});
    ck_tile::HostTensor<ADataType> d_host({e, intermediate_size, hidden_size});


    ck_tile::HostTensor<XResidualDataType> x_residual_host({m, n}, {stride, 1});
    ck_tile::HostTensor<YResidualDataType> y_residual_host({m, n}, {stride, 1});

    ck_tile::HostTensor<YDataType> y_host_ref({m, n}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}, {stride, 1});

    ck_tile::HostTensor<MeanDataType> mean_host_ref({m});
    ck_tile::HostTensor<InvStdDataType> invStd_host_ref({m});
    ck_tile::HostTensor<YScaleDataType> y_scale_host_ref({m});
    ck_tile::HostTensor<YScaleDataType> y_scale_host_dev({m});

    ck_tile::HostTensor<XScaleDataType> x_scale_host({n});
    ck_tile::HostTensor<XScaleDataType> x_scale_host_dev({n});

    ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f}(a_host);
    ck_tile::FillUniformDistribution<XResidualDataType>{-.5f, .5f}(x_residual_host);
    ck_tile::FillUniformDistribution<XScaleDataType>{-1.f, 1.f}(x_scale_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);
    ck_tile::FillUniformDistribution<BetaDataType>{-.5f, .5f}(beta_host);

    ck_tile::DeviceMem x_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_scale_buf(y_scale_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_scale_buf(x_scale_host_dev.get_element_space_size_in_bytes());

    ck_tile::DeviceMem x_residual_buf(x_residual_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_residual_buf(y_residual_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(a_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());
    x_residual_buf.ToDevice(x_residual_host.data());
    x_scale_buf.ToDevice(x_scale_host.data());

    auto prec_str = [&]() {
        auto base_str = prec_i;
        if(prec_i != prec_o)
        {
            base_str += "|" + prec_o;
        }
        if(fused_quant == 1)
        {
            base_str += std::string("(") + prec_sy + ")";
        }
        return base_str;
    }();

    std::cout << "[" << prec_str << "]"
              << " m:" << m << ", n:" << n << ", stride:" << stride << std::flush;

    layernorm2d_fwd_traits traits{
        prec_i, prec_o, prec_sx, prec_sy, SaveMeanVar, fused_add, fused_quant};

    layernorm2d_fwd_args args{x_buf.GetDeviceBuffer(),
                              fused_add != 0 ? x_residual_buf.GetDeviceBuffer() : nullptr,
                              fused_quant == 1 ? x_scale_buf.GetDeviceBuffer() : nullptr,
                              gamma_buf.GetDeviceBuffer(),
                              beta_buf.GetDeviceBuffer(),

                              y_buf.GetDeviceBuffer(),
                              fused_add == 1 ? y_residual_buf.GetDeviceBuffer() : nullptr,
                              fused_quant != 0 ? y_scale_buf.GetDeviceBuffer() : nullptr,
                              nullptr, // p_mean, unsupported yet
                              nullptr, // p_invStd, unsupported yet

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

    std::size_t num_byte = sizeof(ADataType) * m * n + sizeof(GammaDataType) * n +
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
            // TODO we accumulate directly to a_host for simplcity here...

            std::transform(a_host.mData.cbegin(),
                           a_host.mData.cend(),
                           x_residual_host.mData.cbegin(),
                           a_host.mData.begin(),
                           [](auto x_, auto r_) {
                               auto o_ = ck_tile::type_convert<ComputeDataType>(x_) +
                                         ck_tile::type_convert<ComputeDataType>(r_);
                               return ck_tile::type_convert<ADataType>(o_);
                           });
        }
        ck_tile::reference_layernorm2d_fwd<ADataType,
                                           GammaDataType,
                                           BetaDataType,
                                           ComputeDataType,
                                           YDataType,
                                           MeanDataType,
                                           InvStdDataType>(
            a_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);

        if(fused_quant != 0)
        {
            auto dquant_functor = [&](int m_, auto& o_, auto& acc_) {
                int N_ = acc_.mDesc.get_lengths()[1];
                if(fused_quant == 1)
                {
                    for(int n_ = 0; n_ < N_; n_++)
                    {
                        // input smooth outlier
                        acc_(m_, n_) =
                            acc_(m_, n_) * ck_tile::type_convert<ComputeDataType>(x_scale_host(n_));
                    }
                }
                ComputeDataType absmax = static_cast<ComputeDataType>(0);
                for(int n_ = 0; n_ < N_; n_++)
                {
                    const auto a = ck_tile::abs(acc_(m_, n_));
                    absmax       = a > absmax ? a : absmax;
                }
                // printf("cpu:absmax:%f\n", absmax);
                ComputeDataType y_scale = absmax / static_cast<ComputeDataType>(127.0);
                y_scale_host_ref(m_)    = ck_tile::type_convert<YScaleDataType>(y_scale);
                for(int n_ = 0; n_ < N_; n_++)
                {
                    o_(m_, n_) = ck_tile::type_convert<YDataType>(acc_(m_, n_) / y_scale);
                }
            };

            ck_tile::reference_layernorm2d_fwd<ADataType,
                                               GammaDataType,
                                               BetaDataType,
                                               ComputeDataType,
                                               YDataType,
                                               MeanDataType,
                                               InvStdDataType>(a_host,
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
            ck_tile::reference_layernorm2d_fwd<ADataType,
                                               GammaDataType,
                                               BetaDataType,
                                               ComputeDataType,
                                               YDataType,
                                               MeanDataType,
                                               InvStdDataType>(
                a_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);
        }

        y_buf.FromDevice(y_host_dev.data());

        ck_tile::HostTensor<YResidualDataType> y_residual_host_dev({m, n}, {stride, 1});
        if(fused_add == 1)
        {
            y_residual_buf.FromDevice(y_residual_host_dev.data());
        }

        auto [rtol, atol] = get_elimit<InDataType>();

        if(stride == n)
        {
            pass = ck_tile::check_err(
                y_host_dev, y_host_ref, std::string("OUT Error: Incorrect results!"), rtol, atol);
            if(fused_add == 1)
            {
                pass &= ck_tile::check_err(y_residual_host_dev,
                                           a_host,
                                           std::string("ADD Error: Incorrect results!"),
                                           rtol,
                                           atol);
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
                    std::vector<YResidualDataType> y_residual_host_dev_row(
                        y_residual_host_dev.begin() + i_r * stride,
                        y_residual_host_dev.begin() + i_r * stride + n);
                    std::vector<YResidualDataType> y_residual_host_ref_row(
                        a_host.begin() + i_r * stride, a_host.begin() + i_r * stride + n);
                    pass &= ck_tile::check_err(y_residual_host_dev_row,
                                               y_residual_host_ref_row,
                                               std::string("ADD[") + std::to_string(i_r) +
                                                   std::string("] Error: Incorrect results!"),
                                               rtol,
                                               atol);
                }
            }
        }
        if(fused_quant == 1)
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
    int save_mv = arg_parser.get_int("save_mv");

    // no dynamic quant case
    if(prec_i == "fp16" && prec_o == "fp16" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::half_t, ck_tile::half_t, float, float, true>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "fp16" && prec_o == "fp16" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::half_t, ck_tile::half_t, float, float, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, float, float, true>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "bf16" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, float, float, true>(arg_parser) ? 0 : -2;
    }

    // dynamic quant case, only in inference
    else if(prec_i == "fp16" && prec_o == "int8" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::half_t, ck_tile::int8_t, float, float, false>(arg_parser) ? 0 : -2;
    }
    else if(prec_i == "bf16" && prec_o == "int8" && prec_sx == "fp32" && prec_sy == "fp32")
    {
        return run<ck_tile::bf16_t, ck_tile::int8_t, float, float, false>(arg_parser) ? 0 : -2;
    }

    return -3;
}
