#include "ck_tile/host.hpp"
#include "layernorm2d_fwd.hpp"
#include <cstring>

extern float layernorm2d_fwd_fp16(layernorm2d_fwd_args& param, ck_tile::stream_config stream);
extern float layernorm2d_fwd_fp32(layernorm2d_fwd_args& param, ck_tile::stream_config stream);

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "m dimension")
        .insert("e", "1e-5", "epsilon")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp32", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{

    float epsilon         = arg_parser.get_float("e");
    ck_tile::index_t M    = arg_parser.get_int("m");
    ck_tile::index_t N    = arg_parser.get_int("n");
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    using TypeConfig = LayerNormTypeConfig<DataType>;

    using XDataType     = typename TypeConfig::XDataType;
    using YDataType     = typename TypeConfig::YDataType;
    using GammaDataType = typename TypeConfig::GammaDataType;
    using BetaDataType  = typename TypeConfig::BetaDataType;

    using MeanDataType   = ck_tile::null_type;
    using InvStdDataType = ck_tile::null_type;

    using ComputeDataType = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({M, N});
    ck_tile::HostTensor<GammaDataType> gamma_host({N});
    ck_tile::HostTensor<BetaDataType> beta_host({N});

    ck_tile::HostTensor<YDataType> y_host_ref({M, N});
    ck_tile::HostTensor<YDataType> y_host_dev({M, N});

    ck_tile::HostTensor<MeanDataType> mean_host_ref({M});
    ck_tile::HostTensor<InvStdDataType> invStd_host_ref({M});

// TODO - move SAVE_MEAN_INV_STD to user args
#ifdef SAVE_MEAN_INV_STD
    ck_tile::HostTensor<MeanDataType> mean_host_dev({M});
    ck_tile::HostTensor<InvStdDataType> invStd_host_dev({M});
#endif

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-.5f, .5f}(gamma_host);
    ck_tile::FillUniformDistribution<BetaDataType>{-.5f, .5f}(beta_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
#ifdef SAVE_MEAN_INV_STD
    ck_tile::DeviceMem mean_buf(mean_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem invStd_buf(invStd_host_dev.get_element_space_size_in_bytes());
#endif

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());

    layernorm2d_fwd_traits traits{data_type};

    layernorm2d_fwd_args args{x_buf.GetDeviceBuffer(),
                              gamma_buf.GetDeviceBuffer(),
                              beta_buf.GetDeviceBuffer(),
                              y_buf.GetDeviceBuffer(),
#ifdef SAVE_MEAN_INV_STD
                              mean_buf.GetDeviceBuffer(),
                              invStd_buf.GetDeviceBuffer(),
#else
                              nullptr,
                              nullptr,
#endif
                              epsilon,
                              M,
                              N};

    float ave_time = .0;

    if constexpr(std::is_same<DataType, ck_tile::fp16_t>::value)
    {
        ave_time =
            layernorm2d_fwd_fp16(args, ck_tile::stream_config{nullptr, true, 0, warmup, repeat});
    }
    else if constexpr(std::is_same<DataType, float>::value)
    {
        ave_time =
            layernorm2d_fwd_fp32(args, ck_tile::stream_config{nullptr, true, 0, warmup, repeat});
    }

    std::size_t num_byte = sizeof(XDataType) * M * N + sizeof(GammaDataType) * N +
                           sizeof(BetaDataType) * N + sizeof(YDataType) * M * N;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << "[" << data_type << "]"
              << " m:" << M << ", n:" << N << ", " << ave_time * 1.E6 << " ns, " << gb_per_sec
              << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_layernorm2d_fwd<XDataType,
                                           GammaDataType,
                                           BetaDataType,
                                           ComputeDataType,
                                           YDataType,
                                           MeanDataType,
                                           InvStdDataType>(
            x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);

        y_buf.FromDevice(y_host_dev.data());

        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush;
    }

    std::cout << std::endl << std::flush;
    std::cout << "pass = " << pass << std::endl;

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
    if(data_type == "fp32")
    {
        return run<float>(arg_parser) ? 0 : -2;
    }

    return -3;
}
