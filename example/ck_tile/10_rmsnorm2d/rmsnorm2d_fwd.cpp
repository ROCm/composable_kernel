#include "ck_tile/host.hpp"
#include "rmsnorm2d_fwd.hpp"
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
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("e", "1e-5", "epsilon")
        .insert("save_rms", "0", "save rms(invrms) or not. set to 1 in training case")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType, bool SaveRms>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m      = arg_parser.get_int("m");
    ck_tile::index_t n      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = n;
    float epsilon         = arg_parser.get_float("e");
    std::string data_type = arg_parser.get_str("prec");
    int kname             = arg_parser.get_int("kname");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    assert(stride >= n);

    using TypeConfig = RmsnormTypeConfig<DataType>;

    using XDataType     = typename TypeConfig::XDataType;
    using YDataType     = typename TypeConfig::YDataType;
    using GammaDataType = typename TypeConfig::GammaDataType;

    using InvRmsDataType =
        std::conditional_t<SaveRms, typename TypeConfig::InvRmsDataType, ck_tile::null_type>;

    using ComputeDataType = typename TypeConfig::ComputeDataType;

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

    std::cout << "[" << data_type << "]"
              << " m:" << m << ", n:" << n << ", stride:" << stride << std::flush;

    rmsnorm2d_fwd_traits traits{data_type, SaveRms};

    rmsnorm2d_fwd_args args{x_buf.GetDeviceBuffer(),
                            gamma_buf.GetDeviceBuffer(),
                            y_buf.GetDeviceBuffer(),
                            nullptr,
                            epsilon,
                            m,
                            n,
                            stride};

    float ave_time = rmsnorm2d_fwd(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    std::size_t num_byte =
        sizeof(XDataType) * m * n + sizeof(GammaDataType) * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

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

        auto [rtol, atol] = get_elimit<DataType>();
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

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    int save_rms                = arg_parser.get_int("save_rms");
    if(data_type == "fp16" && save_rms)
    {
        return run<ck_tile::half_t, true>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "fp16" && !save_rms)
    {
        return run<ck_tile::half_t, false>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16" && save_rms)
    {
        return run<ck_tile::bf16_t, true>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16" && !save_rms)
    {
        return run<ck_tile::bf16_t, true>(arg_parser) ? 0 : -2;
    }

    return -3;
}
