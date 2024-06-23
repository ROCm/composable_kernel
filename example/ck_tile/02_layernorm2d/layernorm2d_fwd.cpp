#include "ck_tile/host.hpp"
#include "layernorm2d_fwd.hpp"
#include <cstring>

// Host API implementation
float layernorm2d_fwd(layernorm2d_fwd_traits t,
                      layernorm2d_fwd_args a,
                      const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16") == 0)
    {
        using XDataType     = ck_tile::half_t;
        using YDataType     = ck_tile::half_t;
        using GammaDataType = ck_tile::half_t;
        using BetaDataType  = ck_tile::half_t;
#ifdef SAVE_MEAN_INV_STD
        using MeanDataType   = ck_tile::half_t;
        using InvStdDataType = ck_tile::half_t;
#else
        using MeanDataType   = ck_tile::null_type;
        using InvStdDataType = ck_tile::null_type;
#endif
        using ComputeDataType = float;

        using thread_tile = ck_tile::sequence<4, 4>;
        using warp_tile   = ck_tile::sequence<8, 128>;
        using block_tile  = ck_tile::sequence<32, 128>;

        using Shape = ck_tile::TileLayernorm2dShape<thread_tile, warp_tile, block_tile>;

        using PipelineProblem = ck_tile::BlockLayernorm2dFwdProblem<XDataType,
                                                                    GammaDataType,
                                                                    BetaDataType,
                                                                    ComputeDataType,
                                                                    YDataType,
                                                                    MeanDataType,
                                                                    InvStdDataType,
                                                                    Shape>;

        using Kernel = ck_tile::Layernorm2dFwd<PipelineProblem>;

        auto kargs = Kernel::MakeKargs(
            a.p_x, a.p_gamma, a.p_beta, a.p_y, a.p_mean, a.p_invStd, a.epsilon, a.M, a.N);

        const dim3 grids      = Kernel::GridSize(a.M);
        constexpr dim3 blocks = Kernel::BlockSize();

        constexpr ck_tile::index_t kBlockPerCu = Shape::kMWarpPerBlock * Shape::kNWarpPerBlock;

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    }

    return 0;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "m dimension")
        .insert("e", "1e-5", "epsilon")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

int main(int argc, char* argv[])
{

    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    float epsilon         = arg_parser.get_float("e");
    ck_tile::index_t M    = arg_parser.get_int("m");
    ck_tile::index_t N    = arg_parser.get_int("n");
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");

    using XDataType     = ck_tile::half_t;
    using YDataType     = ck_tile::half_t;
    using GammaDataType = ck_tile::half_t;
    using BetaDataType  = ck_tile::half_t;
#ifdef SAVE_MEAN_INV_STD
    using MeanDataType   = ck_tile::half_t;
    using InvStdDataType = ck_tile::half_t;
#else
    using MeanDataType = ck_tile::null_type;
    using InvStdDataType = ck_tile::null_type;
#endif
    using ComputeDataType = float;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({M, N});
    ck_tile::HostTensor<GammaDataType> gamma_host({N});
    ck_tile::HostTensor<BetaDataType> beta_host({N});

    ck_tile::HostTensor<YDataType> y_host_ref({M, N});
    ck_tile::HostTensor<YDataType> y_host_dev({M, N});

    ck_tile::HostTensor<MeanDataType> mean_host_ref({M});
    ck_tile::HostTensor<InvStdDataType> invStd_host_ref({M});

#ifdef SAVE_MEAN_INV_STD
    ck_tile::HostTensor<MeanDataType> mean_host_dev({M});
    ck_tile::HostTensor<InvStdDataType> invStd_host_dev({M});
#endif

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    ck_tile::FillUniformDistribution<GammaDataType>{-5.f, 5.f}(gamma_host);
    ck_tile::FillUniformDistribution<BetaDataType>{-5.f, 5.f}(beta_host);

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

    float ave_time = layernorm2d_fwd(traits, args, ck_tile::stream_config{nullptr, true});

    std::size_t num_byte = sizeof(XDataType) * M * N + sizeof(GammaDataType) * N +
                           sizeof(BetaDataType) * N + sizeof(YDataType) * M * N;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << "[" << data_type << "]"
              << " m:" << M << ", n:" << N << ", " << ave_time << " ms, " << gb_per_sec << " GB/s"
              << std::flush;

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

#ifdef SAVE_MEAN_INV_STD
        mean_buf.FromDevice(mean_host_dev.data());
        pass &= ck_tile::check_err(mean_host_dev, mean_host_ref);

        invStd_buf.FromDevice(invStd_host_dev.data());
        pass &= ck_tile::check_err(invStd_host_dev, invStd_host_ref);
#endif

        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush;
    }

    std::cout << std::endl << std::flush;

    return !pass;
}
