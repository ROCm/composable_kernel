#include <cstring>

#include "ck_tile/host.hpp"
#include "layernorm2d_fwd.hpp"

int main(int argc, char* argv[])
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

    float epsilon      = 1e-5;
    ck_tile::index_t M = 3328;
    ck_tile::index_t N = 4096;

    if(argc == 3)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
    }

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

    auto kargs = Kernel::MakeKargs(x_buf.GetDeviceBuffer(),
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
                                   N);

    constexpr ck_tile::index_t kBlockPerCu = Shape::kMWarpPerBlock * Shape::kNWarpPerBlock;

    float ave_time =
        ck_tile::launch_kernel(ck_tile::stream_config{nullptr, true},
                               ck_tile::make_kernel<Kernel::BlockSize(), kBlockPerCu>(
                                   Kernel{}, Kernel::GridSize(M), Kernel::BlockSize(), 0, kargs));

    std::size_t num_byte = sizeof(XDataType) * M * N + sizeof(GammaDataType) * N +
                           sizeof(BetaDataType) * N + sizeof(YDataType) * M * N;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

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

    bool pass = ck_tile::check_err(y_host_dev, y_host_ref);

#ifdef SAVE_MEAN_INV_STD
    if constexpr(!std::is_same_v<MeanDataType, ck_tile::null_type>)
    {
        mean_buf.FromDevice(mean_host_dev.data());
        pass &= ck_tile::check_err(mean_host_dev, mean_host_ref);
    }

    if constexpr(!std::is_same_v<InvStdDataType, ck_tile::null_type>)
    {
        invStd_buf.FromDevice(invStd_host_dev.data());
        pass &= ck_tile::check_err(invStd_host_dev, invStd_host_ref);
    }
#endif

    return !pass;
}
