#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "reference/reference_layernorm2d.hpp"
#include "layernorm2d_fwd.hpp"

int main(int argc, char* argv[])
{
    using XDataType     = ck::half_t;
    using YDataType     = ck::half_t;
    using GammaDataType = ck::half_t;
    using BetaDataType  = ck::half_t;
#ifdef SAVE_MEAN_INV_STD
    using MeanDataType   = ck::half_t;
    using InvStdDataType = ck::half_t;
#else
    using MeanDataType   = ck::null_type;
    using InvStdDataType = ck::null_type;
#endif
    using ComputeDataType = float;

    using thread_tile = ck::Sequence<4, 4>;
    using warp_tile   = ck::Sequence<8, 128>;
    using block_tile  = ck::Sequence<32, 128>;

    using Shape = ck::tile_program::TileLayernorm2dShape<thread_tile, warp_tile, block_tile>;

    using PipelineProblem = ck::tile_program::block::BlockLayernorm2dFwdProblem<XDataType,
                                                                                GammaDataType,
                                                                                BetaDataType,
                                                                                ComputeDataType,
                                                                                YDataType,
                                                                                MeanDataType,
                                                                                InvStdDataType,
                                                                                Shape>;

    using Kernel = Layernorm2dFwd<PipelineProblem>;

    float epsilon = 1e-5;
    ck::index_t M = 3328;
    ck::index_t N = 4096;

    if(argc == 3)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
    }

    // host verify
    Tensor<XDataType> x_host({M, N});
    Tensor<GammaDataType> gamma_host({N});
    Tensor<BetaDataType> beta_host({N});

    Tensor<YDataType> y_host_ref({M, N});
    Tensor<YDataType> y_host_dev({M, N});

    Tensor<MeanDataType> mean_host_ref({M});
    Tensor<InvStdDataType> invStd_host_ref({M});

#ifdef SAVE_MEAN_INV_STD
    Tensor<MeanDataType> mean_host_dev({M});
    Tensor<InvStdDataType> invStd_host_dev({M});
#endif

    ck::utils::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    ck::utils::FillUniformDistribution<GammaDataType>{-5.f, 5.f}(gamma_host);
    ck::utils::FillUniformDistribution<BetaDataType>{-5.f, 5.f}(beta_host);

    DeviceMem x_buf(x_host.GetElementSpaceSizeInBytes());
    DeviceMem gamma_buf(gamma_host.GetElementSpaceSizeInBytes());
    DeviceMem beta_buf(beta_host.GetElementSpaceSizeInBytes());
    DeviceMem y_buf(y_host_dev.GetElementSpaceSizeInBytes());

#ifdef SAVE_MEAN_INV_STD
    DeviceMem mean_buf(mean_host_dev.GetElementSpaceSizeInBytes());
    DeviceMem invStd_buf(invStd_host_dev.GetElementSpaceSizeInBytes());
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

    float ave_time = launch_kernel(
        StreamConfig{nullptr, true}, Kernel{}, Kernel::GridSize(M), Kernel::BlockSize(), 0, kargs);

    std::size_t num_byte = sizeof(XDataType) * M * N + sizeof(GammaDataType) * N +
                           sizeof(BetaDataType) * N + sizeof(YDataType) * M * N;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    // reference
    reference_layernorm2d_fwd<XDataType,
                              GammaDataType,
                              BetaDataType,
                              ComputeDataType,
                              YDataType,
                              MeanDataType,
                              InvStdDataType>(
        x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);

    y_buf.FromDevice(y_host_dev.data());

    bool pass = ck::utils::check_err(y_host_dev, y_host_ref);

#ifdef SAVE_MEAN_INV_STD
    if constexpr(!ck::is_same_v<MeanDataType, ck::null_type>)
    {
        mean_buf.FromDevice(mean_host_dev.data());
        pass &= ck::utils::check_err(mean_host_dev, mean_host_ref);
    }

    if constexpr(!ck::is_same_v<InvStdDataType, ck::null_type>)
    {
        invStd_buf.FromDevice(invStd_host_dev.data());
        pass &= ck::utils::check_err(invStd_host_dev, invStd_host_ref);
    }
#endif

    return !pass;
}
