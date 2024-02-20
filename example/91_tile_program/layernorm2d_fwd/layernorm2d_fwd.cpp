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
    using TypeConfig = Layernorm2dFwdTypeConfig<ck::half_t>;

    using XDataType       = TypeConfig::XDataType;
    using YDataType       = TypeConfig::YDataType;
    using GammaDataType   = TypeConfig::GammaDataType;
    using BetaDataType    = TypeConfig::BetaDataType;
    using MeanDataType    = TypeConfig::MeanDataType;
    using InvStdDataType  = TypeConfig::InvStdDataType;
    using ComputeDataType = TypeConfig::ComputeDataType;

    const bool SaveMeanVariance = true;

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
    Tensor<MeanDataType> mean_host_dev({M});
    Tensor<InvStdDataType> invStd_host_ref({M});
    Tensor<InvStdDataType> invStd_host_dev({M});

    ComputeDataType epsilon = 1e-5;

    ck::utils::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    ck::utils::FillUniformDistribution<GammaDataType>{-5.f, 5.f}(gamma_host);
    ck::utils::FillUniformDistribution<BetaDataType>{-5.f, 5.f}(beta_host);

    // reference
    reference_layernorm2d_fwd<XDataType,
                              GammaDataType,
                              BetaDataType,
                              ComputeDataType,
                              YDataType,
                              MeanDataType,
                              InvStdDataType>(
        x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);

    DeviceMem x_buf(x_host.GetElementSpaceSizeInBytes());
    DeviceMem gamma_buf(gamma_host.GetElementSpaceSizeInBytes());
    DeviceMem beta_buf(beta_host.GetElementSpaceSizeInBytes());
    DeviceMem y_buf(y_host_dev.GetElementSpaceSizeInBytes());
    DeviceMem mean_buf(mean_host_dev.GetElementSpaceSizeInBytes());
    DeviceMem invStd_buf(invStd_host_dev.GetElementSpaceSizeInBytes());

    x_buf.ToDevice(x_host.mData.data());
    gamma_buf.ToDevice(gamma_host.mData.data());
    beta_buf.ToDevice(beta_host.mData.data());

    constexpr ck::index_t kMPerBlock = 128;
    constexpr ck::index_t kNPerBlock = 128;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M / kMPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    const auto kernel = Layernorm2dFwd<XDataType,
                                       GammaDataType,
                                       BetaDataType,
                                       ComputeDataType,
                                       YDataType,
                                       MeanDataType,
                                       InvStdDataType,
                                       true,
                                       true,
                                       SaveMeanVariance,
                                       kBlockSize,
                                       kMPerBlock,
                                       kNPerBlock>{};

    float ave_time = launch_kernel(StreamConfig{nullptr, true},
                                   kernel,
                                   kGridSize,
                                   kBlockSize,
                                   0,
                                   static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                   static_cast<GammaDataType*>(gamma_buf.GetDeviceBuffer()),
                                   static_cast<BetaDataType*>(beta_buf.GetDeviceBuffer()),
                                   static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                   static_cast<MeanDataType*>(mean_buf.GetDeviceBuffer()),
                                   static_cast<InvStdDataType*>(invStd_buf.GetDeviceBuffer()),
                                   epsilon,
                                   M,
                                   N);

    if constexpr(SaveMeanVariance)
    {
        mean_buf.FromDevice(mean_host_dev.mData.data());
        invStd_buf.FromDevice(invStd_host_dev.mData.data());
    }

    y_buf.FromDevice(y_host_dev.mData.data());

    std::size_t num_byte = sizeof(XDataType) * M * N + sizeof(GammaDataType) * N +
                           sizeof(BetaDataType) * N + sizeof(YDataType) * M * N;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = ck::utils::check_err(y_host_dev, y_host_ref);

    return !pass;
}
