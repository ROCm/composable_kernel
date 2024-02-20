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

#include "layernorm2d_fwd.hpp"

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename MeanDataType,
          typename InvStdDataType>
void reference_layernorm2d_fwd(const Tensor<XDataType>& x_m_n,
                               const Tensor<GammaDataType>& gamma_n,
                               const Tensor<BetaDataType>& beta_n,
                               Tensor<YDataType>& y_m_n,
                               Tensor<MeanDataType>& mean_m,
                               Tensor<InvStdDataType>& invStd_m,
                               ComputeDataType epsilon)
{
    auto layernorm2d_fwd_func = [&](auto m) {
        const int N = x_m_n.mDesc.GetLengths()[1];

        int count                = 0;
        ComputeDataType mean     = 0;
        ComputeDataType variance = 0;
        ComputeDataType divisor  = 0;

        for(int n = 0; n < N; ++n)
        {
            ++count;
            ComputeDataType x     = ck::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType delta = x - mean;
            mean += delta / count;
            ComputeDataType delta2 = x - mean;
            variance += delta * delta2;
        }

        // actual variance
        variance = variance / count;
        divisor  = ck::type_convert<ComputeDataType>(1) / ck::math::sqrt(variance + epsilon);

        mean_m(m)   = ck::type_convert<MeanDataType>(mean);
        invStd_m(m) = ck::type_convert<InvStdDataType>(divisor);

        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x     = ck::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType gamma = ck::type_convert<ComputeDataType>(gamma_n(n));
            ComputeDataType beta  = ck::type_convert<ComputeDataType>(beta_n(n));
            auto y                = (x - mean) * divisor;
            y                     = y * gamma + beta;

            y_m_n(m, n) = ck::type_convert<YDataType>(y);
        }
    };

    make_ParallelTensorFunctor(layernorm2d_fwd_func,
                               mean_m.mDesc.GetLengths()[0])(std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    using XDataType       = ck::half_t;
    using GammaDataType   = ck::half_t;
    using BetaDataType    = ck::half_t;
    using ComputeDataType = float;
    using YDataType       = ck::half_t;
    using MeanDataType    = ck::half_t;
    using InvStdDataType  = ck::half_t;

    const bool SaveMeanVariance = true;

    ck::index_t M = 3328;
    ck::index_t N = 4096;

    if(argc == 3)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
    }

    std::array<ck::index_t, 2> x_lengths{M, N};
    std::array<ck::index_t, 2> x_strides{N, 1};

    std::array<ck::index_t, 1> gamma_lengths{N};
    std::array<ck::index_t, 1> gamma_strides{1};

    std::array<ck::index_t, 1> beta_lengths{N};
    std::array<ck::index_t, 1> beta_strides{1};

    std::array<ck::index_t, 2> y_lengths{M, N};
    std::array<ck::index_t, 2> y_strides{N, 1};

    std::array<ck::index_t, 1> mean_lengths{M};
    std::array<ck::index_t, 1> mean_strides{1};

    std::array<ck::index_t, 1> invStd_lengths{M};
    std::array<ck::index_t, 1> invStd_strides{1};

    // host verify
    Tensor<XDataType> x_host(x_lengths, x_strides);
    Tensor<GammaDataType> gamma_host(gamma_lengths, gamma_strides);
    Tensor<BetaDataType> beta_host(beta_lengths, beta_strides);

    Tensor<YDataType> y_host_ref(y_lengths, y_strides);
    Tensor<YDataType> y_host_dev(y_lengths, y_strides);
    Tensor<MeanDataType> mean_host_ref(mean_lengths, mean_strides);
    Tensor<MeanDataType> mean_host_dev(mean_lengths, mean_strides);
    Tensor<InvStdDataType> invStd_host_ref(invStd_lengths, invStd_strides);
    Tensor<InvStdDataType> invStd_host_dev(invStd_lengths, invStd_strides);

    ComputeDataType epsilon = 1e-5;

    ck::utils::FillUniformDistributionIntegerValue<XDataType>{-5.f, 5.f}(x_host);
    ck::utils::FillUniformDistributionIntegerValue<GammaDataType>{-5.f, 5.f}(gamma_host);
    ck::utils::FillUniformDistributionIntegerValue<BetaDataType>{-5.f, 5.f}(beta_host);

    // reference
    reference_layernorm2d_fwd<XDataType,
                              GammaDataType,
                              BetaDataType,
                              ComputeDataType,
                              YDataType,
                              MeanDataType,
                              InvStdDataType>(
        x_host, gamma_host, beta_host, y_host_ref, mean_host_ref, invStd_host_ref, epsilon);

    DeviceMem x_buf(sizeof(XDataType) * x_host.GetElementSpaceSize());
    DeviceMem gamma_buf(sizeof(GammaDataType) * gamma_host.GetElementSpaceSize());
    DeviceMem beta_buf(sizeof(BetaDataType) * beta_host.GetElementSpaceSize());
    DeviceMem y_buf(sizeof(YDataType) * y_host_dev.GetElementSpaceSize());
    DeviceMem mean_buf(sizeof(MeanDataType) * mean_host_dev.GetElementSpaceSize());
    DeviceMem invStd_buf(sizeof(InvStdDataType) * invStd_host_dev.GetElementSpaceSize());

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

    std::size_t num_btype = sizeof(XDataType) * M * N + sizeof(GammaDataType) * N +
                            sizeof(BetaDataType) * N + sizeof(YDataType) * M * N;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = ck::utils::check_err(y_host_dev, y_host_ref);

    return !pass;
}
