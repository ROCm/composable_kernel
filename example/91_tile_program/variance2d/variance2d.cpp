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

#include "reference/reference_variance2d.hpp"
#include "variance2d.hpp"

int main(int argc, char* argv[])
{
    using XDataType       = ck::half_t;
    using ComputeDataType = float;
    using MeanDataType    = ck::half_t;
    using VarDataType     = ck::half_t;

    ck::index_t M = 3328;
    ck::index_t N = 4096;

    if(argc == 3)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
    }

    // host verify
    Tensor<XDataType> x_host({M, N});
    Tensor<MeanDataType> mean_host_ref({M});
    Tensor<MeanDataType> mean_host_dev({M});
    Tensor<VarDataType> var_host_ref({M});
    Tensor<VarDataType> var_host_dev({M});

    ck::utils::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);

    DeviceMem x_buf(x_host.GetElementSpaceSizeInBytes());
    DeviceMem mean_buf(mean_host_ref.GetElementSpaceSizeInBytes());
    DeviceMem var_buf(var_host_ref.GetElementSpaceSizeInBytes());

    x_buf.ToDevice(x_host.data());

    constexpr ck::index_t kMPerBlock = 128;
    constexpr ck::index_t kNPerBlock = 128;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M / kMPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    const auto kernel = Variance2d<XDataType,
                                   ComputeDataType,
                                   MeanDataType,
                                   VarDataType,
                                   kBlockSize,
                                   kMPerBlock,
                                   kNPerBlock>{};

    float ave_time = launch_kernel(StreamConfig{nullptr, true},
                                   kernel,
                                   kGridSize,
                                   kBlockSize,
                                   0,
                                   static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                   static_cast<MeanDataType*>(mean_buf.GetDeviceBuffer()),
                                   static_cast<VarDataType*>(var_buf.GetDeviceBuffer()),
                                   M,
                                   N);

    mean_buf.FromDevice(mean_host_dev.data());
    var_buf.FromDevice(var_host_dev.data());

    std::size_t num_byte = sizeof(XDataType) * M * N + sizeof(MeanDataType) * M;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    // reference
    reference_variance<XDataType, ComputeDataType, MeanDataType, VarDataType>(
        x_host, mean_host_ref, var_host_ref);

    bool pass = ck::utils::check_err(mean_host_dev, mean_host_ref) &&
                ck::utils::check_err(var_host_dev, var_host_dev);

    return !pass;
}
