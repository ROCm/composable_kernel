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

#include "reference_gemm.hpp"
#include "gemm.hpp"

// elementwise lambda
struct AElementFunction
{
    template <typename X>
    __host__ __device__ auto operator()(const X& x) const
    {
        return x;
    }
};

struct BElementFunction
{
    template <typename X>
    __host__ __device__ auto operator()(const X& x) const
    {
        return x;
    }
};

struct CElementFunction
{
    template <typename X>
    __host__ __device__ auto operator()(const X& x) const
    {
        return x;
    }
};

int main(int argc, char* argv[])
{
    using ADataType   = ck::half_t;
    using BDataType   = ck::half_t;
    using AccDataType = float;
    using CDataType   = ck::half_t;

    ck::index_t M = 3328;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    if(argc == 4)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    std::array<ck::index_t, 2> a_lengths{M, K};
    std::array<ck::index_t, 2> a_strides{K, 1};

    std::array<ck::index_t, 2> b_lengths{N, K};
    std::array<ck::index_t, 2> b_strides{K, 1};

    std::array<ck::index_t, 2> c_lengths{M, N};
    std::array<ck::index_t, 2> c_strides{N, 1};

    // host verify
    Tensor<ADataType> a_host(a_lengths, a_strides);
    Tensor<BDataType> b_host(b_lengths, b_strides);
    Tensor<CDataType> c_host_ref(c_lengths, c_strides);
    Tensor<CDataType> c_host_dev(c_lengths, c_strides);

    ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    // reference gemm
    reference_gemm<ADataType, ADataType, CDataType, float>(a_host, b_host, c_host_ref);

    DeviceMem a_buf(sizeof(ADataType) * a_host.GetElementSpaceSize());
    DeviceMem b_buf(sizeof(BDataType) * b_host.GetElementSpaceSize());
    DeviceMem c_buf(sizeof(CDataType) * c_host_dev.GetElementSpaceSize());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    constexpr ck::index_t kGemmMPerBlock = 256;
    constexpr ck::index_t kGemmNPerBlock = 128;
    constexpr ck::index_t kGemmKPerBlock = 32;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    const auto gemm_kernel = Gemm<ADataType,
                                  BDataType,
                                  AccDataType,
                                  CDataType,
                                  ck::tensor_layout::gemm::RowMajor,
                                  ck::tensor_layout::gemm::ColumnMajor,
                                  ck::tensor_layout::gemm::RowMajor,
                                  AElementFunction,
                                  BElementFunction,
                                  CElementFunction,
                                  kBlockSize,
                                  kGemmMPerBlock,
                                  kGemmNPerBlock,
                                  kGemmKPerBlock>{};

    float ave_time = launch_kernel<kBlockSize, 2>(StreamConfig{nullptr, true},
                                                  gemm_kernel,
                                                  kGridSize,
                                                  kBlockSize,
                                                  0,
                                                  static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                                                  static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                                                  static_cast<CDataType*>(c_buf.GetDeviceBuffer()),
                                                  M,
                                                  N,
                                                  K,
                                                  K,
                                                  K,
                                                  N,
                                                  AElementFunction{},
                                                  BElementFunction{},
                                                  CElementFunction{});

    c_buf.FromDevice(c_host_dev.mData.data());

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !ck::utils::check_err(c_host_dev, c_host_ref);
}
