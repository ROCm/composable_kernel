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

/*
 * Toy code of GEMM
 * Assume simplest case.
 * A [M, K]
 * B [N, K]
 * C [M, N]
 */

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

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    ck::index_t verification = 0;
    ck::index_t M            = 3328;
    ck::index_t N            = 4096;
    ck::index_t K            = 4096;

    if(argc == 2)
    {
        verification = std::stoi(argv[1]);
    }
    if(argc == 5)
    {
        verification = std::stoi(argv[1]);
        M            = std::stoi(argv[1]);
        N            = std::stoi(argv[2]);
        K            = std::stoi(argv[3]);
    }

    const ck::index_t Lda = std::is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor> ? K : M;
    const ck::index_t Ldb = std::is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor> ? K : N;
    const ck::index_t Ldc = std::is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor> ? N : M;

    const auto a_lengths = std::array<ck::index_t, 2>{M, K};
    const auto a_strides = std::is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor>
                               ? std::array<ck::index_t, 2>{Lda, 1}
                               : std::array<ck::index_t, 2>{1, Lda};

    const auto b_lengths = std::array<ck::index_t, 2>{N, K};
    const auto b_strides = std::is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor>
                               ? std::array<ck::index_t, 2>{Ldb, 1}
                               : std::array<ck::index_t, 2>{1, Ldb};

    const auto c_lengths = std::array<ck::index_t, 2>{M, N};
    const auto c_strides = std::is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor>
                               ? std::array<ck::index_t, 2>{Ldc, 1}
                               : std::array<ck::index_t, 2>{1, Ldc};

    // host verify
    Tensor<ADataType> a_host(a_lengths, a_strides);
    Tensor<BDataType> b_host(b_lengths, b_strides);
    Tensor<CDataType> c_host_dev(c_lengths, c_strides);

    ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    DeviceMem a_buf(sizeof(ADataType) * a_host.GetElementSpaceSize());
    DeviceMem b_buf(sizeof(BDataType) * b_host.GetElementSpaceSize());
    DeviceMem c_buf(sizeof(CDataType) * c_host_dev.GetElementSpaceSize());

    a_buf.ToDevice(a_host.mData.data());
    b_buf.ToDevice(b_host.mData.data());

    // Alignment
    constexpr ck::index_t kAAlignment = 32;
    constexpr ck::index_t kBAlignment = 32;
    constexpr ck::index_t kCAlignment = 32;

    constexpr ck::index_t kBlockSize = 256;

    constexpr ck::index_t kGemmMPerBlock = 256;
    constexpr ck::index_t kGemmNPerBlock = 128;
    constexpr ck::index_t kGemmKPerBlock = 32;

    ck::index_t kGridSize = (M / kGemmMPerBlock) * (N / kGemmNPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    const auto gemm_kernel = Gemm<ADataType,
                                  BDataType,
                                  AccDataType,
                                  CDataType,
                                  ALayout,
                                  BLayout,
                                  CLayout,
                                  AElementFunction,
                                  BElementFunction,
                                  CElementFunction,
                                  kAAlignment,
                                  kBAlignment,
                                  kCAlignment,
                                  kBlockSize,
                                  kGemmMPerBlock,
                                  kGemmNPerBlock,
                                  kGemmKPerBlock>{};

    float ave_time =
        launch_kernel<kBlockSize, kBlockPerCu>(StreamConfig{nullptr, true},
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
                                               Lda,
                                               Ldb,
                                               Ldc,
                                               AElementFunction{},
                                               BElementFunction{},
                                               CElementFunction{});
    auto pass = true;

    if(verification)
    {
        // reference gemm
        Tensor<CDataType> c_host_ref(c_lengths, c_strides);
        reference_gemm<ADataType, ADataType, AccDataType, CDataType>(a_host, b_host, c_host_ref);
        c_buf.FromDevice(c_host_dev.mData.data());
        pass &= ck::utils::check_err(c_host_dev, c_host_ref);
    }

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !pass;
}
