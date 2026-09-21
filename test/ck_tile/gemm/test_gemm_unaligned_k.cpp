// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gemm_utils.hpp"
#include "run_gemm_example.inc"
#include "universal_gemm_invoker.hpp"

#include "ck_tile/host.hpp"

#include "gtest/gtest.h"

#include <tuple>

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

template <typename PrecType, typename CDataType>
using KPaddingConfig =
    GemmConfigFixedVectorSize<GemmConfigComputeV3_WMMA<PrecType>, 1, 1, kMaxVectorElems<CDataType>>;

template <typename Tuple>
class TestGemmUnalignedK : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using CDataType   = std::tuple_element_t<3, Tuple>;

    static constexpr bool check_data_type()
    {
#if defined(ARCH_GFX12)
#if defined(CK_USE_GFX1250)
        using DeviceIp = ck_tile::gfx125_t;
#else
        using DeviceIp = ck_tile::gfx120_t;
#endif
#elif defined(ARCH_GFX11)
        using DeviceIp = ck_tile::gfx11_t;
#else
#error "Unsupported architecture for WMMA"
#endif
        using Config = KPaddingConfig<ADataType, CDataType>;
        return ck_tile::has_wmma_traits_v<DeviceIp,
                                          ADataType,
                                          BDataType,
                                          AccDataType,
                                          Config::M_Warp_Tile,
                                          Config::N_Warp_Tile,
                                          Config::K_Warp_Tile>;
    }

    void RunAndVerify(int K, int k_batch = 1)
    {
        if constexpr(!check_data_type())
        {
            GTEST_SKIP() << "Unsupported data type combination for this architecture.";
        }
        else
        {
            constexpr int M = 128;
            constexpr int N = 128;

            const ck_tile::index_t stride_A = K;
            const ck_tile::index_t stride_B = K;
            const ck_tile::index_t stride_C = N;

            // Host tensors
            ck_tile::HostTensor<ADataType> a_m_k(
                ck_tile::host_tensor_descriptor(M, K, stride_A, ck_tile::bool_constant<true>{}));
            ck_tile::HostTensor<BDataType> b_k_n(
                ck_tile::host_tensor_descriptor(K, N, stride_B, ck_tile::bool_constant<false>{}));
            ck_tile::HostTensor<CDataType> gpu_result(
                ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<true>{}));

            ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
            ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);

            // Device buffers
            ck_tile::DeviceMem a_buf(a_m_k.get_element_space_size_in_bytes());
            ck_tile::DeviceMem b_buf(b_k_n.get_element_space_size_in_bytes());
            ck_tile::DeviceMem c_buf(gpu_result.get_element_space_size_in_bytes());
            a_buf.ToDevice(a_m_k.data());
            b_buf.ToDevice(b_k_n.data());
            c_buf.SetZero();

            ck_tile::GemmHostArgs args = {a_buf.GetDeviceBuffer(),
                                          b_buf.GetDeviceBuffer(),
                                          c_buf.GetDeviceBuffer(),
                                          k_batch,
                                          M,
                                          N,
                                          K,
                                          stride_A,
                                          stride_B,
                                          stride_C};

            // Run UniversalInvoker::gemm()
            UniversalInvoker::gemm<KPaddingConfig<ADataType, CDataType>,
                                   ADataType,
                                   BDataType,
                                   ck_tile::tuple<>,
                                   AccDataType,
                                   CDataType,
                                   Row,
                                   Col,
                                   ck_tile::tuple<>,
                                   Row,
                                   /*Persistent=*/false,
                                   ck_tile::element_wise::PassThrough>(
                args, ck_tile::stream_config{nullptr, false});

            c_buf.FromDevice(gpu_result.data());

            ck_tile::HostTensor<CDataType> host_reference(
                ck_tile::host_tensor_descriptor(M, N, stride_C, ck_tile::bool_constant<true>{}));
            host_reference.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k, b_k_n, host_reference);

            const float max_accumulated_value =
                *std::max_element(host_reference.mData.begin(), host_reference.mData.end());
            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                    K, k_batch, max_accumulated_value);

            // Compare both results
            EXPECT_TRUE(do_verify(gpu_result, host_reference, rtol_atol, "GPU"))
                << "K=" << K << ", k_batch=" << k_batch;
        }
    }
};

using UnalignedKDataTypes =
    ::testing::Types<std::tuple<ck_tile::half_t, ck_tile::half_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::bf16_t, ck_tile::bf16_t, float, ck_tile::bf16_t>,
                     std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::bf8_t, ck_tile::bf8_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::int8_t, ck_tile::int8_t, int32_t, int32_t>>;
TYPED_TEST_SUITE(TestGemmUnalignedK, UnalignedKDataTypes);

TYPED_TEST(TestGemmUnalignedK, SmallKAndTail)
{
    // Cover a K smaller than the maximum load width and a full K tile plus a tail.
    constexpr int KTile = GemmConfigComputeV3_WMMA<typename TestFixture::ADataType>::K_Tile;
    for(int K : {5, KTile + 5})
    {
        this->RunAndVerify(K);
    }
}

TYPED_TEST(TestGemmUnalignedK, SplitK)
{
    // The first split is a full warp tile; the second contains the K tail.
    constexpr int K = GemmConfigComputeV3_WMMA<typename TestFixture::ADataType>::K_Warp_Tile + 5;
    this->RunAndVerify(K, /*k_batch=*/2);
}
