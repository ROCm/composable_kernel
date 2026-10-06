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

template <typename PrecType,
          typename CDataType,
          ck_tile::index_t VectorSizeA,
          ck_tile::index_t VectorSizeB,
          ck_tile::index_t VectorSizeC>
using PaddingConfig = GemmConfigVectorSizeFallback<GemmConfigComputeV3_WMMA<PrecType>,
                                                   VectorSizeA,
                                                   VectorSizeB,
                                                   VectorSizeC>;

template <typename Tuple>
class TestGemmUnalignedMNK : public ::testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using AccDataType = std::tuple_element_t<2, Tuple>;
    using CDataType   = std::tuple_element_t<3, Tuple>;

    template <typename ALayout, typename BLayout, typename CLayout, typename Config>
    void RunAndVerify(int M, int N, int K, int k_batch = 1)
    {
        const ck_tile::index_t stride_A = is_row_major(ALayout{}).value ? K : M;
        const ck_tile::index_t stride_B = is_row_major(BLayout{}).value ? N : K;
        const ck_tile::index_t stride_C = is_row_major(CLayout{}).value ? N : M;

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        ck_tile::HostTensor<CDataType> gpu_result(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

        ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
        ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);

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

        UniversalInvoker::gemm<Config,
                               ADataType,
                               BDataType,
                               ck_tile::tuple<>,
                               AccDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               ck_tile::tuple<>,
                               CLayout,
                               /*Persistent=*/false,
                               ck_tile::element_wise::PassThrough>(
            args, ck_tile::stream_config{nullptr, false});

        c_buf.FromDevice(gpu_result.data());

        ck_tile::HostTensor<CDataType> host_reference(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        host_reference.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, host_reference);

        const float max_accumulated_value =
            *std::max_element(host_reference.mData.begin(), host_reference.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, k_batch, max_accumulated_value);

        EXPECT_TRUE(do_verify(gpu_result, host_reference, rtol_atol, "GPU"))
            << "M=" << M << ", N=" << N << ", K=" << K << ", k_batch=" << k_batch;
    }
};

using UnalignedMNKDataTypes =
    ::testing::Types<std::tuple<ck_tile::half_t, ck_tile::half_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::bf16_t, ck_tile::bf16_t, float, ck_tile::bf16_t>,
// fp8/bf8 WMMA compute requires gfx12/gfx950
#ifdef CK_USE_WMMA_FP8
                     std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::half_t>,
                     std::tuple<ck_tile::bf8_t, ck_tile::bf8_t, float, ck_tile::half_t>,
#endif
                     std::tuple<ck_tile::int8_t, ck_tile::int8_t, int32_t, int32_t>>;
TYPED_TEST_SUITE(TestGemmUnalignedMNK, UnalignedMNKDataTypes);

// A=Row (packs K), B=Col (packs K), C=Row (packs N): unaligned K exercises A/B's fallback.
TYPED_TEST(TestGemmUnalignedMNK, UnalignedK)
{
    using ADataType = typename TestFixture::ADataType;
    using CDataType = typename TestFixture::CDataType;
    using Config    = PaddingConfig<ADataType, CDataType, 1, 1, 16 / sizeof(CDataType)>;

    constexpr int KTile = GemmConfigComputeV3_WMMA<ADataType>::K_Tile;
    for(int K : {5, KTile + 5})
    {
        this->template RunAndVerify<Row, Col, Row, Config>(/*M=*/128, /*N=*/128, K);
    }
}

// A=Col (packs M), B=Col (packs K), C=Row (packs N): unaligned M exercises A's fallback.
TYPED_TEST(TestGemmUnalignedMNK, UnalignedM)
{
    using ADataType = typename TestFixture::ADataType;
    using CDataType = typename TestFixture::CDataType;
    using Config =
        PaddingConfig<ADataType, CDataType, 1, 16 / sizeof(ADataType), 16 / sizeof(CDataType)>;

    constexpr int MTile = GemmConfigComputeV3_WMMA<ADataType>::M_Tile;
    for(int M : {5, MTile + 5})
    {
        this->template RunAndVerify<Col, Col, Row, Config>(M, /*N=*/128, /*K=*/128);
    }
}

// A=Row (packs K), B=Row (packs N), C=Row (packs N): unaligned N exercises B/C's fallback.
TYPED_TEST(TestGemmUnalignedMNK, UnalignedN)
{
    using ADataType = typename TestFixture::ADataType;
    using CDataType = typename TestFixture::CDataType;
    using Config    = PaddingConfig<ADataType, CDataType, 16 / sizeof(ADataType), 1, 1>;

    constexpr int NTile = GemmConfigComputeV3_WMMA<ADataType>::N_Tile;
    for(int N : {5, NTile + 5})
    {
        this->template RunAndVerify<Row, Row, Row, Config>(/*M=*/128, N, /*K=*/128);
    }
}
