// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

extern "C" __device__ float __ocml_native_recip_f32(float);

using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F16;
using D1DataType       = F16;
using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
using EDataType        = F16;

using ALayout  = Row;
using BLayout  = Col;
using D0Layout = Row;
using D1Layout = Row;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout  = Row;

// C = A * B
// E = FastGelu(C + D0 + D1)
struct EleFastGeLU
{
    // Fast GeLU
    // https://paperswithcode.com/method/gelu
    // y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    __host__ static constexpr float GetFastGeLU(float x)
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = exp(-u);
        const float cdf = 0.5f + 0.5f * (2.f / (1.f + emu) - 1.f);
        return x * cdf;
    }

#if 0
     __device__ static constexpr float GetFastGeLU(float x)
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = __expf(-u);
        const float cdf = 0.5f + 0.5f * (2.f / (1.f + emu) - 1.f);
        return x * cdf;
    }
#elif 0
    __device__ static constexpr float GetFastGeLU(float x)
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = __expf(-u);
        const float cdf = 0.5f + 0.5f * (2.f * __frcp_rn(1.f + emu) - 1.f);
        return x * cdf;
    }
#else
    __device__ static constexpr float GetFastGeLU(float x)
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = __expf(-u);
        const float cdf = 0.5f + 0.5f * (2.f * __ocml_native_recip_f32(1.f + emu) - 1.f);
        return x * cdf;
    }
#endif

    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const
    {
#if 0
        const float y =
            GetFastGeLU(ck::type_convert<float>(c) + ck::type_convert<float>(d0) + ck::type_convert<float>(d1));
#else
        const float a =
            ck::type_convert<float>(c) + ck::type_convert<float>(d0) + ck::type_convert<float>(d1);
        const float y = a > 0 ? a : 0;
#endif

        e = ck::type_convert<E>(y);
    }
};

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = EleFastGeLU;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle
//######| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
//######|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
//######|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
//######|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        AccDataType,
                                                                        AccDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        PassThrough>;

#include "run_gemm_add_add_fastgelu_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_add_add_fastgelu_example(argc, argv); }
