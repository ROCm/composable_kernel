// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
using CDataType        = ck::half_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

inline __host__ __device__ ck::half2_t
type_convert_packed_i4_to_half2(ck::pk_i4_t x)
{
    uint8_t x_u8 = ck::bit_cast<uint8_t>(x);
    uint8_t x_l  = (x_u8 & 0x0f);
    uint8_t x_h  = (x_u8 & 0xf0) >> 4;

    auto l_f16 = ck::type_convert<ck::half_t>(x_l);
    auto h_f16 = ck::type_convert<ck::half_t>(x_h);

    return {l_f16, h_f16};
}


struct ElementwisePackedI4ToHalf2
{
	__host__ __device__ void
	operator()(ck::half2_t& y, const ck::pk_i4_t& x) const
    {
        y = type_convert_packed_i4_to_half2(x);
    }

	constexpr const static bool is_pack2_invocable = true;
};

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType, BDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
#if 0
        64,
        16, 16, 
        256, 8, 32,
        16,   16,
        1,    1, 
        S<32, 2, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8,  8, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 32, 32, 0,
        1, 1, S<1, 16, 1, 4>, 4,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v2>;
#else
        128,
        16, 32, 
        128, 8, 32,
        16,   16,
        1,    1, 
        S<16, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<4, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 32, 32, 0,
        1, 1, S<1, 16, 1, 8>, 4,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1>;

#endif
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AccDataType,
                                                                        PassThrough,
                                                                        PassThrough,
                                                                        PassThrough>;

#include "run_gemm_example_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
