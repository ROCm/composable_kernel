// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm TensorQuant ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_tensor_quant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, QDataType, AccDataType, QuantGroupSize.
 *
 * Direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly; no dispatcher registry is used (QuantGemmHostArgs is
 * incompatible with the registry backend's GeneratedTileKernelInstance::run()).
 *
 * TensorQuant semantics (matches Old-TE gemm_quant_tensor.cpp):
 *   C[M,N] = (aq_scalar * bq_scalar) * (A[M,K] @ B[K,N])
 * aq_ptr and bq_ptr each point at exactly ONE float; QK_A=QK_B=1 and
 * stride_AQ=stride_BQ=1.
 *
 * TensorQuant neither reshuffles its operands nor has a quant group size, so the
 * whole run() body is quant_bridge::run_scalar_quant_gemm() -- shared verbatim
 * with rowcolquant, which differs only in its scale-buffer extents. This file
 * therefore holds just the exported entry point and those extents.
 *
 * Memory model: host-pointer (this library owns hipMalloc/hipMemcpy/hipFree).
 */

#include <hip/hip_runtime.h>
#include <cstdint>

#include "quant_bridge_common.hpp"

extern "C" {

QUANT_BRIDGE_C_API()

/**
 * Run TensorQuant GEMM: C[M,N] = (AQ * BQ) * (A[M,K] @ B[K,N]) with AQ, BQ
 * single per-tensor float scales. A (row-major [M,K]), B (col-major [K,N]),
 * AQ/BQ (one float each), C (row-major [M,N]) are host pointers; device memory
 * is managed internally. time_ms is an optional output. Returns 0 on success.
 */
int dispatcher_run_tensor_quant_gemm(const void* A,
                                     const void* B,
                                     const void* AQ,
                                     const void* BQ,
                                     void* C,
                                     int64_t M,
                                     int64_t N,
                                     int64_t K,
                                     int64_t stride_A,
                                     int64_t stride_B,
                                     int64_t stride_C,
                                     int k_batch,
                                     float* time_ms)
{
    // TensorQuant reads *aq_ptr / *bq_ptr as single scalar scales, so each scale
    // buffer holds exactly one element. Everything else -- guards, packed-stride
    // contract, device buffers, args fill, launch, copy-back -- is the shared
    // scalar-quant body, identical to rowcolquant's.
    return quant_bridge::
        run_scalar_quant_gemm<SelectedKernel, ADataType, BDataType, CDataType, QDataType>(
            "dispatcher_run_tensor_quant_gemm",
            g_initialized,
            A,
            B,
            AQ,
            BQ,
            C,
            M,
            N,
            K,
            stride_A,
            stride_B,
            stride_C,
            /*aq_elems=*/1,
            /*bq_elems=*/1,
            k_batch,
            time_ms);
}

} // extern "C"
