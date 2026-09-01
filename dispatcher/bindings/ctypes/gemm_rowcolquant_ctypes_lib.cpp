// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Gemm RowColQuant ctypes Library
 *
 * One .so per kernel variant; the kernel is force-included at compile time:
 *   hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_rowcolquant_ctypes_lib.cpp
 * Force-include defines: SelectedKernel, KERNEL_NAME, ADataType, BDataType,
 * CDataType, QDataType, AccDataType.
 *
 * Direct launch -- SelectedKernel::launch(QuantGemmHostArgs, stream_config) is
 * called directly; no dispatcher registry is used.
 *
 * RowColQuant = per-row scale of A (AQ, shape [M, 1], broadcast over N) plus
 * per-column scale of B (BQ, shape [1, N], broadcast over M). Both scale tensors
 * are AccDataType (float). There is NO quant-group size; QK_A == QK_B == 1.
 *
 * RowColQuant neither reshuffles its operands nor has a quant group size, so the
 * whole run() body is quant_bridge::run_scalar_quant_gemm() -- shared verbatim
 * with tensor_quant, which differs only in its scale-buffer extents. This file
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
 * Run RowColQuant GEMM:
 *   C[M,N] = (A[M,K] * AQ[M,1]) @ (B[K,N] * BQ[1,N])
 * with AQ a per-row scale of A (M floats) and BQ a per-column scale of B (N
 * floats). A, B, AQ, BQ, C are host pointers; device memory is managed
 * internally. time_ms is an optional output. Returns 0 on success.
 */
int dispatcher_run_rowcolquant_gemm(const void* A,
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
    // RowColQuant carries one scale per A row ([M,1]) and one per B column
    // ([1,N]), so the scale buffers are M and N elements. Everything else --
    // guards, packed-stride contract, device buffers, args fill, launch,
    // copy-back -- is the shared scalar-quant body, identical to tensor_quant's.
    return quant_bridge::
        run_scalar_quant_gemm<SelectedKernel, ADataType, BDataType, CDataType, QDataType>(
            "dispatcher_run_rowcolquant_gemm",
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
            /*aq_elems=*/M,
            /*bq_elems=*/N,
            k_batch,
            time_ms);
}

} // extern "C"
