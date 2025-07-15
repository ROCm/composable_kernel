.. |rst_start_tag| raw:: html

   <div class="section" id="file-doc">

.. |rst_end_tag| raw:: html

   </div>

|rst_start_tag|

.. highlight:: cpp

.. include_file:: include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp

.. _file_ck_tile_ops_gemm_warp_warp_gemm_attribute_mfma_impl.hpp:

include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp
================================================================

This file defines the `WGAttrCtlEnum` enumeration and various `WarpGemmAttributeMfmaImpl` struct definitions. These structs specify attributes for warp-level GEMM (General Matrix Multiply) operations, particularly those utilizing MFMA (Matrix Fused Multiply-Add) instructions, with different data types and matrix dimensions.

**Contents:**

.. contents::
   :local:
   :depth: 2

.. cpp:enum-class:: WGAttrCtlEnum
   :project: ck_tile

   .. _ex_WGAttrCtlEnum:

   `WGAttrCtlEnum`
   --------------------

   Enumeration controlling the attributes for warp-level GEMM operations, primarily related to register allocation (e.g., vgpr for vector general-purpose register, agpr for accumulator general-purpose register) for C, A, and B matrices in the context of MFMA instructions.

   .. container:: custom-attributes

      .. cpp:enumerator:: Default_ = 0

         Default attribute control.

      .. cpp:enumerator:: Raw_vvv = 1

         C-vgpr, A-vgpr, B-vgpr.

      .. cpp:enumerator:: Raw_vaa = 2

         C-vgpr, A-agpr, B-agpr.

      .. cpp:enumerator:: Raw_vav = 3

         C-vgpr, A-agpr, B-vgpr.

      .. cpp:enumerator:: Raw_vva = 4

         C-vgpr, A-vgpr, B-agpr.

      .. cpp:enumerator:: Raw_avv = 5

         C-agpr, A-vgpr, B-vgpr.

.. _ex_WarpGemmAttributeMfmaImpl_Structs:

WarpGemmAttributeMfmaImpl Struct Definitions
----------------------------------------------

These structs define specific configurations for warp-level GEMM, parameterized by input/output data types (e.g., BF16, FP16, FP8, INT8) and matrix dimensions (M, N, K). The naming convention generally follows `WarpGemmAttributeMfmaImpl<InputTypeA><InputTypeB><OutputType>M<M>N<N>K<K>`.

.. _WarpGemmAttributeMfmaImpl_BF16:

Bf16Bf16F32 Combinations
^^^^^^^^^^^^^^^^^^^^^^^

These structs specify MFMA attributes for operations with BF16 inputs and FP32 accumulation/output.

* `WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K32`
* `WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8`
* `WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16`
* `WarpGemmAttributeMfmaImplBf16Bf16F32M4N64K4`
* `WarpGemmAttributeMfmaImplBf16Bf16F32M64N4K4`
* `WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16`

.. _WarpGemmAttributeMfmaImpl_FP16:

F16F16F32 Combinations
^^^^^^^^^^^^^^^^^^^^^^

These structs specify MFMA attributes for operations with FP16 inputs and FP32 accumulation/output.

* `WarpGemmAttributeMfmaImplF16F16F32M32N32K8`
* `WarpGemmAttributeMfmaImplF16F16F32M16N16K16`
* `WarpGemmAttributeMfmaImplF16F16F32M16N16K32`
* `WarpGemmAttributeMfmaImplF16F16F32M4N64K4`
* `WarpGemmAttributeMfmaImplF16F16F32M64N4K4`
* `WarpGemmAttributeMfmaImplF16F16F32M32N32K16`

.. _WarpGemmAttributeMfmaImpl_FP8:

FP8 Related Base Structs
^^^^^^^^^^^^^^^^^^^^^^^

These are base template structs for MFMA attributes involving FP8 or BF8 input types, typically accumulating to FP32.

* `WarpGemmAttributeMfmaImpl_f32_16x16x32_f8_base`
* `WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base`
* `WarpGemmAttributeMfmaImpl_f32_16x16x128_f8_bf8_base`
* `WarpGemmAttributeMfmaImpl_f32_32x32x64_f8_bf8_base`

.. _WarpGemmAttributeMfmaImpl_INT8:

Int8 Related Structs (`_i32_`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These structs specify MFMA attributes for operations with INT8 inputs and INT32 accumulation/output.

* `WarpGemmAttributeMfmaImpl_i32_32x32x16_i8`
* `WarpGemmAttributeMfmaImpl_i32_16x16x32_i8`
* `WarpGemmAttributeMfmaImpl_i32_16x16x64_i8`
* `WarpGemmAttributeMfmaImpl_i32_32x32x32_i8`

|rst_end_tag|
