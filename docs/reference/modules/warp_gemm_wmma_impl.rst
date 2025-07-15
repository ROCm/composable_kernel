.. |rst_start_tag| raw:: html

   <div class="section" id="file-doc">

.. |rst_end_tag| raw:: html

   </div>

|rst_start_tag|

.. highlight:: cpp

.. include_file:: include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp

.. _file_ck_tile_ops_gemm_warp_warp_gemm_attribute_wmma.hpp:

include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp
============================================================

This file defines the `WGMMAAttrWmma` enumeration and the `WarpGemmAttributeWmma` struct. These specify attributes for warp-level GEMM operations, particularly those utilizing WMMA (Warp Matrix Multiply-Accumulate) instructions.

**Contents:**

.. contents::
   :local:
   :depth: 2

.. cpp:enum-class:: WGMMAAttrWmma
   :project: ck_tile

   .. _ex_WGMMAAttrWmma:

   `WGMMAAttrWmma`
   --------------------

   Enumeration controlling the attributes for WMMA-based warp-level GEMM operations, related to register allocation and specific WMMA instruction variants.

   .. container:: custom-attributes

      .. cpp:enumerator:: Default_ = 0

         Default WMMA attribute control.

      .. cpp:enumerator:: Vgpr_vgpr_acc_vgpr = 1

         c-vgpr, a-vgpr, b-vgpr, acc-vgpr.

      .. cpp:enumerator:: Vgpr_vgpr_acc_agpr = 2

         c-vgpr, a-vgpr, b-vgpr, acc-agpr.

      .. cpp:enumerator:: Vgpr_agpr_acc_vgpr = 3

         c-vgpr, a-agpr, b-vgpr, acc-vgpr.

      .. cpp:enumerator:: Vgpr_agpr_acc_agpr = 4

         c-vgpr, a-agpr, b-agpr, acc-agpr.

      .. cpp:enumerator:: Agpr_vgpr_acc_vgpr = 5

         c-agpr, a-vgpr, b-vgpr, acc-vgpr.

      .. cpp:enumerator:: Agpr_vgpr_acc_agpr = 6

         c-agpr, a-vgpr, b-vgpr, acc-agpr.

      .. cpp:enumerator:: Agpr_agpr_acc_vgpr = 7

         c-agpr, a-agpr, b-agpr, acc-vgpr.

      .. cpp:enumerator:: Agpr_agpr_acc_agpr = 8

         c-agpr, a-agpr, b-agpr, acc-agpr.

.. _WarpGemmAttributeWmma_Structs:

WarpGemmAttributeWmma Struct Definition
-----------------------------------------

This struct defines general attributes for WMMA-based warp-level GEMM operations.

* `WarpGemmAttributeWmma`

|rst_end_tag|

```rst
.. |rst_start_tag| raw:: html

   <div class="section" id="file-doc">

.. |rst_end_tag| raw:: html

   </div>

|rst_start_tag|

.. highlight:: cpp

.. include_file:: include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp

.. _file_ck_tile_ops_gemm_warp_warp_gemm_attribute_wmma_impl_16bit_traits.hpp:

include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp
==============================================================================

This file defines traits structs for WMMA implementations specifically for 16-bit floating-point (FP16 and BF16) inputs accumulating to FP32 outputs.

**Contents:**

.. contents::
   :local:
   :depth: 2

.. _WarpGemmAttributeWmmaImpl16BitTraits_Structs:

WarpGemmAttributeWmmaImpl 16-bit Traits Struct Definitions
------------------------------------------------------------

These structs specify WMMA attributes for operations with 16-bit inputs (FP16 or BF16) and FP32 accumulation/output.

* `WarpGemmAttributeWmmaImplF16F16F32`
* `WarpGemmAttributeWmmaImplBf16Bf16F32`

|rst_end_tag|

```rst
.. |rst_start_tag| raw:: html

   <div class="section" id="file-doc">

.. |rst_end_tag| raw:: html

   </div>

|rst_start_tag|

.. highlight:: cpp

.. include_file:: include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp

.. _file_ck_tile_ops_gemm_warp_warp_gemm_attribute_wmma_impl_8bit_traits.hpp:

include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp
============================================================================

This file defines traits structs for WMMA implementations specifically for 8-bit integer (INT8) inputs accumulating to INT32 outputs.

**Contents:**

.. contents::
   :local:
   :depth: 2

.. _WarpGemmAttributeWmmaImpl8BitTraits_Structs:

WarpGemmAttributeWmmaImpl 8-bit Traits Struct Definitions
----------------------------------------------------------

This struct specifies WMMA attributes for operations with INT8 inputs and INT32 accumulation/output.

* `WarpGemmAttributeWmmaImplInt8Int8Int32`

|rst_end_tag|

```rst
.. |rst_start_tag| raw:: html

   <div class="section" id="file-doc">

.. |rst_end_tag| raw:: html

   </div>

|rst_start_tag|

.. highlight:: cpp

.. include_file:: include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp

.. _file_ck_tile_ops_gemm_warp_warp_gemm_attribute_wmma_impl_base_traits.hpp:

include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp
=============================================================================

This file defines a base traits struct for WMMA implementations, along with specialized traits for various data types.

**Contents:**

.. contents::
   :local:
   :depth: 2

.. _WarpGemmAttributeWmmaImplBaseTraits_Structs:

WarpGemmAttributeWmmaImpl Base Traits Structs and Specializations
-------------------------------------------------------------------

These structs provide common base traits and specific implementations for WMMA operations across different data types.

* `WarpGemmAttributeWmmaImplBase`
* `WmmaTraits` (Base template class)
* `WmmaTraits<double>`
* `WmmaTraits<float>`
* `WmmaTraits<half_t>`
* `WmmaTraits<bhalf_t>`
* `WmmaTraits<int8_t>`
* `WmmaTraits<int>`
* `WmmaTraits<uint8_t>`

|rst_end_tag|

This video provides an overview of the Composable Kernel library's structure and goals, which helps understand the context of these WMMA-related classes: [Lecture 25: Speaking Composable Kernel (CK)](https://www.youtube.com/watch?v=-732zELVbpU).
http://googleusercontent.com/youtube_content/2
