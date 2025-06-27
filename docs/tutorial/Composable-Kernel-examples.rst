==========================
Composable Kernel Examples
==========================

This comprehensive guide covers the Composable Kernel example collection, showcasing high-performance GPU kernels optimized for AI/ML workloads. Each example demonstrates advanced fusion techniques, memory optimization strategies, and integration patterns for modern deep learning frameworks.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
============

The Composable Kernel library provides a comprehensive collection of optimized GPU kernels specifically designed for AI/ML applications. These examples showcase:

- **Advanced Kernel Fusion**: Eliminating intermediate memory usage through operation fusion
- **XDL/MFMA Integration**: Leveraging AMD matrix core instructions for peak performance
- **Memory Hierarchy Optimization**: Efficient use of GPU memory subsystems
- **Framework Integration**: Easy integration with PyTorch, TensorFlow, and other ML frameworks

Foundation Operations
====================

GEMM (General Matrix Multiplication)
-------------------------------------

**Example**: `01_gemm <../example/01_gemm/README.md>`_

The fundamental building block for all AI/ML computations. Implements highly optimized matrix multiplication using:

- **XDL/MFMA Instructions**: Direct hardware acceleration via matrix core instructions
- **Advanced Tiling**: Hierarchical blocking for optimal memory hierarchy utilization
- **Multiple Precision Support**: FP32, FP16, BF16, INT8, and mixed-precision variants

**Key Implementation Features**:

.. code-block:: cpp

   // Device interface instantiation
   using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmXdl<
       ALayout, BLayout, CLayout,           // Memory layouts
       ADataType, BDataType, CDataType,     // Data types
       AccDataType,                         // Accumulation precision
       AElementOp, BElementOp, CElementOp,  // Element-wise operations
       GemmSpecialization,                  // Optimization flags
       BlockSize, MPerBlock, NPerBlock,     // Thread block configuration
       K0PerBlock, K1,                      // K-dimension tiling
       MPerXDL, NPerXDL,                    // Matrix instruction mapping
       MXdlPerWave, NXdlPerWave>;           // Wave-level parallelism

**AI/ML Applications**:
- Transformer feed-forward networks
- CNN fully connected layers  
- Embedding projections
- Attention mechanism building blocks

**Related Examples**: `03_gemm_bias_relu`, `21_gemm_layernorm`, `24_batched_gemm`

Convolution Operations
----------------------

**Example**: `09_convnd_fwd <../example/09_convnd_fwd/README.md>`_

N-dimensional convolution using the implicit GEMM algorithm, supporting modern CNN architectures:

**Implicit GEMM Transformation**:

.. code-block:: cpp

   // Conceptual mapping without explicit data transformation
   // Weight matrix: [C_out, C_in * K_total]
   // Input matrix (implicit): [C_in * K_total, N * H_total] 
   // Output matrix: [C_out, N * H_total]

**Advanced Address Calculation**:

.. code-block:: cpp

   __device__ auto calculate_input_index(
       int output_spatial_idx, int kernel_idx,
       const ConvParam& params) {
       
       // Transform output coordinates to input coordinates
       auto input_coords = apply_convolution_transform(
           output_spatial_idx, kernel_idx, 
           params.stride, params.dilation, params.padding);
       
       return validate_and_compute_address(input_coords);
   }

**Modern CNN Integration**:
- **ResNet**: Bottleneck convolutions (1×1 → 3×3 → 1×1 sequences)
- **EfficientNet**: Depthwise separable convolutions
- **Vision Transformers**: Patch embedding convolutions

**Related Examples**: `11_convnd_fwd_bias`, `62_convnd_activ`, `17_convnd_bwd_data`

Attention Mechanisms
====================

Fused Multi-Head Attention
---------------------------

**Example**: `32_batched_gemm_scale_softmax_gemm <../example/32_batched_gemm_scale_softmax_gemm/README.md>`_

Complete fused attention mechanism implementing the core of Transformer models:

**Mathematical Formulation**:

.. math::

   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

**Advanced Tiled Algorithm**:

.. code-block:: cpp

   // Flash Attention-style implementation
   for (auto q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
       Load_Q_Tile(q_tile);
       
       // Online softmax statistics
       float max_val = -INFINITY, sum_exp = 0.0f;
       Clear_Output_Accumulator();
       
       for (auto kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
           Load_KV_Tile(kv_tile);
           
           // QK^T computation
           Compute_Attention_Scores();
           Apply_Scale_Factor();
           
           // Online softmax update
           Update_Softmax_Statistics(max_val, sum_exp);
           
           // Accumulate weighted values
           Accumulate_PV_Product();
       }
       
       Normalize_And_Store_Output();
   }

**Memory Efficiency**: O(√N) memory complexity vs O(N²) for naive implementation

**Transformer Applications**:
- **Language Models**: GPT, BERT self-attention
- **Vision Transformers**: Patch-based attention for images
- **Multi-Modal Models**: Cross-attention between modalities

**Related Examples**: `31_batched_gemm_gemm`, `47_gemm_bias_softmax_gemm_permute`

Normalization Operations
========================

Layer Normalization
--------------------

**Example**: `27_layernorm2d_fwd <../example/27_layernorm2d_fwd/README.md>`_

Critical for Transformer training stability, implementing numerically stable normalization:

**Welford's Algorithm Implementation**:

.. code-block:: cpp

   struct WelfordData {
       float count, mean, m2;  // Running statistics
   };
   
   __device__ void update_welford(WelfordData& w, float x) {
       w.count += 1.0f;
       float delta = x - w.mean;
       w.mean += delta / w.count;
       w.m2 += delta * (x - w.mean);
   }
   
   // Final variance: w.m2 / w.count

**Transformer Integration Patterns**:

.. code-block:: cpp

   // Pre-LN (modern transformers)
   x = x + Attention(LayerNorm(x))
   x = x + MLP(LayerNorm(x))
   
   // Post-LN (original transformer)
   x = LayerNorm(x + Attention(x))
   x = LayerNorm(x + MLP(x))

**Related Examples**: `53_layernorm2d_bwd`, `21_gemm_layernorm`, `34_batchnorm`

Advanced Fusion Patterns
=========================

Multi-Operation Fusion
-----------------------

**Example**: `21_gemm_layernorm <../example/21_gemm_layernorm/README.md>`_

Demonstrates sophisticated operation fusion combining GEMM with layer normalization:

**Fusion Strategy**:

.. code-block:: cpp

   // Single kernel execution
   C_temp = A × B              // GEMM computation in registers
   statistics = compute_stats(C_temp)  // Mean/variance calculation
   output = layernorm(C_temp, statistics, γ, β)  // Normalization

**Memory Benefits**: Eliminates intermediate tensor storage, reducing memory bandwidth by ~50%

**Framework Integration**: Critical for efficient Transformer MLP blocks

Mixed Precision and Quantization
=================================

FP16×INT8 Mixed Precision
--------------------------

**Example**: `64_fpAintB_gemm <../example/64_fpAintB_gemm/README.md>`_

Advanced quantization technique for efficient AI inference:

**Implementation Strategy**:

.. code-block:: cpp

   // On-the-fly dequantization during GEMM
   template<typename QuantType, typename ComputeType>
   __device__ ComputeType dequantize_and_compute(
       QuantType quantized_value,
       float scale_factor) {
       
       ComputeType dequantized = static_cast<ComputeType>(quantized_value) * scale_factor;
       return dequantized;
   }

**Performance Benefits**:
- 2× memory bandwidth improvement for weight storage
- Maintained numerical accuracy through FP32 accumulation
- Essential for large model deployment

**Related Examples**: `14_gemm_quantization`, `40_conv2d_fwd_quantization`

Performance Optimization Techniques
====================================

Memory Access Optimization
---------------------------

**Vectorized Memory Operations**:

.. code-block:: cpp

   // Optimized vectorized loading
   using VectorType = ck::vector_type<half_t, 8>;
   VectorType* vec_ptr = reinterpret_cast<VectorType*>(tensor_ptr);
   VectorType vec_data = *vec_ptr;  // 128-bit load

**Coalesced Access Patterns**:

.. code-block:: cpp

   // Thread cluster for optimal memory coalescing
   using ThreadClusterLengths = S<4, 64, 1>;  // [K0, M, K1]
   using SrcAccessOrder = S<1, 0, 2>;         // Access pattern optimization

XDL/MFMA Integration
--------------------

**Matrix Instruction Mapping**:

.. code-block:: cpp

   // Direct mapping to hardware matrix instructions
   static constexpr auto MPerXDL = 32;      // Matrix instruction M-dimension
   static constexpr auto NPerXDL = 32;      // Matrix instruction N-dimension
   static constexpr auto MXdlPerWave = 4;   // Instructions per wave (M)
   static constexpr auto NXdlPerWave = 2;   // Instructions per wave (N)

**Register Optimization**: Efficient register allocation for accumulator arrays and intermediate data

Framework Integration
=====================

PyTorch Integration
-------------------

**Custom Operator Pattern**:

.. code-block:: cpp

   #include <torch/extension.h>
   #include "ck/tensor_operation/gpu/device/device_gemm.hpp"
   
   torch::Tensor composable_kernel_gemm(
       torch::Tensor a, torch::Tensor b) {
       
       auto c = torch::empty({a.size(0), b.size(1)}, a.options());
       
       auto gemm_op = DeviceGemmInstance{};
       auto argument = gemm_op.MakeArgument(
           a.data_ptr<at::Half>(),
           b.data_ptr<at::Half>(),
           c.data_ptr<at::Half>(),
           a.size(0), b.size(1), a.size(1),
           a.stride(0), b.stride(0), c.stride(0),
           PassThrough{}, PassThrough{}, PassThrough{});
       
       auto invoker = gemm_op.MakeInvoker();
       invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream()});
       
       return c;
   }
   
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       m.def("gemm", &composable_kernel_gemm, "Composable Kernel GEMM");
   }

**Advantages**:
- Direct integration with PyTorch's autograd system
- Optimal memory management through PyTorch's tensor system
- Easy deployment in existing training pipelines

Performance Analysis and Tuning
================================

Profiling and Optimization
---------------------------

**Performance Metrics**:

.. code-block:: bash

   # Example performance analysis
   ./gemm_xdl_fp16 -M 4096 -N 4096 -K 4096 -v 1 -t 1
   
   # Output analysis:
   # Perf: 2.34 ms, 28.7 TFlops, 1.2 TB/s, DeviceGemmXdl
   #       ^^^       ^^^        ^^^      ^^^
   #    Execution  FLOPS    Bandwidth  Kernel
   #    Time      Achieved   Achieved   Type

**Tuning Parameters**:

.. code-block:: cpp

   // Block size tuning for different problem sizes
   constexpr auto BlockSize = 256;        // Thread block size
   constexpr auto MPerBlock = 256;        // M-dimension tile
   constexpr auto NPerBlock = 128;        // N-dimension tile
   constexpr auto K0PerBlock = 4;         // K-dimension blocking
   
   // Memory transfer optimization
   constexpr auto ABlockTransferSrcVectorDim = 2;
   constexpr auto ABlockTransferDstScalarPerVector = 8;

Example Progression Guide
=========================

Learning Path for AI/ML Engineers
----------------------------------

**Beginner Level**:

1. **01_gemm**: Understand basic matrix multiplication
2. **03_gemm_bias_relu**: Learn simple operation fusion
3. **27_layernorm2d_fwd**: Explore normalization techniques

**Intermediate Level**:

4. **09_convnd_fwd**: Master convolution implementation
5. **24_batched_gemm**: Handle batched operations
6. **32_batched_gemm_scale_softmax_gemm**: Implement attention mechanisms

**Advanced Level**:

7. **21_gemm_layernorm**: Complex multi-operation fusion
8. **64_fpAintB_gemm**: Mixed precision and quantization
9. **47_gemm_bias_softmax_gemm_permute**: Complete fused attention with permutation

**Research Level**:

10. **59_grouped_gemm_multi_ABD**: Multi-input tensor operations
11. **66_complex_contraction_bilinear**: Advanced tensor contractions
12. **67_gemm_microscaling**: Cutting-edge quantization techniques

Cross-Reference Matrix
======================

Operation Dependencies and Relationships
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Base Operation
     - Direct Extensions
     - AI/ML Applications
   * - 01_gemm
     - 03_gemm_bias_relu, 21_gemm_layernorm, 24_batched_gemm
     - Linear layers, embeddings, projections
   * - 09_convnd_fwd
     - 11_convnd_fwd_bias, 62_convnd_activ, 17_convnd_bwd_data
     - CNN layers, feature extraction
   * - 27_layernorm2d_fwd
     - 53_layernorm2d_bwd, 21_gemm_layernorm
     - Transformer normalization
   * - 32_batched_gemm_scale_softmax_gemm
     - 47_gemm_bias_softmax_gemm_permute
     - Attention mechanisms
   * - 23_softmax
     - 32_batched_gemm_scale_softmax_gemm
     - Attention weights, classification

Conclusion
==========

The Composable Kernel examples provide a comprehensive foundation for implementing high-performance AI/ML kernels. Key takeaways:

- **Operation Fusion**: Critical for memory bandwidth optimization
- **Hardware Utilization**: XDL/MFMA instructions provide significant speedups
- **Framework Integration**: Easy deployment in existing ML workflows
- **Precision Control**: Mixed precision enables efficient large model deployment

For detailed implementation specifics and advanced optimization techniques, refer to the individual example documentation and the :doc:`API Reference <../API/Reference>`.
