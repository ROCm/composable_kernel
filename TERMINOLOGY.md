[Back to the main page](./README.md)

# Composable Kernel Terminology

This document provides a technical reference for terminology used in the Composable Kernel library, organized by conceptual progression from hardware to machine learning operations.

---

## Glossary Index (Alphabetical)

- [Activation Function](#activation-function)
- [Add+Multiply](#addmultiply)
- [Alignment](#alignment)
- [Allocator](#allocator)
- [API](#api)
- [Attention / Multi-Head Attention](#attention--multi-head-attention)
- [Autotuning](#autotuning)
- [Backward Pass](#backward-pass)
- [Bank Conflict](#bank-conflict)
- [Batch](#batch)
- [BatchNorm](#batchnorm)
- [Batched GEMM](#batched-gemm)
- [Benchmark](#benchmark)
- [Bias](#bias)
- [Bias Addition](#bias-addition)
- [Broadcasting](#broadcasting)
- [CMake](#cmake)
- [Contraction](#contraction)
- [Continuous Integration](#continuous-integration-ci)
- [Convolution](#convolution-convnd)
- [Col2Im](#im2colcol2im)
- [Dense Tensor](#dense-tensor)
- [Device Code / Host Code](#device-code--host-code)
- [Device Synchronization](#device-synchronization)
- [Dilation](#dilation)
- [Elementwise](#elementwise)
- [Epilogue](#epilogue)
- [Embedding](#embedding)
- [Epoch](#epoch)
- [Forward Pass](#forward-pass)
- [Fused Operation](#fused-operation)
- [FlashAttention](#flashattention)
- [GEMM](#gemm-general-matrix-multiply)
- [GEMV](#gemv)
- [Global Memory](#global-memory)
- [Gradient Accumulation](#gradient-accumulation)
- [Host-Device Transfer](#host-device-transfer)
- [Im2Col](#im2colcol2im)
- [Inference](#inference)
- [Inner Product](#inner-product)
- [Iteration](#iteration)
- [Kernel](#kernel)
- [Kernel Specialization](#kernel-specialization)
- [Launch Parameters](#launch-parameters)
- [Layer](#layer)
- [Loss Function](#loss-function)
- [Macro](#macro)
- [Matrix](#matrix)
- [Matrix-Vector Multiplication](#matrix-vector-multiplication)
- [Memory Coalescing](#memory-coalescing)
- [Mini-batch](#mini-batch)
- [Mixed Precision](#mixed-precision)
- [Normalization](#normalization-batchnorm-layernorm-instancenorm)
- [Norm](#norm)
- [Numerical Precision](#numerical-precision)
- [Numerical Stability](#numerical-stability)
- [Occupancy](#occupancy)
- [Optimizer](#optimizer)
- [Outer Product](#outer-product)
- [Padding](#padding)
- [Parameter](#parameter)
- [Permute/Transpose](#permutetranspose)
- [Pinned Memory](#pinned-memory)
- [Pooling](#pooling)
- [Processing Units](#processing-units)
- [Profiling](#profiling)
- [Quantization](#quantization)
- [Registers](#registers)
- [Reduction](#reduction)
- [Regression Test](#regression-test)
- [Reference Kernel](#reference-kernel)
- [Residual Block](#residual-block)
- [Residual Connection](#residual-connection)
- [Shared Memory / LDS](#shared-memory--lds)
- [SIMT / SIMD](#simt--simd)
- [Softmax](#softmax)
- [Sparse Tensor](#sparse-tensor)
- [Split-K GEMM](#split-k-gemm)
- [Stride](#stride)
- [Test](#test)
- [Thread / Work-item](#thread--work-item)
- [Thread Block / Work Group](#thread-block--work-group)
- [Tile](#tile)
- [Trace](#trace)
- [Transpose](#permutetranspose)
- [Unit Test](#unit-test)
- [Utilization](#utilization)
- [Vector](#vector)
- [Warp / Wavefront](#warp--wavefront)
- [Wrapper](#wrapper)
- [Workspace](#workspace)

---

## 1. Hardware and Memory Hierarchy

### Processing Units
The GPU is composed of multiple hardware units (SMs on NVIDIA, CUs on AMD), each containing many cores that execute threads in parallel. These units manage shared resources and coordinate execution at scale.

### Registers
The fastest memory tier, registers are private to each thread/work-item and used for storing temporary variables during computation. AMD distinguishes between vector (VGPR) and scalar (SGPR) registers, while NVIDIA uses a unified register file.

### Shared Memory / LDS
High-bandwidth, low-latency on-chip memory accessible to all threads within a block (CUDA) or work group (ROCm). It enables fast data sharing and synchronization, but is limited in capacity and must be managed to avoid bank conflicts.

### Global Memory
The main device memory accessible by all threads, offering high capacity but higher latency than shared memory. Efficient global memory access patterns are critical for high performance.

### Pinned Memory
Host memory that is page-locked to accelerate transfers between CPU and GPU, reducing overhead for large data movements.

### Workspace
Temporary memory allocated for intermediate computations during kernel execution. Workspaces are reused across operations to minimize memory allocation overhead.

### Dense Tensor
A tensor in which most elements are nonzero, typically stored in a contiguous block of memory.

### Sparse Tensor
A tensor in which most elements are zero, allowing for memory and computation optimizations by storing only nonzero values and their indices.

---

## 2. Execution Model

### Thread / Work-item
The smallest unit of parallel execution, each running an independent instruction stream on a single data element. Threads are grouped for efficient scheduling and resource sharing.

### Warp / Wavefront
A group of threads that execute instructions in lockstep, forming the SIMD group. Divergence within these groups can impact performance due to serialization.

### Thread Block / Work Group
A collection of threads/work-items that can synchronize and share memory. Blocks/groups are scheduled independently and mapped to hardware units for execution.

### SIMT / SIMD
SIMT (Single-Instruction, Multi-Thread) allows threads in a warp to diverge, while SIMD (Single-Instruction, Multi-Data) enforces strict lockstep execution within wavefronts. These models define how parallelism is expressed and managed on different architectures.

### Occupancy
The ratio of active warps/wavefronts to the maximum supported by a hardware unit, influencing the ability to hide memory latency and maximize throughput.

### Utilization
The degree to which hardware resources (compute, memory, bandwidth) are used during kernel execution. High utilization indicates efficient use of available resources.

### Batch
A collection of data samples processed together in a single forward or backward pass, improving computational efficiency and statistical stability.

### Mini-batch
A subset of the full dataset processed together in one iteration, balancing memory usage and convergence speed during training.

### Iteration
A single update step in training, typically corresponding to processing one mini-batch and updating model parameters.

### Epoch
One complete pass through the entire training dataset during model training.

---

## 3. Programming Model and Kernel Structure

### Kernel
A function executed on the GPU, typically written in HIP or CUDA, that performs parallel computations over input data. Kernels are launched with specific grid and block dimensions to map computation to hardware.

### Kernel Specialization
The process of generating or selecting kernels optimized for specific data types, tensor shapes, or hardware features. Specialization leverages template metaprogramming and compile-time parameters for maximal efficiency.

### Reference Kernel
A baseline kernel implementation used for correctness and performance comparison. Reference kernels are typically simple and unoptimized, serving as a standard for validating optimized versions.

### Launch Parameters
Configuration values (e.g., grid size, block size) that determine how a kernel is mapped to hardware resources. Proper tuning of these parameters is essential for optimal performance.

### Device Code / Host Code
Device code runs on the GPU and is responsible for parallel computation, while host code runs on the CPU, managing data preparation, kernel launches, and result collection.

### Device Synchronization
Mechanisms to ensure all GPU operations are complete before proceeding, using synchronization primitives or API calls to maintain correctness.

### Macro
A preprocessor directive that defines code to be expanded before compilation, often used for code generation and specialization. Macros enable parameterization and code reuse in kernel development.

### Wrapper
A function or class that provides a simplified or unified interface to underlying kernel implementations. Wrappers abstract complexity and enable flexible composition of operations.

### Layer
A modular component of a neural network that transforms input data, such as convolutional, normalization, or activation layers. Layers are composed sequentially or in parallel to build deep models.

### Operation
A computational transformation applied to tensors or data, such as addition, multiplication, convolution, or activation. Operations are the building blocks of layers and kernels in neural networks and linear algebra.

### Parameter
A learnable variable (e.g., weight or bias) in a model, updated during training to minimize the loss function.

### Embedding
A learned representation that maps discrete input tokens (such as words or indices) to continuous vectors, enabling efficient processing of categorical data.

### Allocator
A component or function responsible for managing memory allocation and deallocation on the device or host, ensuring efficient use of memory resources.

### API
(Application Programming Interface) A set of functions, classes, or protocols that allow users to interact with the Composable Kernel library or its components programmatically.

---

## 4. Memory Access and Data Layout

### Memory Coalescing
An optimization where consecutive threads access consecutive memory addresses, allowing a single memory transaction to serve multiple threads. Proper coalescing is vital for achieving peak memory bandwidth.

### Alignment
Ensuring data structures are stored at memory addresses that are multiples of a specific value, which improves access efficiency and avoids misaligned accesses.

### Bank Conflict
Occurs when multiple threads in a warp/wavefront access different addresses mapping to the same shared memory bank, causing serialization and reduced bandwidth.

### Padding
The addition of extra elements (often zeros) to tensor edges, used to control output size in convolution and pooling, or to align data for efficient memory access.

### Permute/Transpose
Operations that rearrange the order of tensor axes, often required to match kernel input formats or optimize memory access patterns.

### Host-Device Transfer
The process of moving data between CPU (host) and GPU (device) memory. Host-device transfers can be a performance bottleneck and are optimized using pinned memory and asynchronous operations.

### Stride
The step size to move from one element to the next in a particular dimension of a tensor or matrix. In convolution and pooling, stride determines how far the kernel moves at each step.

### Dilation
The spacing between kernel elements in convolution operations, allowing the receptive field to grow without increasing kernel size.

### Im2Col/Col2Im
Data transformation techniques that convert image data to column format (im2col) for efficient convolution and back (col2im) to reconstruct the original layout.

---

## 5. Kernel Operations and Optimization

### Elementwise
Operations applied independently to each tensor element, such as addition or multiplication. These are highly parallelizable and benefit from efficient memory access.

### Tile
A sub-region of a tensor or matrix processed by a block or thread, used to improve memory locality and enable blocking strategies in kernels.

### Fused Operation
The combination of multiple operations into a single kernel launch, reducing memory traffic and improving performance by minimizing intermediate storage.

### Autotuning
Systematic exploration of kernel configuration parameters (tile sizes, block dimensions, etc.) to identify optimal settings for specific hardware and workloads, often using search algorithms or machine learning.

### Profiling
The measurement and analysis of kernel performance to identify bottlenecks and guide optimization efforts, using tools and instrumentation.

### Trace
A record of events or operations during kernel execution, used for debugging and performance analysis. Tracing tools help identify performance issues and optimize execution flow.

### Epilogue
The final stage of a kernel or operation, often applying activation functions, bias, or other post-processing steps. Epilogues are critical for integrating kernel outputs into larger computation graphs.

### Add+Multiply
A common fused operation in ML and linear algebra, where an elementwise addition is immediately followed by multiplication, often used for bias and scaling in neural network layers.

---

## 6. Linear Algebra and ML Operations

### GEMM (General Matrix Multiply)
A core operation in linear algebra and deep learning, computing C = αAB + βC for matrices A, B, and C. Efficient GEMM implementations are critical for high-performance ML workloads.

### Batched GEMM
Simultaneous execution of multiple independent GEMM operations, increasing throughput for workloads with many small matrix multiplications.

### Split-K GEMM
A parallelization strategy that partitions the reduction dimension (K) across multiple compute units, increasing parallelism for large matrix multiplications.

### Contraction
A generalization of matrix multiplication, summing over shared indices between tensors. Used for higher-dimensional tensor operations in scientific computing and ML.

### Convolution (ConvND)
An operation extracting features from data (e.g., images) by sliding a kernel over input tensors and computing weighted sums. Supports N-dimensional data and is central to deep learning.

### Pooling
Reduces spatial dimensions of tensors via operations like max or average pooling, commonly used in neural networks to downsample feature maps.

### Normalization (BatchNorm, LayerNorm, InstanceNorm)
Techniques to stabilize and accelerate training by normalizing activations across batches, features, or instances. Each method targets different axes and statistical properties.

### BatchNorm
A normalization technique that standardizes the activations of a previous layer for each mini-batch, improving training speed and stability.

### Activation Function
A non-linear function (e.g., ReLU, GELU, Sigmoid) applied to the output of a [layer](#layer) or [operation](#operation), enabling neural networks to model complex relationships.

### Attention / Multi-Head Attention
Mechanisms that allow models to focus on relevant input elements, with multi-head variants computing multiple attention distributions in parallel. Used extensively in transformer architectures.

### FlashAttention
An optimized attention algorithm that reduces memory complexity by computing attention scores in blocks, leveraging GPU memory hierarchy for efficiency.

### Residual Connection
Adds the input of a layer to its output, facilitating gradient flow and enabling the training of deep networks.

### Residual Block
A sequence of layers with a skip connection, allowing gradients to flow directly through the network and mitigating vanishing gradient problems.

### Bias
An additional parameter added to the output of operations like GEMM or convolution, allowing the model to fit data with nonzero mean.

### Bias Addition
The operation of adding a bias vector to the output of a layer or operation, shifting the result and improving model expressiveness.

### Gradient Accumulation
The process of summing gradients over multiple mini-batches before updating model parameters, useful for large models or limited memory.

### Forward Pass
The computation that propagates input data through a model to produce outputs, used in both training and inference.

### Backward Pass
The computation of gradients with respect to model parameters, enabling optimization during training via backpropagation.

### Inference
Running a trained model to make predictions on new data, as opposed to training.

### Loss Function
A mathematical function that quantifies the difference between model predictions and ground truth, guiding parameter updates during training.

### Optimizer
An algorithm (e.g., SGD, Adam) that updates model parameters based on gradients to minimize the loss function.

### Softmax
A function that converts a vector of values to a probability distribution, commonly used in classification tasks.

### Mixed Precision
The use of multiple numerical precisions (e.g., FP16, FP32) within a single computation to balance performance, memory usage, and numerical stability.

### Numerical Precision
The number of significant digits with which a value is represented and computed, affecting accuracy and stability.

### Numerical Stability
The property of an algorithm to minimize errors due to floating-point arithmetic, critical for reliable training and inference.

### GEMV
(Matrix-Vector Multiplication) The operation of multiplying a matrix by a vector, producing another vector. GEMV is a core linear algebra primitive, widely used in neural networks and scientific computing.

### Matrix
A two-dimensional array of numbers arranged in rows and columns, fundamental to linear algebra and GPU computation. Matrices are the primary data structure for representing weights, activations, and transformations in ML.

### Matrix-Vector Multiplication
See [GEMV](#gemv).

### Vector
A one-dimensional array of numbers, often representing features, weights, or activations in ML models. Vectors are used in matrix-vector and vector-vector operations.

### Inner Product
Also known as the dot product, it computes the sum of elementwise products of two vectors, yielding a scalar. Inner products are fundamental to similarity measures and neural computation.

### Outer Product
The result of multiplying a column vector by a row vector, producing a matrix. Outer products are used in rank-1 updates and some ML algorithms.

### Norm
A function that measures the magnitude of a vector or matrix, such as L2 (Euclidean) or L1 norm. Norms are used in regularization, normalization, and optimization.

---

## 7. Testing, Build, and Infrastructure

### Unit Test
Automated checks verifying the correctness of small, isolated code units such as kernels or functions.

### Test
A general term for any procedure or script that checks the correctness or performance of code, including unit, regression, and integration tests.

### Regression Test
Ensures new changes do not break existing functionality or degrade performance, maintaining code reliability.

### Benchmark
Performance tests measuring kernel, model, or library throughput and latency under various conditions.

### Continuous Integration (CI)
Automated pipelines for building, testing, and validating code changes in a shared repository.

### CMake
A cross-platform build system used to configure and generate build files for the project, supporting modularity and portability.

---

## Scientific Context and References

This terminology is grounded in parallel computing theory, numerical linear algebra, and computer architecture. For further reading, see:
- [Building Efficient GEMM Kernels with CK Tile Vendo](https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html)
- [CK Tile Flash](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html)

This document assumes familiarity with parallel computing, linear algebra, and computer architecture principles.
