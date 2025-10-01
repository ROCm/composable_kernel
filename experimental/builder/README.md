# CK Builder

This is a framework for instantiate kernels and query kernel settings.

The framework is built on a semantic understanding of the convolution kernels.
This is a prototype implemenentation for CK integration in MIOpen.

Draft of kernel description, to be iterated and aligned with existing CK convolution kernels.

- **Signature**
  *The mathematical description of the kernel operation.*

  - **Direction**  
    Describes the convolution direction: forward, backward-input, or backward-weight.

  - **Tensor Layouts**  
    Specifies memory layout conventions for input, weight, and output tensors.

  - **Groups**  
    Indicates grouping strategy: standard, grouped, or depthwise.

  - **Operations**  
    Lists unary and binary operations applied to tensors (e.g., activation, bias add, residual fusion).

  - **Spatial Parameters**  
    Defines padding, stride, and dilation used in the convolution.

  - **Data Types**  
    Specifies precision formats for input, weight, and output tensors.

  - **Broadcasting Semantics**  
    Indicates whether broadcasting is applied for binary operations.

- **Algorithm**  
  *The implementation strategy and hardware-specific optimizations.*

  - **Tiling Strategy**  
    Defines compute block dimensions and wave mapping.

  - **Thread Mapping**  
    Describes how threads are assigned to tiles and waves.

  - **Memory Transfers**  
    Details how A, B, and C tensors are moved through memory hierarchy (global, LDS, registers).

  - **Pipeline**  
    Describes buffering, scheduling, and vectorization strategies.

  - **Tuning Parameters**  
    Includes kernel-specific tuning knobs (e.g., `ak1`, `bk1`, prefetch depth).

  - **Hardware Optimization**  
    Captures architecture-specific choices (e.g., MFMA type, LDS usage, register pressure).
