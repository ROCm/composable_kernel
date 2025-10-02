# CK Builder

This is a framework for instantiate kernels and query kernel settings.

The framework is built on a semantic understanding of the convolution kernels.
This is a prototype implemenentation specific to convolutions for CK integration in MIOpen.

Draft of kernel description, to be iterated and aligned with existing CK convolution kernels.

- **Signature**
  *The mathematical description of the kernel operation.*

  - **Spatial dimensionality**
    1D, 2D, or 3D.

  - **Direction**  
    Describes the convolution direction: forward, backward-data, or backward-weight.

  - **Tensor Layouts**  
    Channels first or channels last.
    Specifies memory layout conventions for input, weight, and output tensors.

  - **Data Types**  
    Tenosor input and output types (FP32, FP16, BF16).

  - **Operations**  
    Lists unary and binary operations applied to tensors (e.g., bias or clamp).

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
