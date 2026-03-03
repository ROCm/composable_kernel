# Composable Kernel Builder Design Documentation

This directory contains the builder framework for Composable Kernel, which provides a compile-time, type-safe interface for constructing convolution operations with various configurations.

## Table of Contents

- [Convolution Signature](#convolution-signature)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Core Components](#core-components)
  - [Concepts and Validation](#concepts-and-validation)
- [Convolution Algorithm](#convolution-algorithm)
- [Convolution Factory](#convolution-factory)
---

## Convolution Signature

### Overview

The convolution signature system provides a **compile-time description** of grouped convolution operations. A signature is a collection of properties that fully characterize a convolution kernel's mathematical and operational behavior, enabling:

- **Compile-time validation**: Ensures type safety and correctness before kernel instantiation
- **Kernel selection**: Matches user requirements to optimized implementations
- **Specialization**: Enables optimized code paths for specific configurations
- **Composability**: Supports building complex operations from simpler components

The signature leverages modern C++20 features, particularly **concepts**, to provide expressive, self-documenting interfaces with compile-time guarantees.

### Architecture

The signature system is organized into a hierarchical structure:

```
┌─────────────────────────────────────────────────────────┐
│                    ConvSignature                        │
├─────────────────────────────────────────────────────────┤
│ Properties:                                             │
│   • spatial_dim: int           (1D, 2D, or 3D)          │
│   • direction: ConvDirection   (Fwd/BwdData/BwdWeight)  │
│   • data_type: DataType        (default data type)      │
│   • accumulation_data_type: DataType                    │
│   • input: ConvTensor          ──┐                      │
│   • weight: ConvTensor         ──│                      │
│   • output: ConvTensor         ──│                      │
└──────────────────────────────────┼──────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │           ConvTensor                    │
              ├─────────────────────────────────────────┤
              │ ╔═════════════════════════════════════╗ │
              │ ║ TensorConfig (required)             ║ │
              │ ╠═════════════════════════════════════╣ │
              │ ║  • layout: TensorLayout              ║ │
              │ ║  • data_type: DataType (optional)   ║ │
              │ ╚═════════════════════════════════════╝ │
              │                                         │
              │ ┌─────────────────────────────────────┐ │
              │ │ TensorOperation (optional)          │ │
              │ ├─────────────────────────────────────┤ │
              │ │  • elementwise_operation            │ │
              │ │  • auxiliary_operand_configs[]      │ │
              │ │    (each is also ConvTensor)  ◄───────┼─┐
              │ └─────────────────────────────────────┘ │ │
              └─────────────────────────────────────────┘ │
                                                          │
                                 Recursive ───────────────┘
```
Key Design Points:
  - ConvSignature contains three ConvTensor instances (input, weight, output)
  - All tensors share the same ConvTensor structure
  - Each ConvTensor has:
    - TensorConfig (required): Defines layout as well as optional data and compute type overrides
    - TensorOperation (optional): Defines fused elementwise operations
  - Auxiliary operands (e.g., bias) in TensorOperation also use the ConvTensor type

### Core Components

#### 1. Signature Level

The top-level signature contains global properties that apply to the entire convolution operation:

```cpp
template <typename T>
concept ConvSignatureDescriptor = requires(T t) {
    { t.spatial_dim } -> std::convertible_to<unsigned int>;  // 1, 2, or 3
    { t.input } -> ConvTensorDescriptor;
    { t.weight } -> ConvTensorDescriptor;
    { t.output } -> ConvTensorDescriptor;
    requires ConvolutionDirectionWellDefinedIfProvided<T>;   // Optional direction
    requires detail::DataTypeWellDefinedIfProvided<T>; // Optional default data type
    requires detail::ElementwiseOpWellDefinedIfProvided<T>; // Optional default elementwise operation
};
```

**Properties:**
- **`spatial_dim`**: Dimensionality of the convolution (1D, 2D, or 3D)
- **`direction`**: Operation type (Optional, defaults to FORWARD)
  - `FORWARD`: Standard forward convolution
  - `BACKWARD_DATA`: Gradient computation w.r.t. input
  - `BACKWARD_WEIGHT`: Gradient computation w.r.t. weights
- **`data_type`**: Default data type for all tensors (FP32, FP16, BF16, FP8, I8, U8). (Optional, defaults to UNDEFINED_DATA_TYPE which indicates the type should be inferred or specified per-tensor, may be overridden by individual tensors)
- **`elementwise_operation`**: Default elementwise operation for all tensors (Optional, defaults to PASS_THROUGH, may be overridden by individual tensors via their `operation` field)
- **`accumulation_data_type`**: Type used for internal accumulation

#### 2. Tensor Level

Each tensor (input, weight, output) has its own descriptor:

```cpp
template <typename T>
concept ConvTensorDescriptor = requires(T t) {
    { t.config } -> TensorConfigDescriptor;
    requires ElementwiseOpWellDefinedIfProvided<T>;
};
```

A tensor descriptor encapsulates:
- **Configuration**: Layout and data type information
- **operation** Fused elementwise operations on this tensor (Optional, default provided by ConvSignatureDescriptor)

#### 3. Tensor Configuration

Describes the memory layout and data types:

```cpp
template <typename T>
concept TensorConfigDescriptor = requires(T t) {
    { t.layout } -> std::convertible_to<TensorLayout>;
    requires detail::DataTypeWellDefinedIfProvided<T>; // Override data type (Optional, default provided by ConvSignatureDescriptor)
};
```

**Layout Types** (dimension-specific):
- **Special Values**:
  - `UNDEFINED_TENSOR_LAYOUT`: Placeholder value indicating layout is not yet specified or should be inferred

- **1D Convolution**:
  - Input: `GNCW`, `GNWC`, `NWGC`, `NGCW`, `G_NW_C_strided`
  - Weight: `GKXC`, `GKCX`, `KXGC`, `G_K_X_C_strided`
  - Output: `GNKW`, `GNWK`, `NWGK`, `NGKW`, `G_NW_K_strided`
  
- **2D Convolution**:
  - Input: `GNCHW`, `GNHWC`, `NHWGC`, `NGCHW`, `G_NHW_C_strided`
  - Weight: `GKYXC`, `GKCYX`, `KYXGC`, `G_K_YX_C_strided`
  - Output: `GNKHW`, `GNHWK`, `NHWGK`, `NGKHW`, `G_NHW_K_strided`
  
- **3D Convolution**:
  - Input: `GNCDHW`, `GNDHWC`, `NDHWGC`, `NGCDHW`, `G_NDHW_C_strided`
  - Weight: `GKZYXC`, `GKCZYX`, `KZYXGC`, `G_K_ZYX_C_strided`
  - Output: `GNKDHW`, `GNDHWK`, `NDHWGK`, `NGKDHW`, `G_NDHW_K_strided`

- **Bias Tensors**:
  - `GC`, `G_C_strided`, `G_K_strided`

Where:
- `G` = Groups
- `N` = Batch size
- `C` = Input channels
- `K` = Output channels (filters)
- `W`, `H`, `D` = Width, Height, Depth (spatial dimensions)
- `X`, `Y`, `Z` = Filter dimensions

#### 4. Tensor Operations

Describes fused elementwise operations applied to a tensor:

```cpp
template <typename T>
concept TensorOperatorDescriptor = requires(T t) {
    { t.elementwise_operation } -> std::convertible_to<ElementwiseOperation>;
    requires AuxiliaryOperandConfigsWellDefinedIfProvided<T>;
};
```

**Supported Operations:**

The `ElementwiseOperation` enum in `types.hpp` defines 35 operations:

- **Identity**: `PASS_THROUGH`
- **Scaling and arithmetic**: `SCALE`, `SCALE_ADD`, `CLAMP`, `ADD_CLAMP`, `BILINEAR`
- **Convolution-specific scaling**: `CONV_SCALE`, `CONV_SCALE_ADD`, `CONV_SCALE_RELU`, `CONV_INVSCALE`
- **Activations**: `RELU`, `LEAKY_RELU`, `CLIPPED_RELU`, `SOFT_RELU`, `GELU`, `SILU`, `SIGMOID`, `TANH`, `ELU`, `SWISH`, `LOGISTIC`, `POWER`, `UNARY_ABS`
- **Composite fused operations**: `BIAS_BNORM_CLAMP`, `SCALEADD_SCALEADD_RELU`, `ADD_RELU_ADD`, `ACTIVATION_MUL_CLAMP`, `ACTIVATION_MUL2_CLAMP`, `ADD_ACTIVATION_MUL_CLAMP`, `ADD_ACTIVATION_MUL2_CLAMP`, `ADD_MUL_ACTIVATION_MUL_CLAMP`, `ADD_MUL2_ACTIVATION_MUL_CLAMP`
- **Dynamic and generic**: `DYNAMIC_UNARY_OP`, `UNARY_COMBINED_OP`, `UNARY_CONVERT`

**Auxiliary Operands:**
Some operations require additional tensor inputs (e.g., bias tensors, scaling factors). These are specified through `auxiliary_operand_configs`, which is an array of `TensorConfigDescriptor` objects describing the layout and data type of each auxiliary input.

### Concepts and Validation

The signature system uses C++20 concepts for compile-time validation at multiple levels:

#### Constraint Concepts

```cpp
// Spatial dimension must be 1, 2, or 3
template <auto N>
concept ConvSpatialDim = std::is_integral_v<decltype(N)> && (N == 1 || N == 2 || N == 3);

// Valid data types for convolution
template <DataType T>
concept ValidConvDataType = 
    (T == DataType::FP32) || (T == DataType::FP16) || (T == DataType::BF16) ||
    (T == DataType::FP8) || (T == DataType::I8) || (T == DataType::U8);
```

#### Validation Concept

```cpp
// Validates a complete signature
template <auto Sig>
concept ValidConvSignature = requires {
    requires ConvSpatialDim<Sig.spatial_dim>;
    requires ValidConvDataType<Sig.data_type>;
};
```

#### Tensor Descriptors

The layout/data type/elementwise operation are described per tensor. This multi-level hierarchy allows:
- **Flexibility**: Each tensor can have independent layout and data type
- **Reusability**: Common configurations can be shared across different signatures
- **Extensibility**: New properties can be added to specific levels without affecting others
- **Clarity**: Separates concerns (global properties vs. tensor-specific properties)

#### Optional Signature Fields

Several fields in the signature are optional:
- **`direction`**: Defaults to `FORWARD` if not specified, reducing boilerplate for the common case
- **Tensor `data_type`**: Falls back to signature's default, allowing mixed-precision with minimal specification
- **Tensor `operation`**: Defaults to `PASS_THROUGH`, supporting both fused and non-fused operations with the same interface

This design follows the principle of "make the common case simple, the complex case possible."

## Convolution Algorithm

The algorithm descriptor specifies **how** a convolution is computed — the implementation strategy including tile sizes, hardware instruction variant, pipeline scheduling, and memory access patterns. It is the complement to the signature, which specifies **what** is computed.

### Algorithm Descriptor Concept

An algorithm descriptor is any struct satisfying the `ConvAlgorithmDescriptor` concept (`conv_algorithm_concepts.hpp`). The required fields depend on the target kernel variant. The dispatcher (`conv_dispatcher.hpp`) uses predicate concepts to classify each algorithm descriptor into one of the supported variants:

- **ReferenceAlgorithm**: Requires only a `specialization` field set to `REFERENCE`. Used for correctness validation.
- **TileAlgorithm**: CK Tile backend. Requires tile-level configuration: block shape, warp tile, block GEMM pipeline, transfer vectorization, and optimizations.
- **Forward-specific** (old CK): XDL V3, XDL, WMMA, DL, Large Tensor. Each requires progressively different fields (thread block, GEMM config, transfer, scheduling, prefetch stages).
- **Backward weight-specific** (old CK): XDL, XDL V3, Two-Stage XDL, DL, Multi-D XDL, WMMA V3, Two-Stage WMMA V3, WMMA, Multi-D WMMA V3.

The `ConvAlgorithmSpecialization` enum provides broad algorithm classes (`REFERENCE`, `LARGE_TENSOR`, `TWO_STAGE`, `MULTIPLE_D`) for requesting a category of algorithm without specifying the full descriptor.

### Algorithm Descriptor Fragmentation

The builder currently requires a different algorithm descriptor shape for each kernel variant. This fragmentation exists along three axes:

1. **Backend** (CK vs CK Tile): The old CK backend flattens ~49 template parameters into a single device operation type (explicit thread block dimensions, block transfer descriptors with LDS configurations, thread cluster arrangements, per-tensor access orders). The CK Tile backend composes higher-level objects — tile partitioner, GEMM pipeline, epilogue pipeline — with ~31 parameters distributed across four composed types.

2. **Instruction set** (MFMA vs WMMA): Within the old CK backend, XDL (MFMA) and WMMA variants require different algorithm descriptor fields. The dispatcher uses separate predicate concepts (`FwdXdlAlgorithm` vs `FwdWmmaAlgorithm`) to classify them, and separate factories to instantiate them.

3. **Direction** (forward vs backward weight vs backward data): Each direction has its own set of factories. Backward weight alone has 9 old CK factory variants (XDL, XDL V3, two-stage XDL, DL, multi-D XDL, WMMA V3, two-stage WMMA V3, WMMA, multi-D WMMA V3).

The result is 16+ per-variant factories, each accepting a different algorithm descriptor shape. MIOpen must currently know which variant it wants and construct the matching descriptor — the builder dispatches but does not unify.

The path toward a single algorithm descriptor runs through the reflection system. `ConvTraits` already provides a common representation for old CK instances across both MFMA and WMMA. Extending this to CK Tile instances will reveal which parameters are genuinely variant-specific versus which can be expressed in a single descriptor and mapped to multiple backends by the dispatcher. See the [reflection documentation](reflect/README.md) for the current state of this bridge.

## Convolution Factory

The factory system translates a (signature, algorithm) pair into a concrete kernel instance. The entry point is `make_conv_instance<SIGNATURE, ALGORITHM, VERSION>()` in `conv_dispatcher.hpp`.

The dispatch proceeds in two phases:

1. **Algorithm classification**: Predicate concepts (`ReferenceAlgorithm`, `TileAlgorithm`, `FwdXdlV3Algorithm`, etc.) inspect the algorithm descriptor's structure to determine which kernel variant it satisfies.

2. **Direction routing**: An `if constexpr` chain routes to the appropriate factory based on convolution direction (forward, backward data, backward weight) and classified algorithm type.

Each factory (e.g., `ConvFwdXdlV3Factory`, `ConvBwdWeightWmmaV3Factory`) transforms builder descriptors into the underlying device operation's template parameters and instantiates the kernel.

The factory design is described in detail in the [factory README](factory/README.md).
