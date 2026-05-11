# Tensor Descriptor vs Tensor Adaptor in Composable Kernel

## Overview

This document explains the key differences between `tensor_descriptor` and `tensor_adaptor` in the Composable Kernel (CK) library. Both are fundamental abstractions for managing tensor layouts and coordinate transformations, but they serve different purposes and have distinct characteristics.

---

## Quick Summary

| Aspect | `tensor_adaptor` | `tensor_descriptor` |
|--------|------------------|---------------------|
| **Purpose** | Coordinate transformation logic | Complete tensor specification |
| **Inheritance** | Base class | Inherits from `tensor_adaptor` |
| **Memory Info** | No memory size tracking | Tracks `element_space_size` |
| **Vector Info** | No vectorization guarantees | Tracks `GuaranteedVectorLengths` and `GuaranteedVectorStrides` |
| **Use Case** | Pure layout transformations | Full tensor with memory bounds |
| **Offset Calculation** | Maps coordinates only | Calculates actual memory offsets |

---

## Detailed Comparison

### 1. `tensor_adaptor` - The Transformation Engine

**Location:** `include/ck_tile/core/tensor/tensor_adaptor.hpp`

#### What It Is
`tensor_adaptor` is a **pure coordinate transformation abstraction**. It defines how to map between different dimensional representations of tensor indices without any knowledge of the underlying memory layout or size.

#### Key Characteristics

```cpp
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename BottomDimensionHiddenIds,
          typename TopDimensionHiddenIds>
struct tensor_adaptor
{
    // Core functionality: coordinate transformation
    template <typename TopIdx>
    CK_TILE_HOST_DEVICE constexpr auto 
    calculate_bottom_index(const TopIdx& idx_top) const;
    
    // Tracks element size (product of dimensions)
    ElementSize element_size_;
    
    // Stores the transformations
    Transforms transforms_;
};
```

#### What It Does
- **Transforms coordinates** from "top" (user-facing) dimensions to "bottom" (memory-facing) dimensions
- **Chains transformations** through hidden intermediate dimensions
- **Supports operations** like:
  - `make_single_stage_tensor_adaptor()` - Create basic transformation
  - `transform_tensor_adaptor()` - Add new transformations
  - `chain_tensor_adaptors()` - Compose multiple adaptors

#### What It Does NOT Do
- ❌ Track total memory space required
- ❌ Provide vectorization guarantees
- ❌ Calculate actual memory offsets (only coordinate mapping)

#### Example Use Case
```cpp
// Split M dimension for tiling: [M, K] -> [M0, M1, K]
auto adaptor = make_single_stage_tensor_adaptor(
    make_tuple(
        make_unmerge_transform(make_tuple(number<M0>{}, number<M1>{})),
        make_pass_through_transform(number<K>{})
    ),
    make_tuple(sequence<0>{}, sequence<1>{}),  // lower dims
    make_tuple(sequence<0, 1>{}, sequence<2>{})  // upper dims
);

// Map coordinates: [M0=2, M1=16, K=32] -> [M=?, K=?]
auto bottom_idx = adaptor.calculate_bottom_index(make_tuple(2, 16, 32));
```

---

### 2. `tensor_descriptor` - The Complete Tensor Specification

**Location:** `include/ck_tile/core/tensor/tensor_descriptor.hpp`

#### What It Is
`tensor_descriptor` is a **complete tensor specification** that extends `tensor_adaptor` with additional memory and performance metadata. It represents a full tensor with known memory bounds and vectorization properties.

#### Key Characteristics

```cpp
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename TopDimensionHiddenIds,
          typename ElementSpaceSize,
          typename GuaranteedVectorLengths_,
          typename GuaranteedVectorSrides_>
struct tensor_descriptor : public tensor_adaptor<...>
{
    // Additional memory information
    ElementSpaceSize element_space_size_;
    
    // Vectorization guarantees
    using GuaranteedVectorLengths = GuaranteedVectorLengths_;
    using GuaranteedVectorStrides = GuaranteedVectorSrides_;
    
    // Calculate actual memory offset
    template <typename Idx>
    CK_TILE_HOST_DEVICE constexpr index_t 
    calculate_offset(const Idx& idx) const;
    
    // Get total memory space
    CK_TILE_HOST_DEVICE constexpr auto 
    get_element_space_size() const;
};
```

#### What It Adds Beyond `tensor_adaptor`
1. **`element_space_size_`** - Total memory space required for the tensor
2. **`GuaranteedVectorLengths`** - Compile-time guarantees about vector access patterns
3. **`GuaranteedVectorStrides`** - Stride information for vectorized operations
4. **`calculate_offset()`** - Computes actual memory offset (not just coordinate mapping)
5. **`get_element_space_size()`** - Returns total memory footprint

#### Example Use Case
```cpp
// Create a naive packed descriptor: [M=128, K=64]
auto desc = make_naive_tensor_descriptor_packed(
    make_tuple(number<128>{}, number<64>{})
);

// Get memory information
auto space_size = desc.get_element_space_size();  // 128 * 64 = 8192

// Calculate actual memory offset
auto offset = desc.calculate_offset(make_tuple(10, 20));  // Returns: 10*64 + 20 = 660

// Get vectorization info
auto vec_info = desc.get_top_dimension_safe_vector_length_strides();
```

---

## Inheritance Relationship

```
tensor_adaptor (Base Class)
    ↓
    │ Adds:
    │ - element_space_size_
    │ - GuaranteedVectorLengths
    │ - GuaranteedVectorStrides
    │ - calculate_offset()
    ↓
tensor_descriptor (Derived Class)
```

The descriptor **IS-A** adaptor (inheritance), meaning:
- Every `tensor_descriptor` can do everything a `tensor_adaptor` can do
- `tensor_descriptor` adds memory and vectorization metadata on top

---

## When to Use Which?

### Use `tensor_adaptor` When:
- ✅ You only need **coordinate transformation logic**
- ✅ Building **reusable transformation patterns**
- ✅ Composing transformations with `chain_tensor_adaptors()`
- ✅ Memory size is not relevant to your operation
- ✅ Working with intermediate transformation stages

### Use `tensor_descriptor` When:
- ✅ You need a **complete tensor specification**
- ✅ Calculating **actual memory offsets**
- ✅ Need to know **total memory footprint**
- ✅ Require **vectorization guarantees** for performance
- ✅ Creating tensors for actual data access (with `tensor_view`)
- ✅ Working with physical memory buffers

---

## Common Patterns

### Pattern 1: Building a Descriptor from an Adaptor

```cpp
// Step 1: Create adaptor with transformations
auto adaptor = make_single_stage_tensor_adaptor(
    transforms, lower_dims, upper_dims
);

// Step 2: Convert to descriptor by adding memory info
auto descriptor = make_tensor_descriptor_from_adaptor(
    adaptor, 
    element_space_size  // Add memory size
);
```

### Pattern 2: Transforming a Descriptor

```cpp
// Start with a descriptor
auto desc_original = make_naive_tensor_descriptor_packed(
    make_tuple(number<M>{}, number<K>{})
);

// Transform it (creates new descriptor)
auto desc_transformed = transform_tensor_descriptor(
    desc_original,
    new_transforms,
    lower_dim_ids,
    upper_dim_ids
);
// Result: New descriptor with updated transformations AND memory info
```

### Pattern 3: Naive Descriptor Creation

```cpp
// Packed layout (row-major, contiguous)
auto desc_packed = make_naive_tensor_descriptor_packed(
    make_tuple(number<M>{}, number<N>{})
);

// Custom strides
auto desc_strided = make_naive_tensor_descriptor(
    make_tuple(number<M>{}, number<N>{}),  // lengths
    make_tuple(number<N>{}, number<1>{})   // strides (row-major)
);

// With offset
auto desc_offset = make_naive_tensor_descriptor_with_offset(
    lengths, strides, offset
);
```

---

## Real-World Example: GEMM Tiling

### Using Adaptor (Transformation Only)
```cpp
// Define how to tile C matrix: [M, N] -> [M0, N0, M1, N1, M2, N2]
auto tiling_adaptor = make_single_stage_tensor_adaptor(
    make_tuple(
        make_unmerge_transform(make_tuple(M0, M1, M2)),
        make_unmerge_transform(make_tuple(N0, N1, N2))
    ),
    make_tuple(sequence<0>{}, sequence<1>{}),
    make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{})
);
// This adaptor can be reused for different matrix sizes
```

### Using Descriptor (Complete Specification)
```cpp
// Create actual C matrix descriptor with memory
auto C_desc = make_naive_tensor_descriptor_packed(
    make_tuple(number<256>{}, number<256>{})  // M=256, N=256
);

// Transform to tiled layout
auto C_tiled_desc = transform_tensor_descriptor(
    C_desc,
    tiling_transforms,
    lower_dims,
    upper_dims
);

// Now can calculate actual offsets
auto offset = C_tiled_desc.calculate_offset(tile_coords);
auto space = C_tiled_desc.get_element_space_size();  // 256*256 = 65536
```

---

## Coordinate Operations

### Creating Coordinates

Both adaptors and descriptors support creating coordinate objects that track positions in tensor space:

```cpp
// For adaptor
auto adaptor_coord = make_tensor_adaptor_coordinate(adaptor, idx_top);

// For descriptor (tensor_coordinate)
auto tensor_coord = make_tensor_coordinate(descriptor, idx_top);
auto offset = tensor_coord.get_offset();  // Get actual memory offset
```

### Moving Coordinates (Efficient Iteration)

A key operation for both is **`move_tensor_adaptor_coordinate()`** / **`move_tensor_coordinate()`**, which efficiently updates coordinates during iteration:

```cpp
// Move adaptor coordinate by a step
move_tensor_adaptor_coordinate(adaptor, coord, idx_diff_top);

// Move tensor coordinate by a step  
move_tensor_coordinate(descriptor, coord, coord_step);
```

**Why "move" instead of recalculating?**
- **Performance:** Moving is much faster than creating a new coordinate from scratch
- **Incremental updates:** Only recalculates transformations that are affected by the change
- **Optimization:** Uses `JudgeDoTransforms` template parameter to skip unnecessary calculations
- **Common use case:** Iterating through tiles in a window (e.g., sliding window operations)

**Example: Iterating through a tiled matrix**
```cpp
// Initial coordinate at [0, 0]
auto coord = make_tensor_coordinate(desc, make_tuple(0, 0));

// Move to next tile: [0, 0] -> [0, 1]
move_tensor_coordinate(desc, coord, make_tuple(0, 1));
// Much faster than: coord = make_tensor_coordinate(desc, make_tuple(0, 1));

// Move to next row: [0, 1] -> [1, 1]  
move_tensor_coordinate(desc, coord, make_tuple(1, 0));
```

**How it works:**
1. Updates the top-level (user-facing) indices
2. Propagates changes through transformation chain
3. Only recalculates affected transformations (optimization)
4. Updates all hidden intermediate indices
5. Computes new bottom index (memory offset)

This is heavily used in tile window operations where threads iterate through memory in a structured pattern.

---

## Key Transformations Supported

Both `tensor_adaptor` and `tensor_descriptor` support these coordinate transformations:

1. **`pass_through`** - Identity mapping (dimension unchanged)
2. **`pad`** - Add padding (left/right)
3. **`embed`** - Flatten multiple dimensions with strides
4. **`merge`** - Combine dimensions into one
5. **`unmerge`** - Split one dimension into multiple
6. **`replicate`** - Broadcast/repeat dimension
7. **`offset`** - Add constant offset

---

## Performance Considerations

### `tensor_adaptor`
- **Lightweight** - Only stores transformation logic
- **Zero runtime overhead** - All transformations compile-time when possible
- **Composable** - Can chain multiple adaptors efficiently

### `tensor_descriptor`
- **Additional metadata** - Stores memory size and vectorization info
- **Enables optimizations** - Vectorization guarantees help compiler
- **Memory bounds checking** - Can validate access patterns
- **Required for actual data access** - Used with `tensor_view` for real memory operations

---

## Summary

Think of it this way:

- **`tensor_adaptor`** = "How to transform coordinates" (the recipe)
- **`tensor_descriptor`** = "A complete tensor with memory" (the recipe + ingredients + kitchen)

The adaptor is the **transformation logic**, while the descriptor is a **complete tensor specification** that includes the transformation logic plus memory and performance metadata.

In practice:
- Use **adaptors** when designing reusable transformation patterns
- Use **descriptors** when working with actual tensors that need memory allocation and data access

---

## References

- **Source Files:**
  - `include/ck_tile/core/tensor/tensor_adaptor.hpp`
  - `include/ck_tile/core/tensor/tensor_descriptor.hpp`
  
- **Tutorial:**
  - `example/ck_tile/99_toy_example/tutorial_02_tensor_adaptors/tensor_adaptors.cpp`

- **Related Concepts:**
  - `tensor_view` - Combines descriptor with actual memory pointer
  - `tensor_coordinate` - Represents a position in tensor space
  - Coordinate transforms - The building blocks of adaptors/descriptors
