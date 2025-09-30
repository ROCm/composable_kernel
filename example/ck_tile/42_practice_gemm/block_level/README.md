# Block Pipeline Tutorial: Understanding AGmem BGmem CReg

This tutorial explains the `PracticeGemmBlockPipelineAGmemBGmemCreg` implementation, which orchestrates data movement and computation for block-level GEMM operations.

## Overview

The block pipeline manages:
- **A**Gmem: Matrix A loaded from Global memory
- **B**Gmem: Matrix B loaded from Global memory  
- **C**Reg: Result matrix C stored in Registers

## Key Concepts

### 1. Tensor Views: Memory with Structure

A **tensor view** combines:
- A memory pointer (where data lives)
- A descriptor (how to access the data) [See file [MakeALdsBlockDescriptor_README.md](./MakeALdsBlockDescriptor_README.md) for more details]
- Shape and strides are part of the descriptor

```cpp
// Creating tensor view for A in LDS
ADataType* p_a_lds = static_cast<ADataType*>(p_smem);
constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);
```

**What's happening:**
- `p_a_lds`: Raw pointer to LDS memory
- `a_lds_block_desc`: Describes the memory layout (dimensions, strides, etc.)
- `a_lds_block`: The complete view that knows both where and how to access data

### 2. Tile Windows: Subsections with Purpose

A **tile window** is a view into a portion of a tensor with:
- The underlying tensor view
- Window dimensions (what subset to access)
- Origin point (where the window starts)
- Tile distribution (how threads map to elements)

```cpp
// Window for loading A from DRAM
auto a_copy_dram_window = make_tile_window(
    a_dram_block_window_tmp.get_bottom_tensor_view(),  // Tensor view extrated from existing tensor view
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),  // Window size
    a_dram_block_window_tmp.get_window_origin(),  // Starting position
    Policy::template MakeADramTileDistribution<Problem>()  // Thread mapping
);
```

### 3. Tile Distributions: Thread-to-Data Mapping

**Tile distribution** defines how each thread in a block accesses elements within a tile window.

## The Pipeline Architecture

### Step 1: Setting Up LDS Memory

```cpp
// Allocate LDS for matrix A
ADataType* p_a_lds = static_cast<ADataType*>(p_smem);
constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

// Allocate LDS for matrix B (after A's space)
BDataType* p_b_lds = static_cast<BDataType*>(
    static_cast<void*>(static_cast<char*>(p_smem) + some_offset_if_needed));
constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);
```

**Key points:**
- LDS memory is shared within a block
- B can be placed after A in the same shared memory by providing the offset

### Step 2: Creating Data Movement Windows

For each matrix, we need four windows:

#### 2.1 DRAM Load Windows
```cpp
// Window to load A from global memory to thread local registers
auto a_copy_dram_window = make_tile_window(
    a_dram_block_window_tmp.get_bottom_tensor_view(),
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    a_dram_block_window_tmp.get_window_origin(),
    Policy::template MakeADramTileDistribution<Problem>()
);
```

#### 2.2 LDS Store Windows
```cpp
// Window to store A into LDS from thread local registers
auto a_copy_lds_window = make_tile_window(
    a_lds_block,
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    {0, 0},  // Always start at LDS origin
    a_copy_dram_window.get_tile_distribution()  // Reuse DRAM distribution
);
```

#### 2.3 LDS GEMM Windows
```cpp
// Window to read A from LDS for GEMM computation again from LDS to thread local registers
auto a_lds_gemm_window = make_tile_window(
    a_lds_block, 
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), 
    {0, 0}  // No explicit distribution - uses GEMM policy
);
```

### Step 3: The Main Loop

```cpp
// Initialize result to zero
tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

while(iCounter > 0)
{
    // 1. Load tiles from DRAM to registers
    a_block_tile = load_tile(a_copy_dram_window);
    b_block_tile = load_tile(b_copy_dram_window);
    
    // 2. Advance windows for next iteration
    move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
    move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
    
    // 3. Store tiles from registers to LDS
    store_tile(a_copy_lds_window, a_block_tile);
    store_tile(b_copy_lds_window, b_block_tile);
    
    // 4. Synchronize to ensure all threads have stored their data
    block_sync_lds();
    
    // 5. Perform GEMM computation using LDS data again from LDS to thread local registers
    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
    
    // 6. Synchronize before next iteration
    block_sync_lds();
    
    iCounter--;
}
```

## Data Flow Visualization

```
┌─────────────────┐                       ┌─────────────────┐
│   DRAM (Global) │                       │  DRAM Windows   │
│   A[M][K]       │──────────────────────▶│a_copy_dram_window│
│   B[N][K]       │                       │b_copy_dram_window│
└─────────────────┘                       └────────┬────────┘
                                                   │ load_tile()
                                                   ▼
                                          ┌─────────────────┐
                                          │    Registers    │
                                          │  a_block_tile   │
                                          │  b_block_tile   │
                                          └────────┬────────┘
                                                   │ store_tile()
                                                   ▼
┌─────────────────┐                       ┌─────────────────┐
│   LDS (Shared)  │                       │   LDS Windows   │
│  a_lds_block    │◀──────────────────────│a_copy_lds_window│
│  b_lds_block    │                       │b_copy_lds_window│
└─────────────────┘                       └─────────────────┘
         │                                          
         │                                ┌─────────────────┐
         └───────────────────────────────▶│   LDS Windows   │
                                          │a_lds_gemm_window│
                                          │b_lds_gemm_window│
                                          └────────┬────────┘
                                                   │ block_gemm()
                                                   ▼
                                          ┌─────────────────┐
                                          │    Registers    │
                                          │  c_block_tile   │
                                          │   (Result)      │
                                          └─────────────────┘
```

## Window Movement Strategy

The pipeline processes the K dimension in chunks:

```cpp
// Step size for moving windows along K dimension
constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, KPerBlock);
constexpr BDramTileWindowStep b_dram_tile_window_step = make_array(0, KPerBlock);
```

For a GEMM operation C = A × B^T:
- A window moves right (along K): `[M, K] → [M, K+KPerBlock]`
- B window moves right (along K): `[N, K] → [N, K+KPerBlock]`
