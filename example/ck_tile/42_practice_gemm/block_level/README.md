# Block-Level GEMM Implementation

This directory contains the block-level implementation of the practice GEMM kernel. At this level, we focus on managing large tiles that fit within a thread block's compute unit (CU).

## Key Concepts

### 1. **Block Tiles**

Block tiles represent the largest unit of work that a single thread block can process:

```cpp
using BlockTile = ck_tile::sequence<256, 128, 32>;  // M, N, K dimensions
```

This means:
- **256 rows (M)**: Vertical dimension processed by the block
- **128 columns (N)**: Horizontal output dimension
- **32 depth (K)**: Inner dimension for matrix multiplication

![Block Tile](./images/block_tile.png)

The block tile size is constrained by:
- **Compute resources**: Register file size, shared memory capacity
- **Memory bandwidth**: LDS and global memory access patterns
- **Occupancy**: Number of blocks that can run simultaneously on a CU

### 2. **Memory Hierarchies in Block GEMM**

The block-level GEMM manages data movement across three memory levels:

```
Global Memory (DRAM)
├── A Matrix Blocks (256×32)
├── B Matrix Blocks (128×32)
└── C Matrix Blocks (256×128)

Shared Memory (LDS)
├── A Block Storage (256×32)
└── B Block Storage (128×32)

Register Memory
└── C Block Accumulation (256×128)
```

### 3. **LDS Memory Layout**

Shared memory (LDS) stores A and B blocks with optimized layouts:

#### A Matrix in LDS
```cpp
// Original layout: make_tuple(MPerBlock, KPerBlock)
// Packed layout: make_tuple(MPerBlock, KPerBlock/KPack, KPack)
// Strided layout: make_tuple(KPerBlock, KPack, 1)
```

This packing improves:
- **Memory coalescing**: Better global memory access patterns
- **Bank conflict reduction**: Optimized LDS access patterns
- **Vectorization**: Enables wider load/store operations

#### B Matrix in LDS
```cpp
// Similar packing strategy for B matrix
// Layout: make_tuple(NPerBlock, KPerBlock/KPack, KPack)
```

### 4. **Tile Distributions**

Tile distributions define how work is divided among threads within a block:

#### A Matrix Distribution
```cpp
tile_distribution_encoding<sequence<1>,
                          tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                          tuple<sequence<1>, sequence<1, 2>>,
                          tuple<sequence<1>, sequence<2, 0>>,
                          sequence<1, 2>,
                          sequence<0, 1>>{}
```

**Distribution Parameters:**
- **M0, M1, M2**: Hierarchical split along M dimension
  - M0: Wave iterations (temporal)
  - M1: Number of waves (spatial)
  - M2: Threads per wave (spatial)
- **K0, K1**: Split along K dimension
  - K0: Thread groups along K
  - K1: Vector size per thread

#### B Matrix Distribution
Similar structure but optimized for B matrix access patterns.

### 5. **Block GEMM Pipeline**

The main pipeline manages the complete GEMM computation:

```cpp
// 1. Load A and B blocks from DRAM to LDS
a_block_tile = load_tile(a_copy_dram_window);
b_block_tile = load_tile(b_copy_dram_window);

// 2. Store blocks to LDS with proper layout
store_tile(a_copy_lds_window, a_block_tile);
store_tile(b_copy_lds_window, b_block_tile);

// 3. Synchronize LDS access
block_sync_lds();

// 4. Perform block GEMM computation
block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

// 5. Repeat for multiple K iterations
```

### 6. **Window Management**

Windows provide views into tensors with specific distributions:

#### DRAM Windows
```cpp
// A DRAM window: 256×32 elements with distribution for coalesced loading
auto a_copy_dram_window = make_tile_window(
    a_dram_block_window_tmp.get_bottom_tensor_view(),
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    a_dram_block_window_tmp.get_window_origin(),
    Policy::template MakeADramTileDistribution<Problem>());

// B DRAM window: 128×32 elements with distribution for coalesced loading
auto b_copy_dram_window = make_tile_window(/* similar for B */);
```

#### LDS Windows
```cpp
// A LDS window for storing loaded data
auto a_copy_lds_window = make_tile_window(
    a_lds_block,
    make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
    {0, 0},
    a_copy_dram_window.get_tile_distribution());

// B LDS window for storing loaded data
auto b_copy_lds_window = /* similar for B */;
```

### 7. **Memory Access Patterns**

#### Coalesced Loading
- **A Matrix**: Threads load consecutive K elements (coalesced along K)
- **B Matrix**: Threads load consecutive K elements (coalesced along K)
- **Vector size**: 8 elements per thread for better memory bandwidth

#### LDS Access Optimization
- **A Matrix**: M-major access pattern (rows first)
- **B Matrix**: N-major access pattern (columns first)
- **Bank conflicts**: Minimized through careful layout selection

### 8. **Performance Considerations**

#### Memory Bandwidth
- **Global Memory**: Coalesced 8-element vector loads
- **Shared Memory**: Conflict-free LDS access patterns
- **Register Usage**: Optimized for MFMA instruction requirements

#### Compute Efficiency
- **Wave Utilization**: 4 waves per block for good occupancy
- **MFMA Instructions**: 16×16×16 matrix operations at warp level
- **Data Reuse**: K-dimension looping maximizes LDS data reuse

#### Scalability
- **Block Size**: 256 threads (4 waves × 64 threads/wave)
- **Occupancy**: Designed for high CU utilization
- **Memory Pressure**: Balanced LDS and register usage

## Implementation Files

- **`practice_gemm_block_policy_agmem_bgmem_creg.hpp`**
  - Defines block-level policy and configurations
  - Contains LDS descriptor creation and tile distributions
  - Manages memory layouts and access patterns

- **`practice_gemm_block_pipeline_agmem_bgmem_creg.hpp`**
  - Implements the main block GEMM pipeline
  - Manages data movement between memory hierarchies
  - Coordinates block-level computation

This block-level implementation provides the foundation for efficient GEMM computation, balancing memory bandwidth, compute efficiency, and scalability across different GPU architectures.
