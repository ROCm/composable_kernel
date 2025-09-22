# Warp-Level GEMM Implementation

This directory contains the warp-level implementation of the practice GEMM kernel. At this level, we focus on the fundamental compute units: **Matrix Fused Multiply-Add (MFMA)** operations performed by individual waves.

## Key Concepts

### 1. **Warp Tiles**

Warp tiles represent the smallest unit of computation that fits within a single wave (64 threads on AMD GPUs):

```cpp
using WaveTile = ck_tile::sequence<16, 16, 16>;  // M, N, K dimensions
```

This means each wave processes:
- **16 rows (M)**: Vertical dimension per MFMA operation
- **16 columns (N)**: Horizontal output dimension per MFMA operation
- **16 depth (K)**: Inner dimension per MFMA operation

### 2. **MFMA Instructions**

The core computation uses AMD's Matrix Fused Multiply-Add instructions:

```cpp
WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution
```

**Key characteristics:**
- **Input types**: F16 × F16 (A and B matrices)
- **Output type**: F32 (accumulation)
- **Matrix size**: 32×32×8 (M×N×K)
- **Transpose C**: Optimized memory layout for accumulation

### 3. **Wave Configuration**

Each thread block contains multiple waves working together:

```cpp
static constexpr index_t kMWarp = 4;  // Waves along M dimension
static constexpr index_t kNWarp = 1;  // Waves along N dimension
```

This configuration creates:
- **4 waves along M**: Each wave handles different M tiles
- **1 wave along N**: Single wave handles N dimension
- **Total block size**: 256 threads (4 × 64)

### 4. **Warp GEMM Pipeline**

The warp-level pipeline manages register-based computation:

```cpp
// Load A and B warp tiles from LDS
AWarpTensor a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));
BWarpTensor b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

// Load C warp tile from register accumulation
CWarpTensor c_warp_tensor;
c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(/*...*/);

// Perform MFMA computation
WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

// Store result back to accumulation registers
c_block_tensor.set_y_sliced_thread_data(/*...*/, c_warp_tensor.get_thread_buffer());
```

### 5. **Data Movement Patterns**

#### Register Loading
- **A Matrix**: Each thread loads 16 elements (16×1 vector)
- **B Matrix**: Each thread loads 16 elements (16×1 vector)
- **C Matrix**: Each thread loads 16 elements (16×1 vector)

#### Memory Layouts
- **A Matrix**: M-major (row-major) access pattern
- **B Matrix**: N-major (column-major) access pattern
- **C Matrix**: Optimized for accumulation reuse

### 6. **Thread Distribution within Warp**

Each wave (64 threads) processes the 16×16×16 tile as follows:

```cpp
// Thread distribution for 16×16×16 MFMA
// 64 threads = 16 rows × 4 columns (of 16-element vectors)
constexpr index_t MPerWarp = 16;  // Rows per MFMA
constexpr index_t NPerWarp = 16;  // Columns per MFMA
constexpr index_t KPerWarp = 16;  // Depth per MFMA
```

**Thread mapping:**
- **M dimension**: 16 threads (one per row)
- **N dimension**: 4 threads (4 columns of 16-element vectors)
- **Total threads**: 64 threads per wave

### 7. **Warp Iterations**

Multiple MFMA operations are needed to cover the block tile:

```cpp
// BlockTile: 256×128×32
// WaveTile:  16×16×16
//
// Iterations needed:
// M direction: 256 / (4 waves × 16) = 4 iterations
// N direction: 128 / (1 wave × 16) = 8 iterations
// K direction: 32 / 16 = 2 iterations

constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);  // 4
constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);  // 8
constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;           // 2
```

### 8. **Register Memory Management**

#### Input Registers
- **A Matrix**: Each thread holds 16×2 = 32 F16 values (double-buffered)
- **B Matrix**: Each thread holds 16×2 = 32 F16 values (double-buffered)
- **Total input**: ~4KB of register space per thread

#### Output Registers
- **C Matrix**: Each thread holds 16×2 = 32 F32 values (accumulation)
- **Additional overhead**: Loop indices, pointers, etc.

#### Register Pressure
- **Total registers**: ~160 registers per thread
- **Occupancy**: 4 waves per block (256 threads)
- **Memory bound**: Limited by LDS bandwidth and capacity

### 9. **Performance Characteristics**

#### Compute Efficiency
- **MFMA utilization**: 32×32×8 operations per instruction
- **Arithmetic intensity**: High due to data reuse in registers
- **Instruction mix**: ~80% MFMA, ~20% load/store

#### Memory Bandwidth
- **LDS bandwidth**: 16 bytes/cycle per CU (coalesced access)
- **Register bandwidth**: High internal bandwidth
- **Data reuse**: K-dimension looping maximizes LDS efficiency

#### Scalability
- **Wave parallelism**: 4 waves per block for good occupancy
- **Block parallelism**: Multiple blocks per CU
- **CU utilization**: Balanced compute and memory operations

## Implementation Files

- **`practice_gemm_warp_policy_asmem_bsmem_creg.hpp`**
  - Defines warp-level policy and MFMA configuration
  - Specifies wave dimensions (MWarp, NWarp)
  - Selects appropriate MFMA instruction variant

- **`practice_gemm_warp_pipeline_asmem_bsmem_creg.hpp`**
  - Implements warp-level GEMM pipeline
  - Manages register-based MFMA computation
  - Coordinates data movement within waves

## Key Insights

1. **MFMA Granularity**: The 16×16×16 tile size is fundamental to MFMA efficiency
2. **Data Layout**: Memory layouts are optimized for MFMA instruction requirements
3. **Thread Cooperation**: All 64 threads within a wave participate in each MFMA operation
4. **Register Management**: Careful register allocation is critical for performance

This warp-level implementation provides the foundation for efficient matrix multiplication, leveraging AMD's MFMA instructions for maximum compute throughput.
