# LDS Bank Conflicts: Theory, Measurement, and Optimization

## Table of Contents
1. [Introduction](#introduction)
2. [Part 1: Understanding Bank Conflicts as a Constraint Satisfaction Problem](#part-1-understanding-bank-conflicts-as-a-constraint-satisfaction-problem)
3. [Part 2: Measuring Bank Conflicts with rocprofv3](#part-2-measuring-bank-conflicts-with-rocprofv3)
4. [Part 3: Bank Conflict Patterns in Transpose](#part-3-bank-conflict-patterns-in-transpose)
5. [Part 4: XOR Swizzling Solution](#part-4-xor-swizzling-solution)
6. [Part 5: Limitations and Alternative Solutions](#part-5-limitations-and-alternative-solutions)
7. [Hands-On Exercises](#hands-on-exercises)
8. [Appendix: CK Tile API Reference](#appendix-ck-tile-api-reference)

---

## Introduction

### What This Tutorial Covers

This tutorial provides a comprehensive understanding of **LDS (Local Data Share) bank conflicts** on AMD GPUs, specifically the MI300 series. You'll learn:

- What bank conflicts are and why they matter for GPU performance
- How to measure bank conflicts using AMD profiling tools
- How bank conflicts arise in the transpose operation
- How to reduce bank conflicts using XOR swizzling in the CK Tile API
- The mathematical limits of bank conflict elimination
- Trade-offs between different optimization approaches

### Prerequisites

- Basic GPU programming knowledge (threads, blocks, warps/wavefronts)
- Understanding of shared memory concepts
- Familiarity with C++ templates
- **Recommended:** Tutorial 08 (LDS staging) and Tutorial 09 (optimized LDS)

### Learning Objectives

By the end of this tutorial, you will be able to:
- Explain bank conflicts as a constraint satisfaction problem
- Profile GPU kernels to identify bank conflict bottlenecks
- Implement XOR swizzling using CK Tile descriptors
- Analyze trade-offs between different bank conflict mitigation strategies
- Understand the theoretical limits of optimization

---

## Part 1: Understanding Bank Conflicts as a Constraint Satisfaction Problem

### LDS Memory Basics

**Local Data Share (LDS)** is AMD's on-chip shared memory, analogous to CUDA's shared memory:

- **Speed**: ~100× faster than global memory (HBM)
- **Capacity**: 64 KB per compute unit on MI300
- **Access**: Shared among all threads in a workgroup (block)
- **Architecture**: 32 banks × 4 bytes = 128 bytes per cycle bandwidth
- **Ports**: Bidirectional ports allow concurrent reads/writes to different banks
- **Phase-based access**: Hardware divides wavefront access into phases (instruction-dependent)

### Bank Architecture Analogy

Think of LDS banks like lanes at a bank or post office:

```
32 Banks (like 32 service windows):
┌────┐ ┌────┐ ┌────┐       ┌────┐
│ 0  │ │ 1  │ │ 2  │  ...  │ 31 │
└────┘ └────┘ └────┘       └────┘

32 Threads (like 32 customers):
Thread 0 → Bank 0  ✓ (served immediately)
Thread 1 → Bank 1  ✓ (served immediately)
Thread 2 → Bank 2  ✓ (served immediately)
...
Thread 31 → Bank 31 ✓ (served immediately)

Result: All 32 threads served in 1 cycle!
```

**What happens with conflicts?**

```
32 Threads, but all want Bank 0:
Thread 0 → Bank 0  ✓ (cycle 1)
Thread 1 → Bank 0  ✗ (wait... cycle 2)
Thread 2 → Bank 0  ✗ (wait... cycle 3)
...
Thread 31 → Bank 0 ✗ (wait... cycle 32)

Result: 32 cycles instead of 1 (32× slower!)
```

### Framing as a Constraint Satisfaction Problem (CSP)

Bank conflict optimization is fundamentally a **constraint satisfaction problem**. We have conflicting requirements that cannot all be perfectly satisfied simultaneously.

#### Constraint 1: Hardware Architecture (Fixed)

```
- Number of banks: 32
- Bank assignment: bank_id = (byte_address / 4) % 32
- Each thread accesses ONE bank per LDS instruction
- Multiple threads → same bank = conflict (serialization)
- Bidirectional ports: Reads and writes can happen simultaneously to different banks
```

**We cannot change this** - it's built into the hardware.

#### Constraint 1a: Phase Grouping Asymmetry (CRITICAL - Fixed by Hardware)

**This is a fundamental architectural constraint that makes bank conflict optimization challenging!**

The hardware divides LDS access from a wavefront into **phases**, with **different groupings for read vs write instructions**:

**For `ds_write_b128` (64 lanes, 8 phases):**
```
Sequential grouping:
  Phase 0: lanes 0-7
  Phase 1: lanes 8-15
  Phase 2: lanes 16-23
  Phase 3: lanes 24-31
  Phase 4: lanes 32-39
  Phase 5: lanes 40-47
  Phase 6: lanes 48-55
  Phase 7: lanes 56-63

Conflict detection: Within each sequential 8-lane group
```

**For `ds_read_b128` (64 lanes, 8 phases):**
```
Non-sequential grouping:
  Phase 0: lanes 0-3 + 20-23
  Phase 1: lanes 4-7 + 16-19
  Phase 2: lanes 8-11 + 28-31
  Phase 3: lanes 12-15 + 24-27
  Phase 4: lanes 32-35 + 52-55
  Phase 5: lanes 36-39 + 48-51
  Phase 6: lanes 40-43 + 60-63
  Phase 7: lanes 44-47 + 56-59

Conflict detection: Within each non-sequential phase group
```

**Why this asymmetry exists:**

From AMD's official documentation:
> "In most LDS access the MFMA instruction in the next step requires vertical data access patterns."

The non-sequential read phase grouping is designed to match how **MFMA (Matrix Fused Multiply-Add) instructions** consume data - they need vertical/column access, not horizontal/row access.

**The critical implication:**

**A memory layout that is conflict-free for WRITES may have conflicts for READS, and vice versa!**

Example:
- Row-major layout: Write is conflict-free (sequential phases match sequential addresses)
- Row-major layout: Read has **2-4 way conflicts** (non-sequential phases hit same banks)

This is why naive layouts fail and why XOR swizzling is necessary - it must optimize for **BOTH** phase patterns simultaneously!

**We cannot change this** - the phase grouping is hardwired into the GPU microarchitecture.

#### Constraint 2: Access Pattern (Fixed by Algorithm)

For matrix transpose, we must:
- **Write phase**: Store [M, K] in row-major order
- **Read phase**: Read [K, M] (transposed = column-major view)

```
Write: Row-major access
offset = m * K + k
Stride-1 in K dimension → works well with sequential write phases

Read: Column-major access (transpose)
offset = k * 1 + m * K
Stride-K in M dimension → conflicts with non-sequential read phases!
```

**We cannot change the algorithm** - transpose requires reading columns from row-major data.

**Combined with Constraint 1a**: The algorithm's column-read pattern conflicts with the non-sequential read phase grouping, creating severe bank conflicts.

#### Constraint 3: Parallelism Requirements

```
Current tile: [64, 32]
Reading column: 64 threads access simultaneously
Problem: 64 threads > 32 banks

Pigeonhole principle: 64 threads, 32 banks
→ Minimum 2 threads per bank (best case)
→ Conflicts are UNAVOIDABLE with 64 threads!

Furthermore: 64 threads with non-sequential read phases
→ Some phases will have multiple threads hitting same banks
→ Even with perfect distribution, phase grouping creates conflicts
```

**We can change this** by using smaller tiles, but at a performance cost.

### Solution Space

Given these constraints, what can we optimize?

| Solution | Modifies Constraint | Trade-off |
|----------|-------------------|-----------|
| **XOR swizzling** | 2 (partially) - permute physical addresses | ✓ No algorithm change, ✓ Simple, ✓ 57% improvement |
| **32×32 tiles** | 3 - reduce threads | ✓ Can achieve zero conflicts, ✗ Lower throughput |
| **Padding** | 2 (partially) - change stride | ✓ May help, ✗ Wastes LDS, ✗ Complex |
| **Double buffering** | 2 - separate read/write | ✓ Flexible, ✗ 2× LDS usage, ✗ Complexity |
| Add more banks | 1 - hardware | ✗ Impossible |
| Change algorithm | 2 - algorithm | ✗ Not transpose anymore |

**This tutorial focuses on XOR swizzling** - the best practical solution within existing constraints.

### Simple Example: Conflict vs No Conflict

```cpp
__shared__ float data[32];
int tid = threadIdx.x;  // 0..31

// GOOD: No conflicts
data[tid] = ...;
// Thread 0 → byte 0  → bank 0
// Thread 1 → byte 4  → bank 1
// Thread 2 → byte 8  → bank 2
// ...
// Thread 31 → byte 124 → bank 31
// All 32 threads served in parallel! ✓

// BAD: 32-way conflict
data[0] = ...;
// ALL threads → byte 0 → bank 0
// Must serialize: 32 cycles instead of 1! ✗

// MODERATE: Transpose problem (FP16, stride-32)
__shared__ half_t matrix[64][32];
// Read column: matrix[tid][0] where tid = 0..63
// Thread 0:  byte 0   → bank 0
// Thread 1:  byte 64  → bank 16
// Thread 2:  byte 128 → bank 0  ← conflict with thread 0!
// Thread 3:  byte 192 → bank 16 ← conflict with thread 1!
// ...
// Pattern: Only 2 banks used (0 and 16) → 32-way conflicts on each!
```

### Key CSP Insight

**We cannot eliminate all constraints simultaneously.**

The XOR swizzling technique we'll learn:
- ✓ Keeps hardware constraints (Constraint 1)
- ✓ Keeps transpose algorithm (Constraint 2)
- ✓ Keeps parallelism level (Constraint 3)
- ✓ Optimizes WITHIN the solution space

It achieves **57% conflict reduction** - a significant practical improvement without changing the fundamental problem structure.

---

## Part 2: Measuring Bank Conflicts with rocprofv3

### AMD GPU Performance Counters

AMD GPUs have hardware performance monitoring counters (PMC) that track various metrics:

**Key counters for bank conflicts:**
- `SQ_LDS_BANK_CONFLICT` - Total number of bank conflicts
- `SQ_INSTS_LDS` - Total number of LDS instructions executed

**Conflict rate formula:**
```
Conflict Rate (%) = (SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS) × 100%
```

### Using rocprofv3 Tool

**Basic profiling command:**
```bash
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/profile_output \
          -- ./bin/your_program
```

**Query results from SQLite database:**
```bash
sqlite3 /tmp/profile_output/*/results.db "
SELECT
    SUM(CASE WHEN counter_name = 'SQ_LDS_BANK_CONFLICT' THEN counter_value ELSE 0 END) as conflicts,
    SUM(CASE WHEN counter_name = 'SQ_INSTS_LDS' THEN counter_value ELSE 0 END) as lds_insts,
    ROUND(100.0 * conflicts / lds_insts, 2) as conflict_rate_percent
FROM pmc_events;"
```

### Hands-On Example: Plain Transpose Results

Let's profile the plain transpose (Tutorial 11, `xor_test_plain_only.cpp`):

**Configuration:**
```
Matrix: [256, 128] → [128, 256] transpose
Element type: FP16 (half_t)
Tile size: [64, 32]
Block size: 256 threads
```

**Profile command:**
```bash
cd relbuild
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/plain \
          -- ./bin/aa_tutorial_11_plain_transpose
```

**Expected Results:**
```
SQ_LDS_BANK_CONFLICT: 7,168
SQ_INSTS_LDS:         608
Conflict Rate:        1,244%
```

### Understanding the Numbers

#### Why >100% Conflict Rate?

A conflict rate over 100% means that **on average, each LDS instruction has multiple conflicts**.

**Calculation:**
```
Conflicts per instruction = 7,168 / 608 ≈ 11.8

This means:
- Each LDS instruction serializes ~12 times
- Instead of 1 cycle → takes ~12 cycles
- Effective bandwidth: 128 bytes/cycle → ~10.7 bytes/cycle
- Performance: 12× slower than ideal!
```

#### What Does 12-Way Conflict Mean?

```
Ideal case (no conflicts):
┌──────────────────────────────────┐
│ Cycle 1: All 32 threads complete │
└──────────────────────────────────┘

12-way conflicts:
┌──────────────────────────────────┐
│ Cycle 1:  ~3 threads complete    │
│ Cycle 2:  ~3 threads complete    │
│ Cycle 3:  ~3 threads complete    │
│ ...                              │
│ Cycle 12: ~3 threads complete    │
└──────────────────────────────────┘

Result: 12× more cycles needed!
```

#### Serialization Penalty Example

Consider reading a column from our [64, 32] LDS tile:

```
Without conflicts:
- 64 threads read 64 elements
- With perfect distribution: 2 threads per bank (theoretical minimum)
- Cycles needed: 2

With severe conflicts (plain LDS):
- 64 threads read 64 elements
- Most threads hit same 2 banks (bank 0 and bank 16)
- ~32 threads per bank
- Cycles needed: 32

Slowdown: 32 / 2 = 16× slower than theoretical optimal!
```

### CSP Perspective: Measuring Constraint Violations

From the CSP viewpoint:

```
Bank conflicts = violations of "one thread per bank" constraint
Conflict rate = severity of constraint violations

Baseline (plain LDS):
- Conflict rate: 1,244%
- Severity: SEVERE violations (~12 threads per bank)

Target (after XOR):
- Conflict rate: ~533%
- Severity: MODERATE violations (~5 threads per bank)

Theoretical minimum (64 threads, 32 banks):
- Conflict rate: ~100%
- Severity: MINIMAL violations (2 threads per bank)
```

**Interpretation:**
- 100% conflict rate is the BEST POSSIBLE with 64 threads and 32 banks
- Any rate above 100% indicates suboptimal distribution across banks
- Our goal: Get as close to 100% as possible

---

## Part 3: Bank Conflict Patterns in Transpose

### Stride Patterns and Bank Mapping

**General bank assignment formula:**
```
bank_id = (byte_address / 4) % 32
```

**For array access: `array[index]`**
```
byte_address = base_address + index × sizeof(element)
bank_id = ((index × sizeof(element)) / 4) % 32
```

### Common Access Patterns (FP16, 2 bytes)

```cpp
__shared__ half_t data[128];
int tid = threadIdx.x;  // 0..31

// Stride-1: MINIMAL conflicts (2-way for FP16)
data[tid]
// Thread 0:  byte 0  → bank (0/4)   % 32 = 0
// Thread 1:  byte 2  → bank (2/4)   % 32 = 0  ← 2 threads per bank (FP16)
// Thread 2:  byte 4  → bank (4/4)   % 32 = 1
// Thread 3:  byte 6  → bank (6/4)   % 32 = 1
// ...
// Pattern: 2 threads per bank (optimal for FP16!)

// Stride-16: MODERATE conflicts
data[tid * 16]
// Thread 0:  byte 0   → bank (0/4)   % 32 = 0
// Thread 1:  byte 32  → bank (32/4)  % 32 = 8
// Thread 2:  byte 64  → bank (64/4)  % 32 = 16
// Thread 3:  byte 96  → bank (96/4)  % 32 = 24
// Thread 4:  byte 128 → bank (128/4) % 32 = 0  ← conflict!
// Pattern: 8-way conflicts

// Stride-32: SEVERE conflicts (transpose case!)
data[tid * 32]  // tid = 0..63
// Thread 0:  byte 0   → bank (0/4)   % 32 = 0
// Thread 1:  byte 64  → bank (64/4)  % 32 = 16
// Thread 2:  byte 128 → bank (128/4) % 32 = 0  ← conflict!
// Thread 3:  byte 192 → bank (192/4) % 32 = 16 ← conflict!
// Thread 4:  byte 256 → bank (256/4) % 32 = 0  ← conflict!
// ...
// Pattern: Only 2 banks used (0, 16) → 32-way conflicts!
```

### Understanding Phase-Based LDS Access

Before diving into the transpose problem, we need to understand how AMD GPUs actually execute LDS instructions at the hardware level.

#### Hardware Phase Division

**Key insight:** The GPU doesn't execute all 64 lanes of a wavefront simultaneously for LDS access. Instead, it divides them into **phases** and processes each phase sequentially.

**Why phases exist:**
- Physical limitation: Not all 64 lanes can access LDS in the exact same clock cycle
- Instruction width: `ds_read_b128` and `ds_write_b128` specify how much data each lane accesses (128 bits = 16 bytes)
- Bank arbitration: The hardware checks for conflicts within each phase

#### ds_write_b128 Phase Pattern

For a write instruction with 64 lanes:

```
Phase 0: lanes 0, 1, 2, 3, 4, 5, 6, 7       (8 lanes)
Phase 1: lanes 8, 9, 10, 11, 12, 13, 14, 15
Phase 2: lanes 16, 17, 18, 19, 20, 21, 22, 23
Phase 3: lanes 24, 25, 26, 27, 28, 29, 30, 31
Phase 4: lanes 32, 33, 34, 35, 36, 37, 38, 39
Phase 5: lanes 40, 41, 42, 43, 44, 45, 46, 47
Phase 6: lanes 48, 49, 50, 51, 52, 53, 54, 55
Phase 7: lanes 56, 57, 58, 59, 60, 61, 62, 63

Pattern: Sequential grouping
Conflict check: Within each 8-lane sequential group
```

**When writing to consecutive addresses:**
- Each phase's 8 lanes access different banks
- No conflicts within each phase
- Result: **Conflict-free write** for row-major data

#### ds_read_b128 Phase Pattern

**Here's where it gets interesting** - reads use a **completely different** phase grouping:

```
Phase 0: lanes 0, 1, 2, 3,  20, 21, 22, 23      (8 lanes, non-sequential!)
Phase 1: lanes 4, 5, 6, 7,  16, 17, 18, 19
Phase 2: lanes 8, 9, 10, 11, 28, 29, 30, 31
Phase 3: lanes 12, 13, 14, 15, 24, 25, 26, 27
Phase 4: lanes 32, 33, 34, 35, 52, 53, 54, 55
Phase 5: lanes 36, 37, 38, 39, 48, 49, 50, 51
Phase 6: lanes 40, 41, 42, 43, 60, 61, 62, 63
Phase 7: lanes 44, 45, 46, 47, 56, 57, 58, 59

Pattern: Non-sequential, paired grouping
Conflict check: Within each non-sequential phase group
```

**Why this strange grouping?**

From AMD documentation:
> "In most LDS access the MFMA instruction in the next step requires vertical data access patterns."

The read phase grouping is designed to match how **MFMA (Matrix Fused Multiply-Add)** instructions consume data. MFMA needs vertical/column data, so the read phases are organized to facilitate this access pattern efficiently.

#### The Critical Problem

**A layout that works for write phases may fail for read phases!**

Example with row-major [64, 32] matrix:

**Write (sequential phases):**
```
Writing row 0: lanes 0-7 write elements [0][0-7]
  → Consecutive addresses
  → Phase 0 accesses 8 different banks
  → No conflicts! ✓
```

**Read (non-sequential phases):**
```
Reading column 0:
  Phase 0 includes lanes 0-3 and 20-23

  lanes 0-3:    read [0][0], [1][0], [2][0], [3][0]
  lanes 20-23:  read [20][0], [21][0], [22][0], [23][0]

  → All in Phase 0, but accessing strided addresses (stride = 64 bytes)
  → Multiple lanes in Phase 0 hit the same banks
  → 4-way conflicts! ✗
```

**This is why transpose is hard:**
- Write uses sequential phases → works great with row-major
- Read uses non-sequential phases → conflicts with column access from row-major
- XOR swizzling must optimize for **BOTH** phase patterns!

---

### The Transpose Problem: Detailed Analysis

Now that we understand phase-based access, let's analyze the transpose problem in detail.

**Configuration:**
```
Matrix: [M, K] = [64, 32]
Storage: Row-major in LDS
Element type: FP16 (2 bytes)
LDS size: 64 × 32 × 2 = 4,096 bytes
Wavefront: 64 lanes
Write instruction: ds_write_b128
Read instruction: ds_read_b128
```

#### Phase 1: Write [M, K] - Row-Major (GOOD for Write Phases)

```cpp
// Store to LDS in row-major order
lds[m][k] = ...;  // offset = m * K + k

// Thread accessing K-dimension sequentially:
lds[0][0], lds[0][1], lds[0][2], ...

// Byte offsets: 0, 2, 4, 6, 8, 10, ...
// Banks: 0, 0, 1, 1, 2, 2, 3, 3, ...

// Result: Stride-1 in K → minimal conflicts (2-way for FP16)
// ✓ Good access pattern!
```

#### Phase 2: Read [K, M] - Column-Major (BAD)

```cpp
// Read transposed: accessing column from row-major data
lds[m][0]  // where m ∈ [0, 64)

// Physical layout (row-major):
lds[0][0]  at byte 0
lds[1][0]  at byte 64   (skip entire row: 32 FP16 = 64 bytes)
lds[2][0]  at byte 128
lds[3][0]  at byte 192
...

// Bank mapping:
lds[0][0]:  byte 0   → bank (0/4)    % 32 = 0
lds[1][0]:  byte 64  → bank (64/4)   % 32 = 16
lds[2][0]:  byte 128 → bank (128/4)  % 32 = 0   ← CONFLICT!
lds[3][0]:  byte 192 → bank (192/4)  % 32 = 16  ← CONFLICT!
lds[4][0]:  byte 256 → bank (256/4)  % 32 = 0   ← CONFLICT!
...
lds[16][0]: byte 1024 → bank (1024/4) % 32 = 0  ← CONFLICT!
```

**Pattern Analysis:**
```
Stride = 32 FP16 = 64 bytes
Bank offset = (64 / 4) % 32 = 16

Bank assignment pattern:
Thread 0:  bank 0
Thread 1:  bank 16
Thread 2:  bank 0   ← repeats!
Thread 3:  bank 16  ← repeats!
Thread 4:  bank 0
...
Thread 32: bank 0
...
Thread 63: bank 16

Total: 64 threads using only 2 banks (0 and 16)
→ 32 threads per bank
→ 32-way bank conflicts!
```

### Visualization: Small Example (4×4 Matrix)

```
Physical layout in LDS (row-major, FP16):
┌─────┬─────┬─────┬─────┐
│ [0,0]│[0,1]│[0,2]│[0,3]│  Row 0: bytes 0-8
├─────┼─────┼─────┼─────┤
│ [1,0]│[1,1]│[1,2]│[1,3]│  Row 1: bytes 8-16
├─────┼─────┼─────┼─────┤
│ [2,0]│[2,1]│[2,2]│[2,3]│  Row 2: bytes 16-24
├─────┼─────┼─────┼─────┤
│ [3,0]│[3,1]│[3,2]│[3,3]│  Row 3: bytes 24-32
└─────┴─────┴─────┴─────┘

Reading column 0: [0,0], [1,0], [2,0], [3,0]
Byte offsets:     0,     8,     16,    24
Banks:            0,     2,     4,     6   (no conflicts in small example)

But scale to [64, 32]:
Reading column 0: 64 elements
Byte offsets:     0, 64, 128, 192, 256, ...  (32 FP16 per row = 64 bytes)
Banks:            0, 16,  0,  16,  0,  ...  (only 2 banks!)
→ 32-way conflicts on each of 2 banks!
```

### CSP Constraint Analysis

**Why does transpose violate Constraint 2 (access pattern)?**

```
Algorithm requirement: Must read columns (transpose)
Physical layout:       Row-major (imposed by write phase)
Result:                Column read = stride-K access = 64-byte jumps
Bank aliasing:         64-byte stride → 16-bank offset → repeats every 2 threads

Constraint violation:
- Need to access 64 elements
- Only 2 distinct banks used
- 32 threads compete for each bank
- Severe serialization
```

**Can we modify the constraints?**

| Constraint | Can Modify? | How? |
|------------|------------|------|
| 1. Hardware (32 banks) | ✗ No | Cannot change GPU architecture |
| 2. Access pattern | ✓ Partially | XOR swizzling permutes physical addresses |
| 3. Parallelism (64 threads) | ✓ Yes | Use 32×32 tiles instead, but hurts performance |

**XOR swizzling modifies Constraint 2** by permuting physical addresses while keeping the logical view unchanged.

---

## Part 4: XOR Swizzling Solution

### The XOR Idea: Address Permutation

**Problem 1:** Stride-32 access causes aliasing to only 2 banks

```
Plain LDS:
Thread 0 → byte 0   → bank 0
Thread 2 → byte 128 → bank 0  ← conflict!
Thread 4 → byte 256 → bank 0  ← conflict!
```

**Problem 2:** Non-sequential read phases create conflicts even with good write pattern

```
Plain LDS, reading column 0:
Phase 0 contains lanes {0, 1, 2, 3, 20, 21, 22, 23}
  lanes 0, 2 → both hit bank 0  ← conflict within phase!
  lanes 20, 22 → both hit bank 0 ← conflict within phase!
```

**Solution:** Permute physical addresses to spread accesses across all 32 banks **for both read AND write phases**

```
XOR permutation: physical_addr = XOR(row_index, col_index)

Result for writes (sequential phases):
  Phase 0: lanes 0-7 access XOR-permuted banks → distributed

Result for reads (non-sequential phases):
  Phase 0: lanes {0,1,2,3,20,21,22,23} access XOR-permuted banks → distributed

Distribution: All 32 banks used instead of just 2, for BOTH operations!
```

**Key Insight:**
- **Logical view unchanged**: [M, K] → [K, M] transpose still works
- **Physical addresses permuted**: Elements stored at XOR'd locations
- **Break aliasing pattern**: Distribute conflicts across all banks
- **Dual optimization**: Works for both sequential write phases AND non-sequential read phases

**XOR Property:**
```
XOR(a, b) ⊕ XOR(a, c) = b ⊕ c  (different when b ≠ c)

This mathematical property spreads consecutive indices across different banks.
```

**Why XOR works for both phase patterns:**

The XOR permutation formula `x' = (y mod (KPerBlock/KPack)) ⊕ x` creates a mapping where:

1. **For write phases (sequential lanes 0-7, 8-15, etc.):**
   - Even if writing consecutive addresses, XOR spreads them across different banks
   - Each phase's 8 lanes hit 8 different banks

2. **For read phases (non-sequential lanes {0-3,20-23}, {4-7,16-19}, etc.):**
   - Even if reading strided addresses, XOR redistributes them
   - Lanes 0 and 20 (both in phase 0) get mapped to different banks via XOR
   - Each phase's 8 lanes hit different banks

This is the "magic" of XOR swizzling - it's not optimized for just one pattern, but for **the specific combination of write and read phase groupings** used by AMD GPUs!

### CK Tile Implementation: Descriptor Transforms

The CK Tile API implements XOR swizzling through a series of **tensor descriptor transformations**.

#### Understanding MLdsLayer

First, calculate the "layer" parameter for bank conflict awareness:

```cpp
constexpr auto DataTypeSize = sizeof(DataType);  // 2 for FP16
constexpr auto MLdsLayer = (32 * 4 / kK / DataTypeSize);
// = (128 / 32 / 2) = 2

// What does this mean?
// - 32 banks × 4 bytes = 128 bytes total per "row" of banks
// - For kK=32 FP16: 32 × 2 = 64 bytes per row
// - Need 2 rows to cover all 32 banks → MLdsLayer = 2
// - This relates to how many elements fit in the bank structure
```

#### Step-by-Step Descriptor Construction

Let's build the XOR descriptor for [M, K] = [64, 32]:

**Step 1: Reshape to expose XOR dimensions**

```cpp
constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
    make_tuple(number<kK / kKPack * MLdsLayer>{},  // 32/8 * 2 = 8
               number<kM / MLdsLayer>{},            // 64/2 = 32
               number<kKPack>{}),                   // 8
    make_tuple(number<kKPack>{},                   // stride: 8
               number<kK * MLdsLayer>{},            // stride: 64
               number<1>{}),                        // stride: 1
    number<kKPack>{},
    number<1>{});

// Resulting shape: [8, 32, 8]
// This reshaping exposes the dimensions that will be XOR'd:
// - Dimension 0: K-related (8 elements)
// - Dimension 1: M-related (32 elements)
// - Dimension 2: Pack (8 elements for vectorization)
```

**Step 2: Apply XOR transform**

```cpp
constexpr auto lds_desc_permuted = transform_tensor_descriptor(
    lds_desc_0,
    make_tuple(
        make_xor_transform(
            make_tuple(number<kM / MLdsLayer>{},           // 32
                       number<kK / kKPack * MLdsLayer>{})), // 8
        make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<1, 0>{},  // XOR dimensions 1 and 0
               sequence<2>{}),    // Pass through dimension 2 (pack)
    make_tuple(sequence<1, 0>{},  // Output dimensions
               sequence<2>{}));

// XOR transform operates on dimensions [1, 0]:
// physical_address = XOR(dim1_index, dim0_index)
//                  = XOR(m_component, k_component)
// This permutes physical locations while keeping logical view!
```

**Step 3: Unmerge layer dimension**

```cpp
constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
    lds_desc_permuted,
    make_tuple(
        make_unmerge_transform(
            make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
        make_pass_through_transform(number<kM / MLdsLayer>{}),
        make_pass_through_transform(number<kKPack>{})),
    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
    make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

// Shape: [Layer, M/Layer, K/Pack, Pack] = [2, 32, 4, 8]
// Split first dimension to separate layer from K
```

**Step 4: Merge back to [M, K]**

```cpp
constexpr auto lds_desc = transform_tensor_descriptor(
    lds_desc_unmerged,
    make_tuple(
        make_merge_transform(
            make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
        make_merge_transform(
            make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
    make_tuple(sequence<1, 0>{},   // Merge M dimensions: [M/Layer, Layer] → M
               sequence<2, 3>{}),  // Merge K dimensions: [K/Pack, Pack] → K
    make_tuple(sequence<0>{},      // Output dimension 0: M
               sequence<1>{}));    // Output dimension 1: K

// Final shape: [M, K] = [64, 32]
// But with XOR permutation baked into address calculation!
```

**Summary of transformations:**
```
Original:  [M, K] = [64, 32]
   ↓ reshape
Step 1:    [K/Pack*Layer, M/Layer, Pack] = [8, 32, 8]
   ↓ XOR transform
Step 2:    [M/Layer, K/Pack*Layer, Pack] = [32, 8, 8] (XOR'd addresses)
   ↓ unmerge layer
Step 3:    [Layer, M/Layer, K/Pack, Pack] = [2, 32, 4, 8]
   ↓ merge back
Step 4:    [M, K] = [64, 32] (with XOR addressing)
```

### Matching Transposed Descriptor: [K, M]

**Critical requirement:** The read descriptor must use the **SAME XOR pattern**!

The key difference is in **Step 4: swap the merge order**:

```cpp
// Write descriptor (Step 4): [M, K]
make_tuple(
    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),  // M first
    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),       // K second
make_tuple(sequence<1, 0>{},   // M dimensions → output 0
           sequence<2, 3>{}),  // K dimensions → output 1

// Read descriptor (Step 4): [K, M]
make_tuple(
    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{})),       // K first!
    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{}))), // M second!
make_tuple(sequence<2, 3>{},   // K dimensions → output 0
           sequence<1, 0>{}),  // M dimensions → output 1
```

**Why this works:**
- **Steps 1-3**: Identical XOR transformation for both descriptors
- **Step 4**: Different merge order creates different logical view
- **Write**: Interprets data as [M, K]
- **Read**: Interprets SAME data as [K, M] (transposed!)
- **Physical memory**: Same XOR-permuted addresses for both

### Hands-On: XOR Transpose Results

**Build and run:**
```bash
cd relbuild
cmake --build . --target aa_tutorial_11_production_transpose -j$(nproc)
./bin/aa_tutorial_11_production_transpose
```

**Profile:**
```bash
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/xor \
          -- ./bin/aa_tutorial_11_production_transpose
```

**Expected Results Comparison:**

```
┌─────────────────────┬──────────────┬──────────────┬─────────────┐
│ Metric              │  Plain LDS   │   XOR LDS    │ Improvement │
├─────────────────────┼──────────────┼──────────────┼─────────────┤
│ SQ_LDS_BANK_CONFLICT│    7,168     │    3,072     │   -4,096    │
│ SQ_INSTS_LDS        │      608     │      608     │      0      │
│ Conflict Rate (%)   │   1,244%     │     533%     │    -711%    │
│ Conflicts per Inst  │     12.4     │      5.3     │    -7.1     │
└─────────────────────┴──────────────┴──────────────┴─────────────┘

Improvement:
- Absolute: 4,096 fewer conflicts (57% reduction)
- Per instruction: 12.4 → 5.3 (57% reduction)
- Still above theoretical minimum (2-way = 100% rate)
```

**Interpretation:**
- ✓ XOR reduces conflicts from ~12-way to ~5-way
- ✓ 57% reduction is significant for practical performance
- ✓ Conflicts not eliminated (theoretical minimum: 2-way with 64 threads/32 banks)
- ✓ XOR swizzling is effective within CSP constraints

### CSP Solution Analysis

**Constraints respected:**

| Constraint | Plain LDS | XOR LDS |
|-----------|-----------|---------|
| 1. Hardware (32 banks) | ✓ | ✓ (same banks, permuted access) |
| 2. Access pattern (transpose) | ✓ | ✓ (algorithm unchanged) |
| 3. Parallelism (64 threads) | ✓ | ✓ (same thread count) |

**Optimization achieved:**
- Spread conflicts across **all 32 banks** instead of just 2
- Reduce average conflicts per bank from 32-way to ~5-way
- Gap to theoretical minimum: 5-way (current) vs 2-way (optimal) = 2.5× away

**Why not optimal?**
- XOR is a **specific permutation pattern** (bitwise XOR operation)
- **Not an exhaustive search** for perfect distribution
- Must work for **both** write phases (sequential) **and** read phases (non-sequential)
- Trade-off: Simple hardware-friendly operation vs perfect optimization
- 57% improvement with **no algorithm changes** is excellent practical result

### Summary: How XOR Addresses Phase Asymmetry

Let's recap why XOR swizzling is necessary and how it works:

**The Challenge:**
```
✗ Row-major layout alone:
  - Write phases (sequential): ✓ Conflict-free
  - Read phases (non-sequential): ✗ 4-way conflicts

✗ Column-major layout alone:
  - Write phases (sequential): ✗ Conflicts
  - Read phases (non-sequential): Maybe better, but breaks write

✓ XOR swizzling:
  - Write phases (sequential): ✓ Reduced conflicts
  - Read phases (non-sequential): ✓ Reduced conflicts
  - Both optimized simultaneously!
```

**The XOR Formula:**
```cpp
x' = (y mod (KPerBlock/KPack)) ⊕ x

Where:
  x = original column index
  y = row index
  x' = XOR-permuted column index
  ⊕ = bitwise XOR
```

**How it helps:**

1. **For write phases (lanes 0-7, 8-15, ...):**
   - Consecutive writes get distributed across banks via XOR
   - Even lanes in same phase hit different banks

2. **For read phases (lanes {0-3,20-23}, {4-7,16-19}, ...):**
   - Strided reads get redistributed across banks via XOR
   - Non-sequential lanes in same phase hit different banks

3. **Key property:**
   - XOR(a, b) ⊕ XOR(a, c) = b ⊕ c
   - This spreads both sequential AND non-sequential access patterns

**Result:**
- Plain: 12.4 conflicts/instruction (only 2 banks used)
- XOR: 5.3 conflicts/instruction (all 32 banks used)
- Improvement: 57% reduction, works for BOTH read and write!

**Why this matters:**
- Without understanding phase asymmetry, XOR swizzling seems like "magic"
- With this knowledge, it's a **principled solution** to a **well-defined hardware constraint**
- This is what makes XOR the standard approach for AMD GPU LDS optimization

---

## Part 5: Limitations and Alternative Solutions

### Why Not Zero Conflicts? The Mathematical Limit

**Pigeonhole Principle:**
```
Pigeons (threads): 64
Holes (banks):     32
Conclusion: At least ⌈64/32⌉ = 2 threads per bank (minimum)
```

**Theoretical Analysis:**

```
Perfect distribution:
- 64 threads ÷ 32 banks = 2 threads per bank exactly
- Conflict rate: 100% (1 conflict per LDS instruction)
- This is the BEST POSSIBLE with 64 threads and 32 banks

Current XOR implementation:
- Conflict rate: 533% (5.3 conflicts per instruction)
- ~2.7 threads per bank on average
- Gap to optimal: 2.7 / 2 = 1.35× (could improve 35% more theoretically)

Plain LDS baseline:
- Conflict rate: 1,244% (12.4 conflicts per instruction)
- 32 threads per bank (only 2 banks used!)
- XOR improvement: 12.4 / 5.3 = 2.34× fewer conflicts per instruction
```

**Why XOR doesn't reach theoretical optimal:**
- XOR is a **fixed mathematical pattern** (bitwise XOR operation)
- Optimal distribution requires **perfect load balancing** across all banks
- XOR **approximates** this but doesn't exhaustively search for perfection
- Trade-off: **Simple hardware-friendly** operation vs perfect distribution

### Alternative Solutions: Exploring the CSP Solution Space

#### Option 1: Change Tile Dimensions (Relax Constraint 3)

```
Current: [64, 32] → [32, 64]
Problem: 64 threads > 32 banks → conflicts unavoidable

Alternative: [32, 32] → [32, 32]
Threads: 32 (reading column)
Banks:   32
Result:  1:1 mapping → ZERO conflicts possible!
```

**Trade-offs:**

| Aspect | 64×32 Tiles (Current) | 32×32 Tiles |
|--------|---------------------|-------------|
| Bank conflicts | 5.3-way (XOR) | 0-2 way (ideal) |
| Kernel launches | N | 4N (same matrix) |
| Parallelism | 64 threads | 32 threads |
| LDS usage | 4 KB | 2 KB |
| Throughput | Higher | Lower (30-40% typically) |

**When to use:**
- Small matrices where launch overhead is acceptable
- Educational purposes (demonstrate zero conflicts)
- Latency-critical transpose operations

**When to avoid:**
- Large matrices (launch overhead dominates)
- Throughput-critical applications

#### Option 2: Padding (Modify Constraint 2 Carefully)

```
Current: kK = 32, stride = 32 FP16 = 64 bytes
Bank stride: (64/4) % 32 = 16 banks

Add padding: kK = 33 (store 33 elements, use 32)
New stride: 33 FP16 = 66 bytes
Bank stride: (66/4) % 32 = 16.5 → rounds to 16 (NO improvement!)

Better padding: kK = 40
New stride: 40 FP16 = 80 bytes
Bank stride: (80/4) % 32 = 20 banks

Analysis:
- Changes stride from 16-bank to 20-bank offset
- More banks involved, but still not optimal
- Wastes LDS: 64 × 40 = 2,560 vs 64 × 32 = 2,048 (25% more)
```

**Trade-offs:**

| Aspect | No Padding | With Padding (kK=40) |
|--------|-----------|---------------------|
| Bank conflicts | 5.3-way (XOR) | 4-6 way (marginal improvement) |
| LDS usage | 4 KB | 5 KB (25% more) |
| Occupancy | Higher | Lower (fewer blocks per CU) |
| Complexity | Simple | Must tune padding value |

**When to use:**
- After profiling shows padding helps your specific case
- Sufficient LDS available (not occupancy-limited)

**When to avoid:**
- Already occupancy-limited
- Padding doesn't mathematically improve your stride pattern

#### Option 3: Double Buffering (Separate Constraint 2 for Read/Write)

```cpp
__shared__ DataType lds_write[64][32];  // Write buffer
__shared__ DataType lds_read[32][64];   // Read buffer (transposed layout)

// Phase 1: Write row-major to lds_write (no conflicts)
lds_write[m][k] = gmem_in[m][k];

// Phase 2: Transpose copy to lds_read (with XOR on lds_read)
lds_read[k][m] = lds_write[m][k];

// Phase 3: Read row-major from lds_read (no conflicts!)
gmem_out[k][m] = lds_read[k][m];
```

**Trade-offs:**

| Aspect | Single Buffer (XOR) | Double Buffer |
|--------|---------------------|---------------|
| Bank conflicts | 5.3-way (read phase) | 0-2 way (both phases) |
| LDS usage | 4 KB | 8 KB (2×) |
| Occupancy | Higher | Lower (fewer blocks) |
| Complexity | Medium | High |
| Extra copy | No | Yes (transpose phase) |

**When to use:**
- LDS-rich workloads (not occupancy-limited)
- Transpose is critical bottleneck (>20% of runtime)

**When to avoid:**
- Occupancy-limited kernels
- Transpose is not the bottleneck

#### Option 4: Wavefront-Level Transpose (Partition Constraint 3)

```
Current: All threads cooperate on [64, 32] tile

Alternative: Each wavefront handles [16, 16] sub-tile
- 64 threads = 1 wavefront
- Process 4 sub-tiles: [16,16] × 4
- Each sub-tile: 16 threads < 32 banks → no conflicts!
```

**Trade-offs:**

| Aspect | Block-Level (Current) | Wavefront-Level |
|--------|----------------------|-----------------|
| Bank conflicts | 5.3-way | 0-1 way |
| Distribution complexity | Simple | High |
| Global memory coalescing | Optimal | May be suboptimal |
| LDS sync overhead | Low | Higher (more tiles) |

**When to use:**
- Advanced optimization after XOR swizzling
- Research/specialized applications

**When to avoid:**
- Production code (too complex)
- First-pass optimization

### When XOR Swizzling Is Enough

#### Practical Performance Impact

Consider a typical GEMM kernel breakdown:

```
┌─────────────────────────────┬──────────┐
│ Kernel Phase                │ % Time   │
├─────────────────────────────┼──────────┤
│ Global memory load          │   30%    │
│ LDS transpose               │   10%    │  ← We optimize this!
│ Compute (MFMA)              │   50%    │
│ Global memory store         │   10%    │
└─────────────────────────────┴──────────┘

XOR impact on transpose:
- Conflict reduction: 57%
- Speedup on transpose portion: ~2.3× (12-way → 5-way)
- Overall GEMM impact: 10% × 2.3× ≈ 8-12% total speedup

Diminishing returns of further optimization:
- Going from 5-way to 2-way (optimal): 2.5× on 10% = 1.5-2% total
- Complex implementations for minimal total gain
- Better to optimize compute or global memory instead!
```

#### When to Optimize Further

Pursue zero conflicts when:
1. **Profiling shows transpose is bottleneck** (>20% of kernel time)
2. **Memory-bound kernels** where every cycle matters
3. **Small matrices** where tile size changes are acceptable
4. **Educational purposes** to demonstrate zero-conflict techniques

#### Recommended Optimization Approach

```
1. Start with XOR swizzling
   - 57% conflict reduction
   - Simple implementation
   - No algorithm changes

2. Profile your kernel
   - Measure actual bottleneck
   - Use rocprofv3 for detailed metrics

3. Only if transpose >20% of runtime:
   - Try different tile sizes (32×32)
   - Profile again to verify improvement
   - Consider double buffering if LDS-rich

4. Don't over-optimize:
   - Focus on the actual bottleneck
   - Simple code is maintainable code
   - 90% performance with 10% effort is usually best
```

### CSP Summary: Trade-off Analysis

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                   Solution Comparison Table                               ║
╠═══════════════╤═══════════╤═══════════╤════════════╤═══════════════════════╣
║ Solution      │ Conflicts │ LDS Usage │ Complexity │ Throughput           ║
╠═══════════════╪═══════════╪═══════════╪════════════╪═══════════════════════╣
║ Plain (no XOR)│   12.4    │    1×     │   Simple   │ Baseline             ║
║ XOR Swizzling │    5.3    │    1×     │   Medium   │ +15-20% (typical)    ║
║ 32×32 tiles   │    0-2    │    1×     │   Medium   │ -10-30% (depends)    ║
║ Padding       │    4-6    │   1.25×   │   Medium   │ -5-10% (occupancy)   ║
║ Double Buffer │     2     │    2×     │   High     │ -20-40% (occupancy)  ║
║ Wavefront     │    0-1    │    1×     │   High     │ Variable             ║
╚═══════════════╧═══════════╧═══════════╧════════════╧═══════════════════════╝
```

**Conclusion:**
- **XOR swizzling provides the best balance** for production code
- Zero conflicts are **mathematically possible** but often not worth the trade-offs
- Understanding constraints helps make **informed optimization decisions**
- **Measure, don't guess**: Always profile before complex optimizations

---

## Hands-On Exercises

### Exercise 1: Baseline Profiling

**Objective:** Measure bank conflicts in plain transpose without XOR swizzling.

**Steps:**

1. **Build the plain transpose tutorial:**
   ```bash
   cd relbuild
   cmake --build . --target aa_tutorial_11_plain_transpose -j$(nproc)
   ```

2. **Run to verify correctness:**
   ```bash
   ./bin/aa_tutorial_11_plain_transpose
   ```
   Expected output: "✓ PASSED"

3. **Profile with rocprofv3:**
   ```bash
   rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
             -d /tmp/plain_profile \
             -- ./bin/aa_tutorial_11_plain_transpose
   ```

4. **Query results:**
   ```bash
   sqlite3 /tmp/plain_profile/*/results.db "
   SELECT
       SUM(CASE WHEN counter_name = 'SQ_LDS_BANK_CONFLICT' THEN counter_value ELSE 0 END) as conflicts,
       SUM(CASE WHEN counter_name = 'SQ_INSTS_LDS' THEN counter_value ELSE 0 END) as lds_insts,
       ROUND(100.0 * conflicts / lds_insts, 2) as conflict_rate_percent
   FROM pmc_events;"
   ```

5. **Calculate conflicts per instruction:**
   ```
   conflicts_per_inst = conflicts / lds_insts
   ```

**Expected Results:**
- SQ_LDS_BANK_CONFLICT: ~7,168
- SQ_INSTS_LDS: ~608
- Conflict rate: ~1,244%
- Conflicts per instruction: ~12.4

**Analysis Questions:**
1. Why is the conflict rate over 100%?
2. How many cycles are wasted per LDS instruction on average?
3. What is the theoretical minimum conflict rate for 64 threads and 32 banks?

### Exercise 2: XOR Optimization

**Objective:** Measure the improvement from XOR swizzling.

**Steps:**

1. **Build the production transpose with XOR:**
   ```bash
   cmake --build . --target aa_tutorial_11_production_transpose -j$(nproc)
   ```

2. **Run to verify correctness:**
   ```bash
   ./bin/aa_tutorial_11_production_transpose
   ```
   Both tests (plain and XOR) should pass.

3. **Profile the XOR version:**
   ```bash
   rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
             -d /tmp/xor_profile \
             -- ./bin/aa_tutorial_11_production_transpose
   ```

4. **Compare with baseline:**
   ```bash
   # Use the analysis script (see Scripts section)
   python3 ../example/ck_tile/99_toy_tutorial/scripts/analyze_bank_conflicts.py \
           /tmp/plain_profile \
           /tmp/xor_profile
   ```

**Expected Results:**
- SQ_LDS_BANK_CONFLICT: ~3,072
- SQ_INSTS_LDS: ~608
- Conflict rate: ~533%
- Conflicts per instruction: ~5.3
- Improvement: 57% reduction

**Analysis Questions:**
1. What percentage of conflicts were eliminated by XOR swizzling?
2. How close are we to the theoretical minimum (2-way conflicts)?
3. Why can't XOR eliminate all conflicts beyond the theoretical minimum?

### Exercise 3: Custom Tile Size (Advanced)

**Objective:** Experiment with 32×32 tiles to achieve near-zero conflicts.

**Steps:**

1. **Modify Tutorial 11m to use 32×32 tiles:**
   - Edit `xor_test_production_transpose.cpp`
   - Change `kM = 64` to `kM = 32`
   - Rebuild

2. **Profile the modified version:**
   ```bash
   rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
             -d /tmp/tile32_profile \
             -- ./bin/aa_tutorial_11_production_transpose
   ```

3. **Compare all three versions:**
   - Plain 64×32
   - XOR 64×32
   - XOR 32×32

**Expected Results (32×32 tiles):**
- Conflict rate: close to 100% (theoretical minimum)
- But: overall throughput may be lower

**Analysis Questions:**
1. Did 32×32 tiles achieve zero (or near-zero) conflicts?
2. How does the total execution time compare to 64×32 tiles?
3. What is the trade-off between fewer conflicts and overall performance?

---

## Appendix: CK Tile API Reference

### Tensor Descriptors

**Naive tensor descriptor (row-major):**
```cpp
auto desc = make_naive_tensor_descriptor(
    make_tuple(M, K),                    // Dimensions
    make_tuple(K, number<1>{}));        // Strides

// Creates descriptor for [M, K] with:
// - offset = m * K + k
// - row-major layout
```

**Packed descriptor:**
```cpp
auto desc = make_naive_tensor_descriptor_packed(make_tuple(M, K));
// Equivalent to naive with stride [K, 1]
```

### Transform Operations

**Pass-through transform:**
```cpp
make_pass_through_transform(number<N>{})
// Identity transformation - dimension unchanged
```

**Merge transform:**
```cpp
make_merge_transform(make_tuple(number<N0>{}, number<N1>{}))
// Merge two dimensions: [N0, N1] → [N0*N1]
// Resulting stride: original_stride_N1
```

**Unmerge transform:**
```cpp
make_unmerge_transform(make_tuple(number<N0>{}, number<N1>{}))
// Split one dimension: [N0*N1] → [N0, N1]
```

**XOR transform:**
```cpp
make_xor_transform(make_tuple(number<N0>{}, number<N1>{}))
// Permute addresses using XOR:
// physical_offset = XOR(index0, index1)
```

### Transform Tensor Descriptor

```cpp
auto new_desc = transform_tensor_descriptor(
    old_desc,
    make_tuple(transform1, transform2, ...),  // Transformations
    make_tuple(sequence<...>{}, ...),          // Input dimension mapping
    make_tuple(sequence<...>{}, ...));         // Output dimension mapping
```

**Example: Transpose [M, K] → [K, M]**
```cpp
auto desc_mk = make_naive_tensor_descriptor(
    make_tuple(M, K),
    make_tuple(K, number<1>{}));

auto desc_km = transform_tensor_descriptor(
    desc_mk,
    make_tuple(make_pass_through_transform(K),
               make_pass_through_transform(M)),
    make_tuple(sequence<1>{},    // Old dim 1 (K)
               sequence<0>{}),   // Old dim 0 (M)
    make_tuple(sequence<0>{},    // New dim 0 (K)
               sequence<1>{}));  // New dim 1 (M)
```

### Tile Distributions

**Static tile distribution encoding:**
```cpp
tile_distribution_encoding<
    sequence<1>,                                        // NumOfDimensionGroups
    tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,     // Distribution per dim
    tuple<sequence<1>, sequence<1, 2>>,                 // ThreadDimMapping
    tuple<sequence<1>, sequence<2, 0>>,                 // ReplicateMapping
    sequence<1, 2>,                                     // DimSpace
    sequence<0, 1>                                      // DimReads
>
```

### Complete XOR Descriptor Example

See the implementation in `xor_test_production_transpose.cpp` for a full working example of:
1. MLdsLayer calculation
2. Step-by-step descriptor transformations
3. Matching write/read descriptors for transpose

---

## References

### AMD Official Documentation

**LDS Architecture and Phase Grouping:**
- [ROCm Blog: Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile Framework](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
  - Detailed explanation of ds_read_b128 and ds_write_b128 phase groupings
  - XOR preshuffle formula and implementation
  - Conflict pattern analysis

- [Composable Kernel: Understanding AMD GPU LDS and Bank Conflicts](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html)
  - Official CK documentation on LDS bank structure
  - Phase division details
  - MFMA instruction requirements

- [HIP Documentation: Hardware Implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
  - LDS bidirectional ports
  - 32-bank architecture
  - Per-cycle access capabilities

**Additional Resources:**
- [Understanding AMD GPU LDS (Interactive Tutorial)](https://ghamarian.github.io/pythonck/hardware/01_lds.html)
  - Interactive Python visualizations
  - Phase grouping diagrams
  - XOR swizzling examples

- AMD MI300 Architecture Guide
- CK Tile API Documentation
- ROCm Profiling Tools Guide

### Key Findings from Documentation

**Phase Grouping Asymmetry:**
- ds_write_b128: Sequential phases (lanes 0-7, 8-15, 16-23, ...)
- ds_read_b128: Non-sequential phases (lanes {0-3,20-23}, {4-7,16-19}, ...)
- Reason: MFMA instructions require vertical data access patterns

**XOR Solution:**
- Formula: `x' = (y mod (KPerBlock/KPack)) ⊕ x`
- Addresses both sequential write and non-sequential read phase patterns
- Achieves conflict-free access without extra LDS storage

### Research Papers
- "Conflict-Free Access Patterns for Shared Memory" (GPU architecture)
- "Optimizing Matrix Transpose on GPUs" (transpose algorithms)

### Related Tutorials
- Tutorial 08: LDS Staging
- Tutorial 09: Optimized LDS
- Tutorial 10: XOR LDS (first introduction)
- Tutorial 13: Production XOR (complete implementation)

### Performance Analysis Tools
- `rocprofv3` - AMD GPU profiler
- `rocprof` - Legacy profiler
- Omniperf - Detailed performance analysis

---

**End of Tutorial**

For questions or feedback, please refer to the CK Tile documentation or open an issue in the composable_kernel repository.
