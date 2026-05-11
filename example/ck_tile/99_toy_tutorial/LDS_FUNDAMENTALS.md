# Understanding AMD GPU LDS and Bank Conflicts: From First Principles

## Table of Contents

1. [Introduction to LDS](#introduction-to-lds)
2. [Bank Architecture](#bank-architecture)
3. [What Are Bank Conflicts?](#what-are-bank-conflicts)
4. [Thread Organization and Phases](#thread-organization-and-phases)
5. [Vector Operations](#vector-operations)
6. [Phase Grouping: The Critical Asymmetry](#phase-grouping-the-critical-asymmetry)
7. [Practical Examples](#practical-examples)
8. [Introduction to Solutions](#introduction-to-solutions)

---

## Introduction to LDS

**Local Data Share (LDS)** is AMD's on-chip shared memory within a compute unit. It serves as a fast scratchpad that all threads (lanes) within a workgroup can access.

### Why LDS Matters

LDS is dramatically faster than global memory:
- **LDS bandwidth**: ~10-20 TB/s (on-chip)
- **Global memory bandwidth**: ~1-2 TB/s (off-chip)
- **Speed difference**: 10-20× faster

However, this speed advantage comes with constraints. To maximize LDS throughput, we must understand and avoid **bank conflicts**.

### Basic Architecture Overview

LDS is organized as an array of **banks**. Think of banks as parallel access lanes:
- Multiple threads can access **different banks** simultaneously (parallel)
- Multiple threads accessing the **same bank** must wait (serialized)

Understanding how memory addresses map to banks is the key to efficient LDS usage.

---

## Bank Architecture

### The 32-Bank Organization

AMD GPUs (GCN and CDNA architectures) organize LDS into:
- **32 banks**
- **4 bytes per bank per cycle**
- **Total bandwidth**: 128 bytes/cycle (32 banks × 4 bytes)

### Bank Assignment Formula

The bank for a given address is determined by:

```
bank = (address_bytes / 4) % 32
```

This means:
- **Address 0** → Bank 0
- **Address 4** → Bank 1
- **Address 8** → Bank 2
- **Address 128** (32 × 4) → Bank 0 again

### Simple Example

```
Address (bytes)    Bank    Calculation
     0              0      (0 / 4) % 32 = 0
     4              1      (4 / 4) % 32 = 1
     8              2      (8 / 4) % 32 = 2
    12              3      (12 / 4) % 32 = 3
   128              0      (128 / 4) % 32 = 0
   132              1      (132 / 4) % 32 = 1
```

Addresses separated by 128 bytes (32 banks × 4 bytes) map to the same bank.

---

## What Are Bank Conflicts?

### Definition

A **bank conflict** occurs when multiple threads in the same execution phase try to access the same bank simultaneously.

When this happens:
- The hardware **serializes** the accesses
- Each conflicting access waits its turn
- Throughput drops proportionally to the conflict degree

### Conflict Degree

- **No conflict**: All threads access different banks → Full throughput
- **2-way conflict**: 2 threads access the same bank → 50% throughput
- **4-way conflict**: 4 threads access the same bank → 25% throughput
- **8-way conflict**: 8 threads access the same bank → 12.5% throughput

### Visual Example: Good vs Bad Access Patterns

**Good Pattern** (No conflicts):
```
Thread 0 → Bank 0
Thread 1 → Bank 1
Thread 2 → Bank 2
Thread 3 → Bank 3
Thread 4 → Bank 4
Thread 5 → Bank 5
Thread 6 → Bank 6
Thread 7 → Bank 7

Result: All 8 threads execute in parallel (1 cycle)
```

**Bad Pattern** (8-way conflict):
```
Thread 0 → Bank 0
Thread 1 → Bank 0
Thread 2 → Bank 0
Thread 3 → Bank 0
Thread 4 → Bank 0
Thread 5 → Bank 0
Thread 6 → Bank 0
Thread 7 → Bank 0

Result: All 8 threads serialize (8 cycles)
```

---

## Thread Organization and Phases

### Wavefront Basics

AMD GPUs execute threads in groups called **wavefronts** (or **waves**):
- **Wave size**: 64 threads (lanes) on CDNA architectures
- All lanes in a wave execute the same instruction (SIMD)
- But not all lanes access LDS simultaneously!

### Hardware Phase Division

The hardware cannot execute all 64 lanes' LDS operations in a single cycle. Instead, it divides them into **phases**.

**Key insight**: Which lanes execute together in each phase depends on the **instruction type**.

### Why Phases Exist

Hardware limitation: Even with 32 banks providing 128 bytes/cycle:
- Each lane may request 16 bytes (4 banks)
- 64 lanes × 16 bytes = 1024 bytes
- But we only have 128 bytes/cycle bandwidth
- Solution: Execute in 8 phases (1024 / 128 = 8)

---

## Vector Operations

### Common LDS Instructions

Two key instructions for 16-byte (128-bit) vector operations:

1. **`ds_write_b128`**: Write 16 bytes from a lane to LDS
2. **`ds_read_b128`**: Read 16 bytes from LDS into a lane

### Typical Use Case

For machine learning workloads with FP16/BF16 data:
- Each element: 2 bytes
- Vector size: 8 elements
- Total per lane: 8 × 2 = 16 bytes

### Bank Coverage Per Lane

When a lane executes a 16-byte operation:
```
16 bytes / 4 bytes per bank = 4 banks
```

Each lane's operation spans **4 consecutive banks**.

**Example**:
```
Lane 0 at address 0:
  - Bank 0 (bytes 0-3)
  - Bank 1 (bytes 4-7)
  - Bank 2 (bytes 8-11)
  - Bank 3 (bytes 12-15)

Lane 1 at address 16:
  - Bank 4 (bytes 16-19)
  - Bank 5 (bytes 20-23)
  - Bank 6 (bytes 24-27)
  - Bank 7 (bytes 28-31)
```

---

## Phase Grouping: The Critical Asymmetry

### The Problem

Here's the crucial detail that makes LDS optimization challenging:

**Write and read instructions use different phase groupings!**

### Write Phases (`ds_write_b128`)

Phases are **sequential groups of 8 lanes**:

```
Phase 0: lanes 0-7
Phase 1: lanes 8-15
Phase 2: lanes 16-23
Phase 3: lanes 24-31
Phase 4: lanes 32-39
Phase 5: lanes 40-47
Phase 6: lanes 48-55
Phase 7: lanes 56-63
```

This is intuitive and straightforward.

### Read Phases (`ds_read_b128`)

Phases are **non-sequential, interleaved groups**:

```
Phase 0: lanes 0-3   + lanes 20-23
Phase 1: lanes 4-7   + lanes 16-19
Phase 2: lanes 8-11  + lanes 28-31
Phase 3: lanes 12-15 + lanes 24-27
Phase 4: lanes 32-35 + lanes 52-55
Phase 5: lanes 36-39 + lanes 48-51
Phase 6: lanes 40-43 + lanes 60-63
Phase 7: lanes 44-47 + lanes 56-59
```

Notice the pattern:
- Lanes are split into groups from different parts of the wavefront
- Adjacent lanes in the low range are paired with non-adjacent lanes in the high range

### Why This Matters

An LDS layout that works perfectly for writes may create severe conflicts for reads!

**Key Insight**: You cannot simply check if threads within each write phase avoid conflicts. You must also verify that threads within each read phase avoid conflicts.

### Visualization: Write Phase Pattern

The following Python code shows how `ds_write_b128` phases map to banks with a simple row-major layout:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Parameters
wave_size = 64
op_bytes = 16        # ds_write_b128 writes 16 bytes per lane
stride_bytes = 16    # Each lane starts 16 bytes after the previous
banks = 32
bank_width = 4

# Phase colors (8 phases)
phase_colors = [
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261",
    "#e76f51", "#6a4c93", "#8ab17d", "#577590"
]
phase_cmap = ListedColormap(phase_colors)

# Grid: rows = phases, columns = banks
phase_grid = -np.ones((8, banks), dtype=int)
lane_labels = [["" for _ in range(banks)] for _ in range(8)]

# Compute bank access for each lane
for lane in range(wave_size):
    phase = lane // 8  # Sequential phase assignment for writes
    row = phase
    addr = lane * stride_bytes
    start_bank = (addr // bank_width) % banks

    # Each lane accesses 4 consecutive banks
    for i in range(op_bytes // bank_width):
        b = (start_bank + i) % banks
        phase_grid[row, b] = phase
        if lane_labels[row][b]:
            lane_labels[row][b] += "/"
        lane_labels[row][b] += str(lane)

# Plot
fig, ax = plt.subplots(figsize=(25, 10))
im = ax.imshow(phase_grid, cmap=phase_cmap, aspect='auto', vmin=0, vmax=7)

ax.set_title(
    f"LDS Write (ds_write_b128): Bank Mapping\n"
    "Rows = phase (8 lanes per phase), Color = phase, Label = lane ID(s)"
)
ax.set_xlabel("Bank index (0-31)")
ax.set_ylabel("Phase")

ax.set_xticks(range(0, banks, 2))
ax.set_yticks(range(8))
ax.set_yticklabels([f"P{p}" for p in range(8)])

# Add lane labels
for row in range(8):
    for b in range(banks):
        if lane_labels[row][b]:
            ax.text(b, row, lane_labels[row][b],
                   ha='center', va='center', color='white',
                   fontsize=15, weight='bold')

# Grid lines
ax.set_xticks(np.arange(-0.5, banks, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
ax.grid(which="minor", color=(0,0,0,0.1), linewidth=0.5)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(8))
cbar.ax.set_ylabel("Phase")
cbar.set_ticklabels([f"P{p}" for p in range(8)])

plt.tight_layout()
plt.show()
```

**Expected Result**: With `stride_bytes = 16`, each phase has lanes accessing different banks → **No write conflicts!**

---

## Practical Examples

### Row-Major Matrix Storage

Consider storing a matrix in LDS where:
- Each lane handles 8 FP16 elements (16 bytes)
- 64 lanes → 512 elements per row
- Sequential layout: lane 0 at address 0, lane 1 at address 16, etc.

### Write Access Pattern

Using the write phase grouping (sequential lanes):
- **Phase 0**: lanes 0-7 access banks 0-31 (no overlap)
- **Phase 1**: lanes 8-15 access banks 0-31 (no overlap)
- ...
- **Phase 7**: lanes 56-63 access banks 0-31 (no overlap)

**Result**: ✓ Conflict-free writes!

### Read Access Pattern (Transpose)

Now imagine reading this data in a transposed pattern (common in GEMM):
- Different threads need elements from different rows
- The non-sequential read phase grouping comes into play

Using the read phase grouping:
- **Phase 0**: lanes {0-3, 20-23} may access overlapping banks
- Multiple lanes in the same phase access the same banks

**Result**: ✗ 4-way bank conflicts on reads!

### Visualization: Read Phase Conflicts

The following Python code demonstrates how the same layout that was conflict-free for writes produces conflicts for reads:

```python
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# Hardware constants
banks = 32
bank_width = 4
instr_bytes = 16
num_lanes = 64
banks_per_instr = instr_bytes // bank_width  # 4
row_padding = 0  # no padding

# Read-phase mapping for ds_read_b128 (non-sequential!)
read_phase_lanes = {
    0: list(range(0, 4)) + list(range(20, 24)),
    1: list(range(4, 8)) + list(range(16, 20)),
    2: list(range(8, 12)) + list(range(28, 32)),
    3: list(range(12, 16)) + list(range(24, 28)),
    4: list(range(32, 36)) + list(range(52, 56)),
    5: list(range(36, 40)) + list(range(48, 52)),
    6: list(range(40, 44)) + list(range(60, 64)),
    7: list(range(44, 48)) + list(range(56, 60)),
}

# Reverse map: lane -> phase
lane_to_phase = {}
for p, lanes in read_phase_lanes.items():
    for l in lanes:
        lane_to_phase[l] = p

# Phase colors
phase_colors = [
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261",
    "#e76f51", "#6a4c93", "#8ab17d", "#577590"
]
phase_cmap = ListedColormap(phase_colors)

def lane_start_bank(lane_id):
    """Starting bank for a lane in row-major layout."""
    row_id = lane_id // 8
    phys_row = lane_id % 8
    p = row_padding * phys_row  # padding offset

    start_bank = (row_id * banks_per_instr) % banks
    start_bank = (start_bank + p) % banks
    return start_bank

# Grid: rows = physical row (lane % 8), columns = banks
row_bank_grid = -np.ones((8, banks), dtype=int)
row_labels = [[[] for _ in range(banks)] for _ in range(8)]

for lane in range(num_lanes):
    row = lane % 8  # Physical row for plotting
    sb = lane_start_bank(lane)
    phase = lane_to_phase[lane]

    # Mark the 4 banks this lane accesses
    for i in range(banks_per_instr):
        b = (sb + i) % banks
        row_bank_grid[row, b] = phase
        row_labels[row][b].append(lane)

# Plot
fig, ax = plt.subplots(figsize=(25, 10))
bg = np.ones_like(row_bank_grid, dtype=float)
ax.imshow(bg, cmap=ListedColormap(["#efefef"]),
          extent=(-0.5, banks-0.5, 7.5, -0.5))
im = ax.imshow(np.where(row_bank_grid >= 0, row_bank_grid, 0),
               cmap=phase_cmap, interpolation='nearest', aspect='auto')

ax.set_title(
    "LDS Read (ds_read_b128): Bank Access Pattern\n"
    "Color = Read Phase; Label = lane IDs accessing each bank\n"
    "Notice: Multiple lanes from the same phase hit the same banks (conflicts!)"
)
ax.set_xlabel("Bank index (0-31)")
ax.set_ylabel("Row (lane % 8)")

ax.set_xticks(range(0, banks, 2))
ax.set_yticks(range(8))
ax.set_yticklabels([f"Row {r}" for r in range(8)])

# Grid lines
ax.set_xticks(np.arange(-0.5, banks, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
ax.grid(which="minor", color=(0,0,0,0.1), linewidth=0.5)

# Add lane ID labels
for r in range(8):
    for b in range(banks):
        if row_labels[r][b]:
            text = "/".join(str(x) for x in sorted(row_labels[r][b]))
            # Highlight conflicts (multiple lanes in same cell)
            color = 'red' if len(row_labels[r][b]) > 1 else 'white'
            ax.text(b, r, text, ha='center', va='center',
                   color=color, fontsize=15, weight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                   ticks=range(len(read_phase_lanes)))
cbar.ax.set_ylabel("Phase")
cbar.set_ticklabels([f"P{p}" for p in range(len(read_phase_lanes))])

plt.tight_layout()
plt.show()
```

**Expected Result**: You'll see multiple lane IDs in the same cell (marked in red), indicating that lanes within the same read phase access the same banks. This creates 4-way conflicts!

### Why Padding Doesn't Easily Help

You might think: "Let's add padding between rows to shift the bank assignments."

Try modifying `row_padding` in the code above to values like 4, 8, 12, etc. You'll find:
- Padding helps in some cases but not completely
- It wastes LDS storage (precious resource)
- Finding the right padding value is non-trivial
- Still may not eliminate all conflicts

---

## Introduction to Solutions

The read/write phase asymmetry makes simple solutions inadequate. However, there are advanced techniques:

### 1. XOR Swizzling (Preshuffling)

Instead of storing data sequentially, permute the column indices using XOR operations. This technique:
- Redistributes elements to avoid bank conflicts
- Works without extra storage (unlike padding)
- Is commonly used in production ML kernels

**Basic Idea**:
```
Original column index: x
Row index: y
Permuted column index: x' = (y % N) XOR x
```

The XOR operation cleverly redistributes accesses so that lanes within each read phase hit different banks.

### 2. Advanced Layout Strategies

- Tiled layouts that respect phase boundaries
- Multi-bank-stride patterns
- Block-wise transposition during load/store

### XOR Swizzling Preview

Here's a simple example showing how XOR transforms row indices:

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parameters
num_rows = 8    # y values (row IDs)
num_cols = 8    # x values (columns)
cell_w = 1.0
cell_h = 1.0

fig, axes = plt.subplots(num_rows, 1, figsize=(10, 2 * num_rows))

for r in range(num_rows):
    ax = axes[r]

    # Original x row
    for x in range(num_cols):
        ax.add_patch(Rectangle((x, 0), cell_w, cell_h, fill=False))
        ax.text(x + 0.5, 0.5, f"{x}", ha="center", va="center", fontsize=12)

    # Shuffled x' row (using XOR)
    for x in range(num_cols):
        xprime = r ^ x  # XOR operation
        ax.add_patch(Rectangle((x, -1), cell_w, cell_h, fill=False))
        ax.text(x + 0.5, -0.5, f"{xprime}", ha="center", va="center", fontsize=12)

    # Row labels
    ax.text(-1.5, 0.5, "x", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(-1.5, -0.5, "x'", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(num_cols + 1, -0.25, f"row r={r}", ha="left", va="center",
           fontsize=12, fontweight="bold")

    # Formatting
    ax.set_xlim(-2, num_cols + 2)
    ax.set_ylim(-1.5, 1)
    ax.axis("off")

fig.suptitle("XOR preshuffle mapping per row  (x' = r XOR x)",
            fontsize=16, y=0.92)
plt.tight_layout()
plt.show()
```

Notice how each row gets a different permutation based on the XOR with its row index.

### Complete XOR Comparison

The following visualization shows the full before/after comparison with XOR swizzling applied:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -------------------------
# Parameters
# -------------------------
banks = 32
bank_width = 4
instr_bytes = 16
num_lanes = 64
banks_per_instr = instr_bytes // bank_width  # 4

elem_size_bytes = 2
KPack = 8
RowStride = 64  # elements

# Derived
num_cols = RowStride // KPack   # columns in thread-space = 8
num_rows = num_lanes // num_cols  # 8

# Read-phase mapping
read_phase_lanes = {
    0: list(range(0, 4)) + list(range(20, 24)),
    1: list(range(4, 8)) + list(range(16, 20)),
    2: list(range(8, 12)) + list(range(28, 32)),
    3: list(range(12, 16)) + list(range(24, 28)),
    4: list(range(32, 36)) + list(range(52, 56)),
    5: list(range(36, 40)) + list(range(48, 52)),
    6: list(range(40, 44)) + list(range(60, 64)),
    7: list(range(44, 48)) + list(range(56, 60)),
}
lane_to_phase = {l: p for p, ls in read_phase_lanes.items() for l in ls}

phase_colors = [
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261",
    "#e76f51", "#6a4c93", "#8ab17d", "#577590"
]
phase_cmap = ListedColormap(phase_colors)

mapping_choice = 'A'  # Lane-to-(x,y) mapping

def lane_xy(lane, mapping='A'):
    """Convert lane ID to (x, y) coordinates."""
    if mapping == 'A':
        x = lane // num_rows
        y = lane % num_rows
    else:
        x = lane % num_cols
        y = lane // num_cols
    return int(x), int(y)

def recomposed_lane_from_xy(x, y, mapping='A'):
    """Convert (x, y) back to lane ID."""
    if mapping == 'A':
        return int(x * num_rows + y)
    else:
        return int(y * num_cols + x)

def start_bank_from_laneid(laneid):
    """Starting bank for a lane."""
    row_id = laneid // 8
    start_bank = (row_id * banks_per_instr) % banks
    return start_bank

# Build original grid (no XOR)
orig_grid = -np.ones((num_rows, banks), dtype=int)
orig_labels = [[[] for _ in range(banks)] for _ in range(num_rows)]

for lane in range(num_lanes):
    phys_row_plot = lane % num_rows
    start_bank = start_bank_from_laneid(lane)
    phase = lane_to_phase.get(lane, -1)
    for i in range(banks_per_instr):
        b = (start_bank + i) % banks
        orig_grid[phys_row_plot, b] = phase
        orig_labels[phys_row_plot][b].append(lane)

# Build XOR-preshuffled grid
shuf_grid = -np.ones((num_rows, banks), dtype=int)
shuf_labels = [[[] for _ in range(banks)] for _ in range(num_rows)]

for lane in range(num_lanes):
    x, y = lane_xy(lane, mapping=mapping_choice)
    xprime = (y % num_cols) ^ x  # XOR permutation
    shuffled_lane = recomposed_lane_from_xy(xprime, y, mapping=mapping_choice)
    start_shuf = start_bank_from_laneid(shuffled_lane)
    phase = lane_to_phase.get(lane, -1)
    phys_row_plot = lane % num_rows
    for i in range(banks_per_instr):
        b_shuf = (start_shuf + i) % banks
        shuf_grid[phys_row_plot, b_shuf] = phase
        shuf_labels[phys_row_plot][b_shuf].append(lane)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(25, 12), constrained_layout=True)

def draw(ax, grid, labels, title):
    bg = np.ones_like(grid, dtype=float)
    ax.imshow(bg, cmap=ListedColormap(["#efefef"]),
              extent=(-0.5, banks-0.5, num_rows-0.5, -0.5))
    im = ax.imshow(np.where(grid >= 0, grid, 0), cmap=phase_cmap,
                   interpolation='nearest', aspect='auto')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Bank index (0-31)")
    ax.set_ylabel("Row (lane % 8)")
    ax.set_xticks(range(0, banks, 2))
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels([f"Row {r}" for r in range(num_rows)])
    ax.set_xticks(np.arange(-0.5, banks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
    ax.grid(which="minor", color=(0,0,0,0.08), linewidth=0.5)
    for r in range(num_rows):
        for b in range(banks):
            if labels[r][b]:
                text = "/".join(map(str, sorted(labels[r][b])))
                # Highlight conflicts
                color = 'red' if len(labels[r][b]) > 1 else 'white'
                ax.text(b, r, text, ha='center', va='center',
                       color=color, fontsize=15, weight='bold')
    return im

im0 = draw(axs[0], orig_grid, orig_labels,
          "Original Layout (4-way conflicts in red)")
im1 = draw(axs[1], shuf_grid, shuf_labels,
          "XOR Preshuffled Layout (conflict-free!)")

# Colorbar
cbar = fig.colorbar(im1, ax=axs, fraction=0.046, pad=0.02,
                   ticks=range(len(read_phase_lanes)))
cbar.ax.set_ylabel("Phase")
cbar.set_ticklabels([f"P{p}" for p in range(len(read_phase_lanes))])

plt.show()
```

**Expected Result**:
- Top plot: Red text shows 4-way conflicts (multiple lanes per bank)
- Bottom plot: No conflicts! Each bank cell has only one lane ID

---

## Summary

### Key Takeaways

1. **LDS is fast but constrained**: 32 banks, 4 bytes each, 128 bytes/cycle total
2. **Bank conflicts serialize accesses**: Multiple threads → same bank → performance loss
3. **Phase groupings differ**: Write uses sequential lanes, read uses non-sequential
4. **Simple layouts cause problems**: Row-major may be conflict-free for writes but creates 4-way conflicts for reads
5. **XOR swizzling helps**: Permutes data layout to avoid conflicts without extra storage

### What's Next

This document covered the fundamentals of LDS bank conflicts. To actually implement conflict-free LDS access in CK Tile:

1. **Learn CK Tile tensor descriptors**: How to describe memory layouts
2. **Study coordinate transformations**: How XOR operations are encoded
3. **Understand distributed tensors**: How tiles map to threads
4. **Practice with examples**: Build conflict-free kernels step by step

See the **CK Tile tutorials** (Tutorial 11-13) for hands-on implementation using the CK Tile API.

---

## Further Reading

- AMD CDNA Architecture Whitepaper
- CK Tile Tutorial 11: XOR Test (bank conflict patterns)
- CK Tile Tutorial 13: Production XOR GEMM (complete implementation)
- `tutorial_11_xor_test/BANK_CONFLICT_SUMMARY.md` (in this repository)

---

## Appendix: Quick Reference

### Bank Formula
```
bank = (address_bytes / 4) % 32
```

### Write Phases (Sequential)
```
P0: 0-7    P1: 8-15   P2: 16-23  P3: 24-31
P4: 32-39  P5: 40-47  P6: 48-55  P7: 56-63
```

### Read Phases (Non-Sequential)
```
P0: 0-3,20-23    P1: 4-7,16-19    P2: 8-11,28-31   P3: 12-15,24-27
P4: 32-35,52-55  P5: 36-39,48-51  P6: 40-43,60-63  P7: 44-47,56-59
```

### XOR Permutation
```
x' = (row % num_cols) XOR column
```
