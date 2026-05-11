╔═══════════════════════════════════════════════════════════════════════════╗
║           STORAGE LAYOUT ALTERNATIVES & BANK CONFLICTS                    ║
║                  Concrete Examples with Numbers                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

This document explores different ways to store a matrix in LDS and shows
exactly what bank conflicts occur with each approach, using simple numbers.

We'll use the same constraints from LDS_CONSTRAINTS.md and apply them to
different storage strategies.

================================================================================
SETUP: Common Configuration for All Examples
================================================================================

Matrix: 8 rows × 32 columns = 256 elements total
Element type: FP16 (2 bytes each)
Row width: 32 elements × 2 bytes = 64 bytes per row
Each lane reads/writes: 8 elements = 16 bytes (ds_read_b128/ds_write_b128)

Hardware constraints (from LDS_CONSTRAINTS.md):
  - 32 banks total
  - Each bank: 4 bytes wide
  - Total bandwidth: 128 bytes/cycle
  - Each lane: accesses 4 consecutive banks (16 bytes ÷ 4 bytes/bank)
  - Only 8 lanes execute per cycle (128 bytes ÷ 16 bytes/lane)
  - Bank formula: bank = (address_bytes / 4) % 32

Read phase grouping (non-sequential):
  Phase 0: lanes {0, 1, 2, 3, 20, 21, 22, 23}  (we'll use lanes 0-3 for simplicity)

WHY THIS SIZE?
  - 64-byte row width is realistic for ML workloads
  - Stride of 64 bytes = exactly half of bank period (128 bytes)
  - This creates the bank conflict pattern we want to demonstrate
  - Smaller matrices (16-byte rows) won't show conflicts clearly

================================================================================
EXAMPLE 1: ROW-MAJOR LAYOUT (Standard Sequential)
================================================================================

Storage pattern:
  Row 0: columns [0-31]   stored at addresses [0-63]     (32 elem × 2 bytes = 64 bytes)
  Row 1: columns [0-31]   stored at addresses [64-127]
  Row 2: columns [0-31]   stored at addresses [128-191]
  Row 3: columns [0-31]   stored at addresses [192-255]
  Row 4: columns [0-31]   stored at addresses [256-319]
  Row 5: columns [0-31]   stored at addresses [320-383]
  Row 6: columns [0-31]   stored at addresses [384-447]
  Row 7: columns [0-31]   stored at addresses [448-511]

Memory layout:
┌──────────────────────────────────────────────────────────────────────┐
│ Addr   0-63:  [row0_col0, row0_col1, ..., row0_col31]  (64 bytes)  │
│ Addr  64-127: [row1_col0, row1_col1, ..., row1_col31]  (64 bytes)  │
│ Addr 128-191: [row2_col0, row2_col1, ..., row2_col31]  (64 bytes)  │
│ Addr 192-255: [row3_col0, row3_col1, ..., row3_col31]  (64 bytes)  │
│ ...                                                                  │
└──────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
SCENARIO 1A: WRITING ROWS (Sequential Access)
--------------------------------------------------------------------------------

Each lane writes 8 elements = 16 bytes.
In phase 0, we have 8 lanes, so they write 8 rows total (simplified).

Lane 0 writes first 16 bytes of row 0 (addresses 0-15):
  Starting bank: (0/4) % 32 = 0
  Banks: {0, 1, 2, 3}  (4 consecutive banks for 16 bytes)
  Each bank used ONCE → ✓ NO CONFLICT!

Lane 1 writes first 16 bytes of row 1 (addresses 64-79):
  Starting bank: (64/4) % 32 = 16
  Banks: {16, 17, 18, 19}
  Each bank used ONCE → ✓ NO CONFLICT!

Lane 2 writes first 16 bytes of row 2 (addresses 128-143):
  Starting bank: (128/4) % 32 = 0
  Banks: {0, 1, 2, 3}
  ✓ NO CONFLICT! (Lane 0 and Lane 2 are in different phases)

Lane 3 writes first 16 bytes of row 3 (addresses 192-207):
  Starting bank: (192/4) % 32 = 16
  Banks: {16, 17, 18, 19}
  ✓ NO CONFLICT! (Lane 1 and Lane 3 are in different phases)

RESULT FOR WRITING: ✓ Conflict-free!
All lanes in phase 0 access different banks.

--------------------------------------------------------------------------------
SCENARIO 1B: READING COLUMNS (Transpose/Strided Access)
--------------------------------------------------------------------------------

Now read COLUMN 0 (transposed access):
  Need: elem[0][0], elem[1][0], elem[2][0], elem[3][0], elem[4][0], elem[5][0], elem[6][0], elem[7][0]

Lane 0 reads first 8 elements of column 0:

Step 1: Calculate addresses
  Remember: row width = 64 bytes, each element = 2 bytes

  elem[0][0] at row 0, col 0 → address = 0 × 64 + 0 × 2 = 0
  elem[1][0] at row 1, col 0 → address = 1 × 64 + 0 × 2 = 64
  elem[2][0] at row 2, col 0 → address = 2 × 64 + 0 × 2 = 128
  elem[3][0] at row 3, col 0 → address = 3 × 64 + 0 × 2 = 192
  elem[4][0] at row 4, col 0 → address = 4 × 64 + 0 × 2 = 256
  elem[5][0] at row 5, col 0 → address = 5 × 64 + 0 × 2 = 320
  elem[6][0] at row 6, col 0 → address = 6 × 64 + 0 × 2 = 384
  elem[7][0] at row 7, col 0 → address = 7 × 64 + 0 × 2 = 448

  Stride = 64 bytes between consecutive elements!

Step 2: Calculate banks
  Address 0:   bank = (0/4) % 32   = 0
  Address 64:  bank = (64/4) % 32  = 16
  Address 128: bank = (128/4) % 32 = 32 % 32 = 0  ← REPEATS bank 0!
  Address 192: bank = (192/4) % 32 = 48 % 32 = 16 ← REPEATS bank 16!
  Address 256: bank = (256/4) % 32 = 64 % 32 = 0  ← REPEATS bank 0!
  Address 320: bank = (320/4) % 32 = 80 % 32 = 16 ← REPEATS bank 16!
  Address 384: bank = (384/4) % 32 = 96 % 32 = 0  ← REPEATS bank 0!
  Address 448: bank = (448/4) % 32 = 112 % 32 = 16 ← REPEATS bank 16!

  Pattern: {0, 16, 0, 16, 0, 16, 0, 16}

Step 3: Count bank usage
  Bank 0:  used 4 times (addresses 0, 128, 256, 384)
  Bank 16: used 4 times (addresses 64, 192, 320, 448)

Step 4: Conflict detected!
  Lane 0's SINGLE ds_read_b128 instruction needs:
    - Bank 0:  FOUR separate accesses
    - Bank 16: FOUR separate accesses

  Hardware must serialize:
    Cycle 1: Read from bank 0  (gets element from address 0)
    Cycle 2: Read from bank 0  (gets element from address 128)
    Cycle 3: Read from bank 0  (gets element from address 256)
    Cycle 4: Read from bank 0  (gets element from address 384)
    (Same pattern for bank 16)

  ✗ 4-WAY BANK CONFLICT!

EXPLANATION:
  Stride = 64 bytes (row width)
  Bank period = 128 bytes (32 banks × 4 bytes)

  Since 64 = 128/2, we alternate between only 2 banks!
  Each bank needed 4 times → hardware serializes into 4 cycles per bank.

RESULT FOR TRANSPOSE READ: ✗ Severe 4-way conflicts!

================================================================================
EXAMPLE 2: COLUMN-MAJOR LAYOUT
================================================================================

What if we store data column-by-column instead?

Storage pattern (8 rows × 32 cols):
  Col 0: 8 rows [row0-7, col0]   at addresses [0-15]     (8 elem × 2 bytes = 16 bytes)
  Col 1: 8 rows [row0-7, col1]   at addresses [16-31]
  Col 2: 8 rows [row0-7, col2]   at addresses [32-47]
  ...
  Col 31: 8 rows [row0-7, col31] at addresses [496-511]

Memory layout:
┌──────────────────────────────────────────────────────────────────────┐
│ Addr   0-15:  [row0_col0, row1_col0, ..., row7_col0]  (16 bytes)   │
│ Addr  16-31:  [row0_col1, row1_col1, ..., row7_col1]  (16 bytes)   │
│ Addr  32-47:  [row0_col2, row1_col2, ..., row7_col2]  (16 bytes)   │
│ ...                                                                  │
└──────────────────────────────────────────────────────────────────────┘

--------------------------------------------------------------------------------
SCENARIO 2A: READING COLUMNS (Sequential in this layout)
--------------------------------------------------------------------------------

Lane 0 reads column 0 (addresses 0-15):
  Starting bank: (0/4) % 32 = 0
  Banks: {0, 1, 2, 3}
  ✓ NO CONFLICT!

Lane 1 reads column 1 (addresses 16-31):
  Starting bank: (16/4) % 32 = 4
  Banks: {4, 5, 6, 7}
  ✓ NO CONFLICT!

RESULT: ✓ Transpose reads are now conflict-free!

--------------------------------------------------------------------------------
SCENARIO 2B: READING ROWS (Strided in this layout)
--------------------------------------------------------------------------------

Now to read ROW 0, we need:
  elem[0][0], elem[0][1], elem[0][2], elem[0][3], elem[0][4], elem[0][5], elem[0][6], elem[0][7]

These are stored at:
  elem[0][0] in col 0 → address = 0 + 0×2 = 0
  elem[0][1] in col 1 → address = 16 + 0×2 = 16
  elem[0][2] in col 2 → address = 32 + 0×2 = 32
  elem[0][3] in col 3 → address = 48 + 0×2 = 48
  elem[0][4] in col 4 → address = 64 + 0×2 = 64
  elem[0][5] in col 5 → address = 80 + 0×2 = 80
  elem[0][6] in col 6 → address = 96 + 0×2 = 96
  elem[0][7] in col 7 → address = 112 + 0×2 = 112

Banks:
  Address 0:   bank = 0
  Address 16:  bank = 4
  Address 32:  bank = 8
  Address 48:  bank = 12
  Address 64:  bank = 16
  Address 80:  bank = 20
  Address 96:  bank = 24
  Address 112: bank = 28

  Pattern: {0, 4, 8, 12, 16, 20, 24, 28}

  Each bank used once → ✓ NO CONFLICT!

RESULT: ✓ Row reads are conflict-free with 16-byte stride!

CONCLUSION: Column-major is OPPOSITE of row-major:
  - Row-major: good for row writes/reads, bad for column reads (transpose)
  - Column-major: good for column reads (transpose), bad for row access

Neither is universally better! Depends on access pattern.

================================================================================
EXAMPLE 3: ROW-MAJOR WITH PADDING
================================================================================

Idea: Add padding to each row to shift bank alignment.

Row width = 32 elements × 2 bytes = 64 bytes
Add 32 bytes padding per row → total stride = 96 bytes

Storage pattern:
  Row 0: addresses [0-63],     padding [64-95]
  Row 1: addresses [96-159],   padding [160-191]
  Row 2: addresses [192-255],  padding [256-287]
  Row 3: addresses [288-351],  padding [352-383]
  ...

--------------------------------------------------------------------------------
Reading COLUMN 0 with 96-byte stride:
--------------------------------------------------------------------------------

  elem[0][0]: address = 0 × 96 + 0 = 0    → bank = 0
  elem[1][0]: address = 1 × 96 + 0 = 96   → bank = (96/4) % 32 = 24
  elem[2][0]: address = 2 × 96 + 0 = 192  → bank = (192/4) % 32 = 16
  elem[3][0]: address = 3 × 96 + 0 = 288  → bank = (288/4) % 32 = 8
  elem[4][0]: address = 4 × 96 + 0 = 384  → bank = (384/4) % 32 = 0  ← REPEAT!
  elem[5][0]: address = 5 × 96 + 0 = 480  → bank = (480/4) % 32 = 24 ← REPEAT!
  elem[6][0]: address = 6 × 96 + 0 = 576  → bank = (576/4) % 32 = 16 ← REPEAT!
  elem[7][0]: address = 7 × 96 + 0 = 672  → bank = (672/4) % 32 = 8  ← REPEAT!

  Pattern: {0, 24, 16, 8, 0, 24, 16, 8}

  Bank 0:  used 2 times
  Bank 8:  used 2 times
  Bank 16: used 2 times
  Bank 24: used 2 times

  ✗ 2-WAY CONFLICT!

IMPROVEMENT: Reduced from 4-way to 2-way!

But cost:
  - Wasted 32 bytes per row
  - For 64 rows: 64 × 32 = 2048 bytes wasted
  - LDS is precious (only 64 KB total)

ANALYSIS:
  Stride = 96 bytes
  Bank period = 128 bytes
  GCD(96, 128) = 32 → repeats every 32 bytes = 8 banks
  96 / GCD = 3 bank positions per row
  Pattern repeats every LCM(96, 128) = 384 bytes

  Not perfect, but better than no padding!

================================================================================
EXAMPLE 4: XOR SWIZZLING (Our Solution)
================================================================================

Keep row-major layout (64-byte rows, no padding) but permute addresses using XOR.

XOR formula (simplified):
  x' = (row_id % 8) XOR col_id

This permutes the column index based on row, spreading accesses across banks.

--------------------------------------------------------------------------------
WITHOUT XOR (from Example 1):
--------------------------------------------------------------------------------
  Column 0 access: banks {0, 16, 0, 16, 0, 16, 0, 16}
  → 4-way conflict (only 2 unique banks)

--------------------------------------------------------------------------------
WITH XOR:
--------------------------------------------------------------------------------

Reading column 0, the XOR permutation changes which physical column we access:

  Row 0: x' = (0 % 8) XOR 0 = 0 XOR 0 = 0 → physical col 0 → address 0
  Row 1: x' = (1 % 8) XOR 0 = 1 XOR 0 = 1 → physical col 1 → address 64 + 2 = 66
  Row 2: x' = (2 % 8) XOR 0 = 2 XOR 0 = 2 → physical col 2 → address 128 + 4 = 132
  Row 3: x' = (3 % 8) XOR 0 = 3 XOR 0 = 3 → physical col 3 → address 192 + 6 = 198
  Row 4: x' = (4 % 8) XOR 0 = 4 XOR 0 = 4 → physical col 4 → address 256 + 8 = 264
  Row 5: x' = (5 % 8) XOR 0 = 5 XOR 0 = 5 → physical col 5 → address 320 + 10 = 330
  Row 6: x' = (6 % 8) XOR 0 = 6 XOR 0 = 6 → physical col 6 → address 384 + 12 = 396
  Row 7: x' = (7 % 8) XOR 0 = 7 XOR 0 = 7 → physical col 7 → address 448 + 14 = 462

Banks:
  Address 0:   bank = 0
  Address 66:  bank = (66/4) % 32 = 16
  Address 132: bank = (132/4) % 32 = 1
  Address 198: bank = (198/4) % 32 = 17
  Address 264: bank = (264/4) % 32 = 2
  Address 330: bank = (330/4) % 32 = 18
  Address 396: bank = (396/4) % 32 = 3
  Address 462: bank = (462/4) % 32 = 19

  Pattern: {0, 16, 1, 17, 2, 18, 3, 19}

  All 8 banks are UNIQUE! ✓ NO CONFLICT!

RESULT: ✓ Perfect distribution across banks!

BENEFITS:
  - No wasted LDS space (no padding)
  - Spreads accesses across all 32 banks
  - Works for both read and write patterns
  - Mathematical permutation ensures even distribution

================================================================================
SUMMARY COMPARISON TABLE
================================================================================

┌────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
│ Storage Layout     │ Row Access   │ Column Access│ LDS Waste│ Performance │
├────────────────────┼──────────────┼──────────────┼──────────┼─────────────┤
│ Row-major          │ ✓ No conflict│ ✗ 4-way      │ 0%       │ Good writes │
│ (sequential)       │              │   conflict   │          │ Bad reads   │
│                    │              │              │          │             │
│ Column-major       │ ✓ No conflict│ ✓ No conflict│ 0%       │ Opposite of │
│                    │ (if 16B      │              │          │ row-major   │
│                    │  stride)     │              │          │             │
│                    │              │              │          │             │
│ Row-major +        │ ✓ No conflict│ ✗ 2-way      │ ~33%     │ Better but  │
│ 32B padding        │              │   conflict   │ (large!) │ wasteful    │
│                    │              │              │          │             │
│ Row-major +        │ ✓ No conflict│ ✓ No conflict│ 0%       │ Best of     │
│ XOR swizzle        │              │ (or ~2-way)  │          │ all worlds! │
└────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘

================================================================================
KEY INSIGHTS
================================================================================

1. **Bank conflict = stride pattern issue**
   - When stride is a divisor of bank period (128 bytes), conflicts occur
   - 64-byte stride → exactly half of 128 → alternates between 2 banks
   - Pattern repeats every GCD(stride, 128) bytes

2. **No universal "best" layout**
   - Row-major: good for sequential row access, bad for column access
   - Column-major: opposite tradeoffs
   - Depends on kernel's dominant access pattern

3. **Padding helps but wastes space**
   - Can reduce conflicts (4-way → 2-way with right padding)
   - LDS is scarce (64 KB total on CDNA)
   - Often not worth the space cost

4. **XOR swizzling is optimal**
   - Permutes addresses mathematically
   - Spreads strided accesses across all banks
   - Zero space overhead
   - Works for multiple access patterns simultaneously

5. **Constraint interaction matters**
   - Must consider ALL constraints together:
     * Bank assignment formula
     * Lane grouping in phases
     * 16-byte access size (4 banks)
     * Stride between elements
   - Simple analysis misses subtle interactions!

================================================================================
PRACTICAL RECOMMENDATION
================================================================================

For GEMM and ML workloads:
  1. Use row-major for matrices (matches global memory)
  2. Apply XOR swizzling for transpose access
  3. Avoid padding (wastes precious LDS)
  4. Test actual hardware performance (theory ≠ practice)

For other access patterns:
  1. Analyze the dominant stride
  2. Calculate bank access pattern
  3. Check for repeating bank usage
  4. Apply XOR or choose layout accordingly

╚═══════════════════════════════════════════════════════════════════════════╝
