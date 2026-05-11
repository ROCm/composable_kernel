================================================================================
CONCRETE EXAMPLE: Lane-by-Lane Transpose Reading
================================================================================

Configuration:
- Matrix: 64 rows × 32 columns (FP16 = 2 bytes each)
- Each row: 32 elements × 2 bytes = 64 bytes
- LDS size: 64 × 64 = 4096 bytes
- Each lane reads: 8 FP16 = 16 bytes = 4 banks worth

================================================================================
WRITE PHASE (Store [M=64, K=32] to LDS)
================================================================================

Physical memory layout (row-major, WITHOUT XOR first):

Address Range  | Content                    | Banks Used
---------------|----------------------------|------------------
0-63           | Row 0 (cols 0-31)         | 0-15
64-127         | Row 1 (cols 0-31)         | 16-31, 0-15
128-191        | Row 2 (cols 0-31)         | 0-15
192-255        | Row 3 (cols 0-31)         | 16-31, 0-15
256-319        | Row 4 (cols 0-31)         | 0-15
...

Write by lanes (sequential):
- Lanes 0-3:   Write row 0  (addresses 0-63)
- Lanes 4-7:   Write row 1  (addresses 64-127)
- Lanes 8-11:  Write row 2  (addresses 128-191)
- Lanes 12-15: Write row 3  (addresses 192-255)
...

================================================================================
TRANSPOSE READ (Read [K=32, M=64] from SAME LDS)
================================================================================

Now we read COLUMNS as ROWS.

Reading transposed "row 0" means reading COLUMN 0 from all 64 original rows:

Transposed row 0 = [elem[0][0], elem[1][0], elem[2][0], ..., elem[63][0]]
                    └─ 64 elements total from column 0 ─┘

================================================================================
LANE DISTRIBUTION FOR READING
================================================================================

The 64 elements are distributed among lanes. With kKPack=8:
- 64 elements ÷ 8 elements/lane = 8 lanes read this transposed row

Lane assignment (from distribution):
- Lane 0:  reads elements 0-7   of transposed row 0 = original rows 0-7,  col 0
- Lane 1:  reads elements 8-15  of transposed row 0 = original rows 8-15, col 0
- Lane 2:  reads elements 16-23 of transposed row 0 = original rows 16-23,col 0
- Lane 3:  reads elements 24-31 of transposed row 0 = original rows 24-31,col 0
- Lane 4:  reads elements 32-39 of transposed row 0 = original rows 32-39,col 0
- Lane 5:  reads elements 40-47 of transposed row 0 = original rows 40-47,col 0
- Lane 6:  reads elements 48-55 of transposed row 0 = original rows 48-55,col 0
- Lane 7:  reads elements 56-63 of transposed row 0 = original rows 56-63,col 0

================================================================================
DETAILED: LANE 0 READING
================================================================================

Lane 0 needs to read 8 FP16 elements from COLUMN 0, rows 0-7:

Element    | Original Position | Byte Address        | Bank
-----------|-------------------|---------------------|-------
elem[0][0] | row 0, col 0     | 0*64 + 0*2 = 0     | 0
elem[1][0] | row 1, col 0     | 1*64 + 0*2 = 64    | 16
elem[2][0] | row 2, col 0     | 2*64 + 0*2 = 128   | 0  ← CONFLICT!
elem[3][0] | row 3, col 0     | 3*64 + 0*2 = 192   | 16 ← CONFLICT!
elem[4][0] | row 4, col 0     | 4*64 + 0*2 = 256   | 0  ← CONFLICT!
elem[5][0] | row 5, col 0     | 5*64 + 0*2 = 320   | 16 ← CONFLICT!
elem[6][0] | row 6, col 0     | 6*64 + 0*2 = 384   | 0  ← CONFLICT!
elem[7][0] | row 7, col 0     | 7*64 + 0*2 = 448   | 16 ← CONFLICT!

Bank calculation: bank = (address_bytes / 4) % 32
- Address 0:   bank = (0/4) % 32 = 0
- Address 64:  bank = (64/4) % 32 = 16
- Address 128: bank = (128/4) % 32 = 0  ← Same as address 0!
- Address 192: bank = (192/4) % 32 = 16 ← Same as address 64!
- ...

Result: Lane 0 accesses:
  Bank 0:  4 times (addresses 0, 128, 256, 384)
  Bank 16: 4 times (addresses 64, 192, 320, 448)

This is a 4-WAY BANK CONFLICT!

The ds_read_b128 instruction must be serialized into 4 separate reads:
  Cycle 1: Read from bank 0  (gets 1st occurrence)
  Cycle 2: Read from bank 0  (gets 2nd occurrence)
  Cycle 3: Read from bank 0  (gets 3rd occurrence)
  Cycle 4: Read from bank 0  (gets 4th occurrence)
  
Then same for bank 16.

================================================================================
WHY THIS HAPPENS
================================================================================

Stride between consecutive elements = 64 bytes (one full row)
Banks wrap every 128 bytes (32 banks × 4 bytes)

Pattern:
  64 bytes = 16 bank stride
  After 2 rows (128 bytes), we're back to bank 0

This creates the pattern: {0, 16, 0, 16, 0, 16, 0, 16}

================================================================================
READING PHASE 0 (Multiple Lanes)
================================================================================

Phase 0 includes lanes: {0, 1, 2, 3, 20, 21, 22, 23}

If these lanes read different transposed rows (different columns):

Lane 0 (col 0):  banks {0, 16, 0, 16, 0, 16, 0, 16}  - 4-way conflict
Lane 1 (col 1):  banks {2, 18, 2, 18, 2, 18, 2, 18}  - 4-way conflict  
Lane 2 (col 2):  banks {4, 20, 4, 20, 4, 20, 4, 20}  - 4-way conflict
Lane 3 (col 3):  banks {6, 22, 6, 22, 6, 22, 6, 22}  - 4-way conflict

Each column access has the same conflict pattern!

================================================================================
WITH XOR SWIZZLING
================================================================================

The XOR descriptor permutes physical addresses:

Instead of:
  elem[0][0] → bank 0
  elem[1][0] → bank 16
  elem[2][0] → bank 0  (conflict!)
  
XOR gives:
  elem[0][0] → bank A
  elem[1][0] → bank B  (B ≠ A)
  elem[2][0] → bank C  (C ≠ A, potentially ≠ B)
  elem[3][0] → bank D  (different pattern)

The XOR formula distributes these across MORE banks, reducing conflicts
from 4-way (2 banks, 4 hits each) to ~2-way (many banks, fewer repeats).

================================================================================
KEY INSIGHT
================================================================================

Transpose reading creates STRIDED access patterns (stride = row width).
When stride is not a multiple of total bank bandwidth, we get conflicts.

XOR swizzling breaks the regular stride pattern by permuting addresses,
spreading accesses across all 32 banks instead of concentrating them.

Result: 57% reduction in bank conflicts! (from 4-way to ~2-way average)
