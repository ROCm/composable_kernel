# Complete LDS Access Constraints

## Hardware Architecture Constraints

### Constraint 1: LDS Bank Structure
```
Total banks:        32 banks
Bank width:         4 bytes each
Total bandwidth:    32 × 4 = 128 bytes per cycle
```

**What this means:**
- In one cycle, the LDS can provide at most 128 bytes total
- This is shared among ALL lanes executing in that cycle

### Constraint 2: Bank Address Mapping
```
Formula: bank_id = (address_in_bytes / 4) % 32

Example:
  Address 0   → bank 0
  Address 4   → bank 1
  Address 8   → bank 2
  Address 128 → bank 0  (wraps: (128/4) % 32 = 32 % 32 = 0)
```

**What this means:**
- Every 4 bytes maps to the next bank
- After 128 bytes (32 banks × 4), banks wrap around to 0

### Constraint 3: Wavefront Structure
```
Wavefront size:     64 lanes (threads)
Execution model:    SIMD (all lanes execute same instruction)
```

**What this means:**
- All 64 lanes try to execute the same LDS instruction simultaneously
- But hardware must divide them into phases due to bandwidth limits

## Per-Lane Constraints

### Constraint 4: Instruction Sizes
```
Instruction         | Bytes per Lane | Banks per Lane | Total for 64 lanes
--------------------|----------------|----------------|-------------------
ds_read_b32         | 4 bytes        | 1 bank         | 256 bytes
ds_read_b64         | 8 bytes        | 2 banks        | 512 bytes
ds_read_b128        | 16 bytes       | 4 banks        | 1024 bytes
ds_write_b32        | 4 bytes        | 1 bank         | 256 bytes
ds_write_b64        | 8 bytes        | 2 banks        | 512 bytes
ds_write_b128       | 16 bytes       | 4 banks        | 1024 bytes
```

**For ds_read_b128 / ds_write_b128 (most common in ML workloads):**
- Each lane accesses: **16 bytes**
- Each lane uses: **4 consecutive banks**
- All 64 lanes need: **1024 bytes total**

### Constraint 5: Banks Accessed by One Lane
```
For ds_read_b128 at address A:

Lane accesses addresses: [A, A+1, A+2, ..., A+15]
This spans:              4 consecutive banks

Example:
  Lane reads from address 0:
    Bytes 0-3   → bank 0
    Bytes 4-7   → bank 1
    Bytes 8-11  → bank 2
    Bytes 12-15 → bank 3

  Lane reads from address 16:
    Bytes 16-19 → bank 4
    Bytes 20-23 → bank 5
    Bytes 24-27 → bank 6
    Bytes 28-31 → bank 7
```

**CRITICAL:** A single lane's 16-byte access spans 4 **consecutive** banks starting at `(address/4) % 32`.

### Constraint 6: Maximum Banks Per Lane Per Instruction
```
Answer: 4 banks (for ds_read_b128 / ds_write_b128)

BUT: These must be 4 CONSECUTIVE banks in physical layout.
```

**What this means:**
- A lane CANNOT arbitrarily choose which 4 banks to access
- The banks are determined by the starting address
- If address = 0: accesses banks {0, 1, 2, 3}
- If address = 64: accesses banks {16, 17, 18, 19}
- If address = 128: accesses banks {0, 1, 2, 3} again

### Constraint 7: Bank Conflicts Within One Lane

**QUESTION: What if a lane needs the same bank multiple times?**

**ANSWER: Hardware CANNOT do this in parallel - must serialize!**

Example (transpose read):
```
Lane 0 reading column 0, rows 0-7:

  Element from row 0: address 0   → needs bank 0 (bytes 0-3)
  Element from row 1: address 64  → needs bank 16 (bytes 64-67)
  Element from row 2: address 128 → needs bank 0 (bytes 128-131)  ← CONFLICT!
  Element from row 3: address 192 → needs bank 16 (bytes 192-195) ← CONFLICT!
  ...
```

**The lane needs bank 0 FOUR times in one instruction!**

**Hardware solution: Serialize into 4 separate reads:**
```
Cycle 1: Read bank 0  → gets byte 0-3    (from row 0)
Cycle 2: Read bank 0  → gets byte 128-131 (from row 2)
Cycle 3: Read bank 0  → gets byte 256-259 (from row 4)
Cycle 4: Read bank 0  → gets byte 384-387 (from row 6)
```

This is a **4-way bank conflict within a single lane**.

## Phase Execution Constraints

### Constraint 8: Bandwidth Limitation
```
Available: 128 bytes/cycle (32 banks × 4 bytes)
Needed:    1024 bytes for all 64 lanes (64 × 16 bytes)

Solution: Divide into phases
  1024 bytes / 128 bytes per cycle = 8 phases required
```

### Constraint 9: Lanes Per Phase
```
For ds_read_b128 / ds_write_b128:

Each phase can execute:  8 lanes
  (8 lanes × 16 bytes = 128 bytes = exactly the bandwidth)

Total phases needed: 64 lanes / 8 lanes per phase = 8 phases
```

### Constraint 10: Phase Groupings

**Write phases (sequential):**
```
Phase 0: Lanes 0-7
Phase 1: Lanes 8-15
Phase 2: Lanes 16-23
Phase 3: Lanes 24-31
Phase 4: Lanes 32-39
Phase 5: Lanes 40-47
Phase 6: Lanes 48-55
Phase 7: Lanes 56-63
```

**Read phases (non-sequential - AMD GCN/CDNA):**
```
Phase 0: Lanes {0, 1, 2, 3, 20, 21, 22, 23}
Phase 1: Lanes {4, 5, 6, 7, 16, 17, 18, 19}
Phase 2: Lanes {8, 9, 10, 11, 28, 29, 30, 31}
Phase 3: Lanes {12, 13, 14, 15, 24, 25, 26, 27}
Phase 4: Lanes {32, 33, 34, 35, 52, 53, 54, 55}
Phase 5: Lanes {36, 37, 38, 39, 48, 49, 50, 51}
Phase 6: Lanes {40, 41, 42, 43, 60, 61, 62, 63}
Phase 7: Lanes {44, 45, 46, 47, 56, 57, 58, 59}
```

### Constraint 11: Bank Conflicts Are Checked Per-Phase

**CRITICAL:** Bank conflicts are ONLY checked among:
1. The 8 lanes executing in the current phase
2. Within each individual lane's access

**NOT checked:**
- Between different phases (they execute at different times)
- Between lanes in different phases

## Conflict Detection Constraints

### Constraint 12: Conflict-Free Execution Within a Phase

**For Phase 0 to execute in 1 cycle (conflict-free):**

**Between lanes in the phase:**
- All 8 lanes must access DIFFERENT sets of banks
- If any two lanes try to access the same bank → conflict!

**Within each lane:**
- Each lane's 4 banks must be UNIQUE (no repeated banks)
- If a lane needs the same bank twice → conflict!

### Constraint 13: Example - Write Phase 0 (Conflict-Free)

```
Writing sequential data (row-major):

Lane 0: address 0-15     → banks {0, 1, 2, 3}
Lane 1: address 16-31    → banks {4, 5, 6, 7}
Lane 2: address 32-47    → banks {8, 9, 10, 11}
Lane 3: address 48-63    → banks {12, 13, 14, 15}
Lane 4: address 64-79    → banks {16, 17, 18, 19}
Lane 5: address 80-95    → banks {20, 21, 22, 23}
Lane 6: address 96-111   → banks {24, 25, 26, 27}
Lane 7: address 112-127  → banks {28, 29, 30, 31}

Check:
  ✓ Each lane uses 4 different banks (within-lane: OK)
  ✓ No bank appears in multiple lanes (between-lanes: OK)
  ✓ Total: 32 banks used, 128 bytes accessed

Result: Executes in 1 cycle!
```

### Constraint 14: Example - Transpose Read (4-Way Conflict)

```
Reading column 0 (stride = 64 bytes), Lane 0:

Lane 0 needs addresses: {0, 64, 128, 192, 256, 320, 384, 448}

Calculate which banks:
  Address 0:   byte 0   → bank 0
  Address 64:  byte 64  → bank (64/4)%32 = 16
  Address 128: byte 128 → bank (128/4)%32 = 0  ← CONFLICT!
  Address 192: byte 192 → bank (192/4)%32 = 16 ← CONFLICT!
  Address 256: byte 256 → bank (256/4)%32 = 0  ← CONFLICT!
  Address 320: byte 320 → bank (320/4)%32 = 16 ← CONFLICT!
  Address 384: byte 384 → bank (384/4)%32 = 0  ← CONFLICT!
  Address 448: byte 448 → bank (448/4)%32 = 16 ← CONFLICT!

Bank usage:
  Bank 0:  4 accesses (bytes 0, 128, 256, 384)
  Bank 16: 4 accesses (bytes 64, 192, 320, 448)

Check:
  ✗ Lane needs bank 0 FOUR times (within-lane conflict!)
  ✗ Lane needs bank 16 FOUR times (within-lane conflict!)

Result: 4-way conflict! Executes in 4 cycles instead of 1.
```

## Summary Table: All Constraints

| Constraint | Value | Unit |
|------------|-------|------|
| **LDS Structure** |
| Total banks | 32 | banks |
| Bank width | 4 | bytes |
| Total bandwidth | 128 | bytes/cycle |
| Bank wrap period | 128 | bytes |
| **Wavefront** |
| Total lanes | 64 | lanes |
| Execution model | SIMD | - |
| **Per-Lane (ds_read_b128)** |
| Bytes per lane | 16 | bytes |
| Banks per lane | 4 | consecutive banks |
| Bank selection | Determined by address | - |
| Max banks per instruction | 4 | banks |
| Banks must be | Consecutive | - |
| **Phase Execution** |
| Lanes per phase | 8 | lanes |
| Phases per wavefront | 8 | phases |
| Bytes per phase | 128 | bytes |
| Cycles per phase (no conflict) | 1 | cycle |
| Cycles per phase (N-way conflict) | N | cycles |
| **Conflict Detection** |
| Checked between | Lanes in same phase | - |
| Checked within | Each lane's access | - |
| NOT checked between | Different phases | - |

## Key Takeaways

1. **Each lane accesses 4 consecutive banks** (for 16-byte instruction)
2. **Banks are determined by starting address**, not freely chosen
3. **Only 8 lanes execute per cycle** (bandwidth limit)
4. **Conflicts happen when a lane needs the same bank multiple times**
5. **Transpose creates strided access → same banks repeat → conflicts!**
6. **XOR swizzling changes address mapping to spread accesses across more banks**

## Example Calculation: Why Transpose Has Conflicts

**Given:**
- Matrix: 64 rows × 32 columns, FP16 (2 bytes)
- Row width: 32 × 2 = 64 bytes
- Reading column 0 (stride = 64 bytes)

**Lane 0 reads 8 elements from column 0:**
```
Row 0: address 0     → bank (0/4)%32   = 0
Row 1: address 64    → bank (64/4)%32  = 16
Row 2: address 128   → bank (128/4)%32 = 0   ← Same as row 0!
Row 3: address 192   → bank (192/4)%32 = 16  ← Same as row 1!
Row 4: address 256   → bank (256/4)%32 = 0   ← Conflict!
Row 5: address 320   → bank (320/4)%32 = 16  ← Conflict!
Row 6: address 384   → bank (384/4)%32 = 0   ← Conflict!
Row 7: address 448   → bank (448/4)%32 = 16  ← Conflict!
```

**Pattern:** {0, 16, 0, 16, 0, 16, 0, 16}

**Why this pattern?**
- Stride (64 bytes) / 4 = 16 bank offset
- 16 banks forward from bank 0 → bank 16
- 16 banks forward from bank 16 → bank 32 % 32 = bank 0 (wraps!)

**Result:** Lane needs each of 2 banks (0 and 16) exactly 4 times → **4-way conflict!**

**Hardware must serialize:** 4 accesses instead of 1 → **4× slower!**
