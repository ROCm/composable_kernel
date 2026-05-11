# Understanding Phase Grouping: Why Not All 64 Lanes Execute Simultaneously

## The Core Question

**Why can't all 64 lanes in a wavefront access LDS at the same time?**

The answer: **Hardware bandwidth limitations.**

## Hardware Constraint

**LDS bandwidth: 128 bytes per cycle** (32 banks × 4 bytes each)

**But:**
- 64 lanes in a wavefront
- Each lane wants to read/write 16 bytes (for `ds_read_b128` / `ds_write_b128`)
- Total demand: 64 × 16 = **1024 bytes**

**Problem:** 1024 bytes > 128 bytes bandwidth!

**Solution:** Hardware divides the 64 lanes into **phases** and executes them sequentially.

```
1024 bytes total ÷ 128 bytes per cycle = 8 phases required
```

## Phase Division for Write (ds_write_b128)

**SEQUENTIAL grouping** - easiest to understand:

```
Phase 0: Lanes 0-7    (8 lanes × 16 bytes = 128 bytes) ✓ Fits in one cycle
Phase 1: Lanes 8-15   (8 lanes × 16 bytes = 128 bytes) ✓ Fits in one cycle
Phase 2: Lanes 16-23  (8 lanes × 16 bytes = 128 bytes) ✓ Fits in one cycle
Phase 3: Lanes 24-31
Phase 4: Lanes 32-39
Phase 5: Lanes 40-47
Phase 6: Lanes 48-55
Phase 7: Lanes 56-63
```

**Each phase executes in ONE cycle (if no conflicts).**

Total write time: 8 cycles minimum (one per phase).

## Phase Division for Read (ds_read_b128)

**NON-SEQUENTIAL grouping** - hardware-specific pattern:

```
Phase 0: Lanes {0, 1, 2, 3, 20, 21, 22, 23}    (8 lanes total)
Phase 1: Lanes {4, 5, 6, 7, 16, 17, 18, 19}    (8 lanes total)
Phase 2: Lanes {8, 9, 10, 11, 28, 29, 30, 31}  (8 lanes total)
Phase 3: Lanes {12, 13, 14, 15, 24, 25, 26, 27}
Phase 4: Lanes {32, 33, 34, 35, 52, 53, 54, 55}
Phase 5: Lanes {36, 37, 38, 39, 48, 49, 50, 51}
Phase 6: Lanes {40, 41, 42, 43, 60, 61, 62, 63}
Phase 7: Lanes {44, 45, 46, 47, 56, 57, 58, 59}
```

**Why non-sequential?** Hardware scheduling optimization (AMD-specific).

Each phase still executes 8 lanes = 128 bytes, fitting in the bandwidth.

## Bank Conflict Check: Per-Phase Only!

**CRITICAL UNDERSTANDING:**

Bank conflicts are checked **ONLY within each phase**, not across all 64 lanes!

### Example: Write Phase 0

```
Phase 0: Lanes 0-7 executing simultaneously

Lane 0: writes to addresses 0-15    → banks 0-3
Lane 1: writes to addresses 16-31   → banks 4-7
Lane 2: writes to addresses 32-47   → banks 8-11
Lane 3: writes to addresses 48-63   → banks 12-15
Lane 4: writes to addresses 64-79   → banks 16-19
Lane 5: writes to addresses 80-95   → banks 20-23
Lane 6: writes to addresses 96-111  → banks 24-27
Lane 7: writes to addresses 112-127 → banks 28-31
```

**Result:** All 8 lanes access DIFFERENT banks → **NO CONFLICT** ✓

This phase completes in 1 cycle.

### Example: Read Phase 0 (Transpose)

```
Phase 0: Lanes {0, 1, 2, 3, 20, 21, 22, 23}

Reading column 0 (transposed row 0):
Lane 0: reads rows 0-7,  col 0  → addresses {0, 64, 128, 192, 256, 320, 384, 448}
                                → banks {0, 16, 0, 16, 0, 16, 0, 16}

Reading column 1 (transposed row 1):
Lane 1: reads rows 8-15, col 1  → addresses {... similar pattern}
                                → banks {2, 18, 2, 18, 2, 18, 2, 18}

... (other lanes)
```

**Within Lane 0's single ds_read_b128 instruction:**
- Needs bank 0: 4 times
- Needs bank 16: 4 times
- **Conflict!** Hardware must serialize this into 4 accesses.

**This is a WITHIN-LANE conflict** (lane 0 conflicting with itself).

Even though lanes 0, 1, 2, 3 don't conflict with each other (they access different columns), **each individual lane has internal conflicts** due to the stride pattern.

## Why This Matters for Transpose

### Writing [M, K] - Sequential Access

```
Lane 0: writes row 0, elements 0-7   → sequential addresses → 4 consecutive banks
Lane 1: writes row 0, elements 8-15  → sequential addresses → next 4 banks
Lane 2: writes row 0, elements 16-23 → sequential addresses → next 4 banks
...
```

**No conflicts because:**
1. Within a phase, lanes write to different parts of the same row
2. Sequential addresses = consecutive banks
3. Each lane's 4 banks don't overlap with other lanes' 4 banks

### Reading [K, M] - Strided Access

```
Lane 0: reads column 0, rows 0-7     → strided addresses (stride = 64 bytes)
                                     → {bank 0, bank 16, bank 0, bank 16, ...}
```

**Conflict because:**
1. Stride (64 bytes) = 16 bank offset
2. After 2 elements (128 bytes), we wrap back to same banks
3. Lane 0's 8 reads hit only 2 banks, 4 times each
4. **This is a conflict WITHIN the single lane's access**

## Visual Example: Phase 0 Execution

**Write Phase 0 (conflict-free):**
```
       Banks:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 ...
Lane 0:       [■][■][■][■]
Lane 1:                   [■][■][■][■]
Lane 2:                               [■][■][■][■]
Lane 3:                                           [■][■][■][■]
Lane 4:                                                       [■][■][■][■]
...

All different banks → Execute in 1 cycle ✓
```

**Read Phase 0 (conflicts):**
```
       Banks:  0  1  2  3  4  5 ... 16 17 18 19 20 ...
Lane 0:       [■]         .        [■]               ← Reading col 0
              [■]         .        [■]               ← 4 times bank 0
              [■]         .        [■]               ← 4 times bank 16
              [■]         .        [■]

CONFLICT! Lane 0 needs bank 0 four times → serialize into 4 accesses
```

## Summary

1. **Hardware can only execute 8 lanes simultaneously** (128 bytes bandwidth)
2. **64 lanes ÷ 8 lanes per phase = 8 phases**
3. **Bank conflicts are checked PER PHASE** (among the 8 executing lanes)
4. **For transpose:**
   - Write: sequential access → each lane hits different banks → conflict-free
   - Read: strided access → each lane hits SAME banks repeatedly → conflicts!
5. **XOR swizzling:** Breaks the stride pattern so each lane hits more diverse banks

## Key Insight

The problem isn't lanes conflicting with OTHER lanes in the same phase.

The problem is **EACH INDIVIDUAL LANE conflicting with ITSELF** because the strided access pattern makes that single lane try to access the same bank multiple times within one instruction!

```
Lane 0's ds_read_b128 instruction tries to load 16 bytes from addresses:
{0, 64, 128, 192, 256, 320, 384, 448}

These map to banks:
{0, 16, 0, 16, 0, 16, 0, 16}

Hardware sees: "This lane needs bank 0 four times in one instruction!"
→ Must serialize into 4 separate accesses
```
