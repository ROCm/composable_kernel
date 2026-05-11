# LDS Bank Conflict Calculation - Current Understanding & Open Questions

## Profiler Measurements (Ground Truth)
```
WITHOUT XOR: 7,168 conflicts
WITH XOR:    3,072 conflicts
```

## The Problem

We're trying to build a model that correctly calculates bank conflicts to match the profiler. Multiple attempts have produced different results, and we need to reconcile the assembly evidence with the conflict counting model.

---

## Key Assembly Evidence

### WITHOUT XOR (`01_row_major-hip-amdgcn-amd-amdhsa-gfx942.s`)
```asm
ds_read_u16 v8,  v6
ds_read_u16 v9,  v6 offset:64
ds_read_u16 v10, v6 offset:128
ds_read_u16 v11, v6 offset:192
ds_read_u16 v12, v6 offset:256
ds_read_u16 v13, v6 offset:320
ds_read_u16 v14, v6 offset:384
ds_read_u16 v15, v6 offset:448
```

**Key observation:** All 8 reads use the **SAME base register (v6)** with different offsets.
- Offsets: 0, 64, 128, 192, 256, 320, 384, 448 bytes
- These are row offsets (64 bytes = 32 elements × 2 bytes = one row)
- **Each thread reads 8 rows of the SAME column**

### WITH XOR (`04_row_major_xor-hip-amdgcn-amd-amdhsa-gfx942.s`)
```asm
ds_read_u16 v14, v23
ds_read_u16 v15, v24
ds_read_u16 v16, v34
ds_read_u16 v17, v25
ds_read_u16 v18, v28 offset:128
ds_read_u16 v19, v26
ds_read_u16 v20, v27 offset:128
ds_read_u16 v21, v22 offset:256
```

**Key observation:** Each read uses a **DIFFERENT base register** (v23, v24, v34, v25, v28, v26, v27, v22).
- XOR transform scrambles addresses
- Addresses are distributed across different banks

---

## Two Types of Conflicts

### 1. INTRA-LANE Conflicts
One thread's 8 sequential reads hitting the same bank multiple times.

**WITHOUT XOR** - One thread reading column k=0 at rows 0-7:
```
Row 0: byte 0   → slot 0   → bank 0
Row 1: byte 64  → slot 16  → bank 16
Row 2: byte 128 → slot 32  → bank 0
Row 3: byte 192 → slot 48  → bank 16
Row 4: byte 256 → slot 64  → bank 0
Row 5: byte 320 → slot 80  → bank 16
Row 6: byte 384 → slot 96  → bank 0
Row 7: byte 448 → slot 112 → bank 16
```
Banks hit: {0, 16, 0, 16, 0, 16, 0, 16}
- Bank 0: 4 hits
- Bank 16: 4 hits

**Question:** Do these sequential reads cause conflicts? Or are conflicts only counted for simultaneous accesses across threads?

### 2. INTER-LANE Conflicts
Multiple threads hitting the same bank at the same time during a single ds_read instruction.

When all 64 threads execute `ds_read_u16 v8, v6` simultaneously:
- Each thread has a different value in v6 (their column's base address)
- They all access LDS at the same time
- Conflicts occur if multiple threads hit the same bank

---

## Open Questions

### Q1: What Does the Profiler Actually Count?
- Only inter-lane conflicts (multiple threads, same instruction)?
- Only intra-lane conflicts (one thread, sequential reads)?
- Both?

### Q2: Phase Groupings
The tile distribution creates phase groupings like:
```
Phase 0: {0, 1, 2, 3, 20, 21, 22, 23}
```

But lanes 0,1,2,3 have K2=0,1,2,3 meaning they read **different columns** (k=0,1,2,3), not the same column.

If 8 lanes in a phase read different columns at the same row, they hit different banks (FP16 adjacent k values share slots). This would give **0 inter-lane conflicts**.

But the profiler shows 7,168 conflicts. Where do they come from?

### Q3: Between-Wavefront Execution
- Do all 4 wavefronts execute simultaneously?
- We verified each wavefront uses exclusive banks (no inter-WF conflicts)
- WF0: banks {0,1,2,3}, WF1: {4,5,6,7}, WF2: {8,9,10,11}, WF3: {12,13,14,15}

---

## Relevant Files

### Test Programs
| File | Description |
|------|-------------|
| `debug_fp16_conflicts.cpp` | Main conflict calculator (needs fixing) |
| `test_inter_wf_conflicts.cpp` | Shows wavefronts use exclusive banks |
| `test_slot_based_conflicts.cpp` | Slot-based model (matches XOR: 3,072) |
| `test_wavefront_model.cpp` | Thread-count model (close to no-XOR: 7,680) |
| `test_fp16_same_slot.cpp` | FP16 same-slot optimization analysis |

### Documentation
| File | Description |
|------|-------------|
| `BANK_CONFLICT_CALCULATION_FINAL.md` | Previous "final" understanding |
| `CONFLICT_MODEL_STATUS.md` | This file - current status |

### Assembly Files (in /data0/aghamari/composable_kernel/)
| File | Description |
|------|-------------|
| `01_row_major-hip-amdgcn-amd-amdhsa-gfx942.s` | WITHOUT XOR assembly |
| `04_row_major_xor-hip-amdgcn-amd-amdhsa-gfx942.s` | WITH XOR assembly |

---

## Models Tried

### Model 1: Per-Column, Per-DM (BANK_CONFLICT_CALCULATION_FINAL.md)
```
32 columns × 8 dm steps × 7 conflicts × 4 blocks = 7,168 ✓
```
- Assumes 8 threads read same column simultaneously
- Each hits same bank with different slots → 7 conflicts
- **Matches profiler for WITHOUT XOR**

### Model 2: Slot-Based (test_slot_based_conflicts.cpp)
```
Conflicts = (unique_slots_per_bank - 1)
```
- FP16 adjacent k values share slot
- **Matches profiler for WITH XOR (3,072)**
- Doesn't match WITHOUT XOR (3,584 vs 7,168)

### Model 3: Phase + Inter-WF (debug_fp16_conflicts.cpp latest)
```
8 phases × 8 M1 steps × 4 wavefronts
```
- Uses actual phase groupings
- Shows 0 conflicts because phase lanes read different columns
- **Doesn't match either profiler value**

---

## The Core Confusion

The assembly shows each thread reads 8 rows of one column (same v6 base, different offsets).

But the tile distribution's phase grouping puts lanes with different K2 values together, meaning they read different columns.

**Resolution needed:**
1. Are the phases correct for this distribution?
2. Or is the "8 threads read same column" model from the assembly correct?
3. How do we reconcile the two?

---

## Next Steps

1. Verify the actual phase groupings for this specific tile distribution
2. Determine if profiler counts intra-lane, inter-lane, or both
3. Build a model that accounts for both wavefront execution and the actual memory access pattern from assembly
