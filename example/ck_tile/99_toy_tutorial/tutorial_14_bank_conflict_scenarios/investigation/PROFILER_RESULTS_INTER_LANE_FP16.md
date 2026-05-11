# Profiler Results: Inter-Lane FP16 Bank Conflict Tests

## Summary

**VERIFIED:** FP16 optimization works for INTER-lane conflicts when threads access **different FP16 elements in the same 4-byte slot**!

## Test Results

| Test | Pattern | SQ_LDS_BANK_CONFLICT | Interpretation |
|------|---------|---------------------|----------------|
| **Test 1** | Inter-lane same-slot<br/>(2 threads, same slot, different FP16) | **0** | ✓ NO conflicts!<br/>FP16 optimization WORKS for inter-lane! |
| **Test 2** | Inter-lane different-slots<br/>(4 threads, same bank, different slots) | **7** | ✓ HIGH conflicts<br/>(expected ~4-way serialization) |
| **Test 3** | EXACT transpose pattern<br/>(Phase 0, dm=0) | **0** | ✓ NO conflicts!<br/>Our actual pattern benefits from FP16! |
| **Test 4** | No conflicts baseline<br/>(all different banks) | **0** | ✓ Baseline verified |

## What This Means

### Test 1: Inter-Lane Same-Slot (CRITICAL FINDING!)
```
Thread 0: reads m=0, k=0 -> offset 0 -> slot 0, bank 0 (first FP16)
Thread 1: reads m=0, k=1 -> offset 1 -> slot 0, bank 0 (second FP16)
Both threads execute SIMULTANEOUSLY in same wavefront
```

**Result: 0 conflicts**

**Interpretation:** The AMD GPU hardware CAN service 2 different threads reading 2 different FP16 elements from the SAME 4-byte slot in a single cycle!

This is NOT just an intra-lane optimization - it works for **inter-lane** (cross-thread) as well!

### Test 2: Inter-Lane Different-Slots (Validation)
```
Thread 0: m=0, offset 0   -> slot 0, bank 0
Thread 1: m=2, offset 64  -> slot 32, bank 0 (different slot!)
Thread 2: m=4, offset 128 -> slot 64, bank 0 (different slot!)
Thread 3: m=6, offset 192 -> slot 96, bank 0 (different slot!)
```

**Result: 7 conflicts** (close to expected 4-way serialization)

**Interpretation:** When threads hit the same bank but DIFFERENT slots, true bank conflicts occur. The FP16 optimization does NOT help here.

### Test 3: EXACT Transpose Pattern (CRITICAL!)
```
Phase 0 lanes: {0, 1, 2, 3, 20, 21, 22, 23}
All read m=0, k={0,1,2,3,4,5,6,7}

Pairs that hit same slot:
  Lanes 0,1:   both hit bank 0, slot 0 (different FP16 in slot)
  Lanes 2,3:   both hit bank 1, slot 1 (different FP16 in slot)
  Lanes 20,21: both hit bank 2, slot 2 (different FP16 in slot)
  Lanes 22,23: both hit bank 3, slot 3 (different FP16 in slot)
```

**Result: 0 conflicts**

**Interpretation:** Our ACTUAL transpose pattern at dm=0 has ZERO conflicts because all lane pairs that hit the same bank are accessing different FP16 elements within the same slot, which the hardware services efficiently!

### Test 4: Baseline (Validation)
**Result: 0 conflicts**
All threads access different banks - no contention, as expected.

---

## Impact on Our Conflict Analysis

### PROBLEM: Our Calculations are WRONG!

We calculated:
```
Inter-lane conflicts per step dm=0:
  Bank 0: lanes {0,1} -> 2 lanes -> (2-1) = 1 conflict
  Bank 1: lanes {2,3} -> 2 lanes -> (2-1) = 1 conflict
  Bank 2: lanes {20,21} -> 2 lanes -> (2-1) = 1 conflict
  Bank 3: lanes {22,23} -> 2 lanes -> (2-1) = 1 conflict
  Total: 4 conflicts per step × 8 steps = 32 per k_base
```

**But Test 3 shows: 0 conflicts!**

This means lanes 0,1 hitting the same slot do NOT conflict because they're reading different FP16 elements (offset 0 and offset 1).

### The Corrected Understanding

**Inter-lane conflicts only occur when:**
1. Multiple lanes hit the same bank AND
2. They access DIFFERENT 4-byte slots

**Inter-lane conflicts DO NOT occur when:**
- Multiple lanes hit the same bank but access DIFFERENT FP16 elements in the SAME 4-byte slot
- Hardware can service 2 FP16 reads from one slot per cycle, even from different lanes!

---

## Why Did Our Profiler Match Then?

This is now a MYSTERY. We need to reconsider our entire conflict model because:

1. **Test 3 shows 0 conflicts** for the exact pattern at dm=0
2. **But profiler shows 3,072 and 7,168** for the full kernel
3. **Our calculations predicted exactly those numbers** based on (nlanes-1)

### Possible Explanations:

**A. Not all steps are same-slot pairs:**
- We only looked at dm=0
- At other dm values (dm=1,2,3...), lanes might hit different slots in the same bank
- Need to analyze all 8 steps (dm=0 through dm=7)

**B. The real conflicts come from elsewhere:**
- Maybe intra-lane conflicts are more complex than we thought
- Maybe the LDS is accessed multiple times per logical read
- Need to trace the actual LDS access pattern more carefully

**C. Our phase grouping is wrong:**
- Maybe lanes don't execute exactly as we think
- The distribution might create different bank patterns

---

## Next Steps

1. **Analyze all 8 steps (dm=0-7)** for Phase 0, k_base=0:
   - Check if lanes still pair up on same slots at dm=1,2,3...
   - Or if they start hitting different slots in the same bank

2. **Test the actual load_tile pattern:**
   - Our simple test might not match the real tile_window behavior
   - Create test that uses actual CK load_tile

3. **Profile with more detail:**
   - Add prints showing which offsets each lane accesses
   - Verify slot calculations for all 8 steps

4. **Recalculate based on new understanding:**
   - FP16 same-slot optimization works inter-lane
   - Only count conflicts when lanes hit same bank, different slots

---

## The Hardware Capability (Confirmed)

**AMD gfx942 LDS can:**
- Service 2 FP16 reads from the same 4-byte bank slot in one cycle
- This works even when the reads come from different lanes/threads
- This is a true hardware optimization, not just compiler/scheduling

**AMD gfx942 LDS cannot:**
- Service 2 reads from the same bank if they're to different 4-byte slots
- These must be serialized (bank conflict)

---

## Detailed Results

### Test 1: inter_lane_same_slot
```
Kernel: _Z20inter_lane_same_slotPKDF16_Pf
SQ_LDS_BANK_CONFLICT: 0
```

### Test 2: inter_lane_different_slots
```
Kernel: _Z26inter_lane_different_slotsPKDF16_Pf
SQ_LDS_BANK_CONFLICT: 7
```

### Test 3: exact_transpose_pattern
```
Kernel: _Z23exact_transpose_patternPKDF16_Pf
SQ_LDS_BANK_CONFLICT: 0
```

### Test 4: no_conflicts_baseline
```
Kernel: _Z21no_conflicts_baselinePKDF16_Pf
SQ_LDS_BANK_CONFLICT: 0
```

---

## Conclusion

The profiler results **definitively prove** that:

1. ✓ FP16 optimization works for **inter-lane** conflicts (not just intra-lane)
2. ✓ When 2 threads access different FP16 elements in the same slot: **0 conflicts**
3. ✓ When 2+ threads access the same bank but different slots: **HIGH conflicts**
4. ✓ Our exact transpose pattern at dm=0: **0 conflicts**

**However, this creates a NEW mystery:** Why do we see 3,072 and 7,168 conflicts in the full kernel if the basic pattern shows 0 conflicts?

The answer must lie in:
- Other dm steps having different slot patterns
- Multiple accesses per logical element
- Or our understanding of the phase/distribution being incomplete

**Further investigation required!**
