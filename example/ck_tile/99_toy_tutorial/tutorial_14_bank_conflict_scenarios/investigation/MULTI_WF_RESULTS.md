# Multi-Wavefront Bank Conflict Test Results

## Summary

This document presents the results from `test_inter_lane_fp16.cpp` with extended multi-wavefront tests to understand inter-WF bank conflict behavior.

## Test Results

| Test | Description | Pattern | Conflicts | Finding |
|------|-------------|---------|-----------|---------|
| **Test 1** | Inter-lane same-slot | 8 threads, pairs hit same slot (intra-lane) | **0** | ✅ FP16 optimization works intra-lane |
| **Test 2** | Inter-lane different-slots | 8 threads, all hit bank 0 (different slots) | **7** | ⚠️ Baseline conflicts (single WF) |
| **Test 3** | Exact transpose pattern | Phase 0 lanes, pairs hit same slot | **0** | ✅ FP16 optimization works for actual pattern |
| **Test 4** | No conflicts baseline | All different banks | **0** | ✅ Baseline validation |
| **Test 5** | Two WFs, same banks | 16 threads (2 WFs × 8), all hit bank 0 | **14** | 🔥 **2× conflicts vs Test 2!** |
| **Test 6** | Four WFs, same banks | 32 threads (4 WFs × 8), all hit bank 0 | **28** | 🔥 **4× conflicts vs Test 2!** |
| **Test 7** | Inter-WF same slots | 4 threads (2 WFs), same slots | **0** | ✅ FP16 optimization works inter-WF |
| **Test 8** | Actual distribution pattern | 4 WFs with K1 distribution | **0** | ✅ Each WF uses exclusive banks |

## Critical Findings

### 1. **Inter-Wavefront Conflicts ARE Cumulative**

The most important discovery:

- **Test 2 (1 WF):** 7 conflicts
- **Test 5 (2 WFs):** 14 conflicts = **2 × Test 2**
- **Test 6 (4 WFs):** 28 conflicts = **4 × Test 2**

**Conclusion:** When multiple wavefronts access the same banks:
- Conflicts scale **linearly** with the number of wavefronts
- This means wavefronts do NOT execute completely independently when accessing LDS
- Bank conflicts from different wavefronts **add together**

### 2. **FP16 Same-Slot Optimization Works Across Wavefronts**

- **Test 1 (intra-lane):** 0 conflicts ✅
- **Test 3 (actual pattern):** 0 conflicts ✅
- **Test 7 (inter-WF):** 0 conflicts ✅

**Conclusion:** The FP16 optimization (2 FP16 elements in same slot, no conflict) works:
- Within a lane (expected)
- Across lanes in same wavefront (expected)
- **Across different wavefronts** (NEW finding!)

This confirms that when 2 threads from **different wavefronts** access the same slot but different FP16 elements, there's no conflict.

### 3. **K1 Distribution Ensures Wavefront Isolation**

- **Test 8:** 0 conflicts with 4 wavefronts

**Conclusion:** The K1 distribution strategy is correct:
- WF0: k=0-7 → banks 0-3
- WF1: k=8-15 → banks 4-7
- WF2: k=16-23 → banks 8-11
- WF3: k=24-31 → banks 12-15

Each wavefront uses **exclusive banks**, so no inter-WF conflicts occur in the actual distribution pattern.

## Implications for Bank Conflict Analysis

### Why XOR Still Shows Conflicts in Production Code

From our previous tests, we saw **some** conflicts even with XOR applied. The multi-WF results suggest:

1. **Within each wavefront:** XOR eliminates intra-WF conflicts (as proven by Test 1, 3, 7)
2. **Across wavefronts:** If any two wavefronts accidentally access the **same banks after XOR**, conflicts will accumulate

### Conflict Calculation Model Update

**Previous assumption (WRONG):**
- Only count conflicts within a single wavefront
- Different wavefronts don't interfere

**Corrected model:**
- Count conflicts within each wavefront
- **ADD** conflicts from different wavefronts accessing the same banks
- Total conflicts = Σ(conflicts per WF) when banks overlap

### Example Scenario

If at some step:
- WF0 has 3 threads hitting bank 5
- WF1 has 2 threads hitting bank 5
- Total conflicts = (3-1) + (2-1) + **cross-WF conflicts**

Based on Test 5/6 results, the total would be:
- **Total conflicts = 4** (5 threads - 1, all serialized together)

This matches the linear scaling we observed:
- 8 threads → 7 conflicts
- 16 threads → 14 conflicts
- 32 threads → 28 conflicts

## Action Items

### ✅ Confirmed

1. FP16 same-slot optimization works across all boundaries (intra-lane, inter-lane, inter-WF)
2. K1 distribution strategy correctly separates wavefronts into exclusive banks
3. Inter-wavefront conflicts **are cumulative** and add linearly

### ⚠️ Next Steps

1. **Re-analyze production XOR conflicts** with multi-WF model:
   - Identify if any XOR'd addresses from different WFs collide on same banks
   - Check if phase grouping causes temporary bank overlaps across WFs

2. **Update conflict calculator** to account for inter-WF conflicts:
   - Track banks accessed by each wavefront separately
   - Calculate cross-WF conflicts when banks overlap
   - Sum total conflicts across all WFs

3. **Verify XOR transpose implementation:**
   - Ensure XOR parameters prevent inter-WF bank collisions
   - Confirm phase grouping maintains WF isolation

## Conclusion

**The multi-wavefront tests reveal a critical aspect of bank conflict behavior:**

- Conflicts are NOT isolated per-wavefront
- When multiple WFs access the same banks, conflicts accumulate linearly
- This explains why we see residual conflicts even with XOR applied
- The solution is to ensure **wavefront-level bank isolation**, which K1 distribution achieves

**Key takeaway:** XOR eliminates conflicts within a wavefront, but if the distribution causes multiple wavefronts to hit the same banks, conflicts will still occur and add up.
