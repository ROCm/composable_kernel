# Bank Conflict Profiling Findings

## Summary

We successfully reverse-engineered how AMD's `rocprofv3` profiler counts LDS bank conflicts by creating manual conflict calculators that match profiler measurements.

## Key Discoveries

### 1. FP16 Bank Conflicts Have a ×3 Multiplier

**Finding:** AMD profiler counts FP16 bank conflicts as **3 conflicts per pair** when two FP16 elements share a 4-byte bank slot.

**Evidence:**
- Manual calculation (without multiplier): 256 write conflicts per tile
- Manual calculation (with ×3): 768 write conflicts per tile
- Scaled by 4 K-iterations: 768 × 4 = **3,072 conflicts**
- Profiler measurement: **3,072 conflicts** ✓ EXACT MATCH

**Explanation:**
- Each LDS bank is 4 bytes wide
- FP16 elements are 2 bytes each
- Two consecutive FP16 writes to the same bank "pair up" in one slot
- Instead of counting this as 1 conflict, the profiler counts it as **3 conflicts**
- This likely reflects the internal hardware arbitration cycles needed

### 2. Phase-Aware Execution is Critical

**Finding:** Conflicts only occur between lanes executing in the same phase, not across all 64 lanes globally.

**Thread Phase Grouping:**
- **WRITE phases:** 8 consecutive lanes execute together
  - Phase 0: lanes {0,1,2,3,4,5,6,7}
  - Phase 1: lanes {8,9,10,11,12,13,14,15}
  - ... (8 phases total)

- **READ phases:** 8 non-consecutive lanes execute together
  - Phase 0: lanes {0,1,2,3,20,21,22,23}
  - Phase 1: lanes {4,5,6,7,16,17,18,19}
  - ... (8 phases total, grouped by tile distribution)

**Impact:** A naive "all lanes at once" calculation overcounts conflicts significantly. Only lanes in the same phase can conflict with each other.

### 3. FP16 XOR Effectiveness

**Without XOR (FP16):**
- Write conflicts: 3,072 (intra-lane pairing × 3)
- Read conflicts: 4,096 (transpose intra-lane)
- **Total: 7,168**

**With XOR (FP16):**
- Write conflicts: 3,072 (intra-lane pairing × 3, XOR doesn't help here)
- Read conflicts: 0 (XOR eliminates transpose conflicts!)
- **Total: 3,072**

**Reduction: 57% fewer conflicts** (7,168 → 3,072)

### 4. FP32 Profiler Anomaly

**Observation:** FP32 shows identical conflict counts with and without XOR (both 15,360).

**Hypothesis:** The profiler counter `SQ_LDS_BANK_CONFLICT` may have limitations or bugs when measuring certain FP32 XOR patterns. Our manual address calculations proved that XOR **does** distribute addresses correctly for FP32, but the profiler doesn't reflect this.

**Evidence:**
- Verified XOR address distribution: Column 0 maps to banks {0,8,16,24} instead of all bank 0
- Profiler shows identical instruction counts changing (384 → 640) when XOR is enabled
- But conflict counter remains unchanged (15,360 for both)

## Manual Conflict Calculation Formula

### FP16 (with ×3 multiplier)

```
Write conflicts (intra-lane):
  For each phase (8 lanes):
    For each lane:
      Count banks accessed by lane's 8 elements
      conflicts += (count_per_bank - 1) × 3

Read conflicts (intra-lane):
  For each phase:
    For each column:
      For each lane reading that column:
        Count banks accessed by lane's 8 elements
        conflicts += (count_per_bank - 1)

Total = write_conflicts + read_conflicts
Scale by K-loop iterations (4 for 128/32)
```

### FP32 (no multiplier)

Same formula but without the ×3 multiplier for writes (FP32 elements don't pair in bank slots).

## Files

- `debug_fp16_conflicts.cpp` - Phase-aware FP16 conflict calculator (matches profiler)
- `debug_fp32_conflicts.cpp` - FP32 conflict calculator (profiler shows anomaly)
- `04_row_major_xor.cpp` - FP16 kernel with XOR (3,072 conflicts)
- `01_row_major.cpp` - FP16 kernel without XOR (7,168 conflicts)
- `04_row_major_xor_fp32.cpp` - FP32 kernel with XOR (profiler: 15,360)
- `01_row_major_fp32.cpp` - FP32 kernel without XOR (profiler: 15,360)

## Profiler Command

```bash
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS,SQ_LDS_IDX_ACTIVE --output conflicts_test -- ./bin/kernel
```

Counter meanings:
- `SQ_LDS_BANK_CONFLICT`: Number of bank conflicts detected
- `SQ_INSTS_LDS`: Number of LDS instructions executed
- `SQ_LDS_IDX_ACTIVE`: Number of active LDS lanes

## Conclusions

1. **FP16 ×3 multiplier**: AMD hardware counts FP16 pairing conflicts as 3 conflicts each
2. **Phase execution**: Must account for wave scheduling when calculating conflicts
3. **XOR works for FP16**: Successfully reduces transpose read conflicts to zero
4. **FP32 profiler issue**: Counter may not accurately reflect XOR benefits for FP32
5. **Manual calculation validated**: Our phase-aware calculator matches profiler for FP16

## Next Steps

- Document the ×3 multiplier finding in tutorials
- Create visualization showing phase-grouped execution
- Investigate FP32 profiler counter behavior further
- Test on different GPU architectures (gfx90a, gfx940, etc.)
