# Intra-Lane Conflict Test Results

## SHOCKING DISCOVERY: ALL Tests Show 0 Conflicts!

| Test # | Description | Pattern | SQ_LDS_BANK_CONFLICT |
|--------|-------------|---------|----------------------|
| **1** | ONE thread, transpose (k=0) | Banks {0,16,0,16,0,16,0,16}, Different slots | **0** |
| **2** | ONE thread, different banks | Banks {0,1,2,3,4,5,6,7} | **0** |
| **3** | ONE thread, bank 0 only, 8 different slots | All bank 0, slots {0,32,64,96,128,160,192,224} | **0** |
| **4** | Full Phase 0 (8 lanes) | Each lane reads transpose pattern | **0** |
| **5** | Scaled (32 threads) | 32 lanes each with transpose pattern | **0** |

## What This Means

**Even INTRA-lane conflicts show 0!**

Test 3 was designed to create maximum intra-lane conflicts:
- ONE thread
- Accesses bank 0 EIGHT times
- Each access is a DIFFERENT slot (0, 32, 64, 96, 128, 160, 192, 224)
- Expected: Must serialize → ~7-8 conflicts

**Result: 0 conflicts!**

## Implications

1. **Our simple isolated tests CANNOT recreate the 7,168/3,072 conflicts**
2. **Something fundamentally different happens in the real kernel**
3. **Possible explanations:**
   - Vector loads (ds_read_b128) behave differently than individual reads
   - The real load_tile uses different LDS access patterns
   - Multiple LDS operations per logical read
   - Compiler transformations we don't see
   - Our test kernels are too simple and get optimized differently

## The Mystery Deepens

**What we've proven:**
- ✓ Inter-lane same-slot: 0 conflicts (FP16 optimization)
- ✓ All dm steps: 0 conflicts
- ✓ Intra-lane different-slots: 0 conflicts (!!)

**What we CANNOT explain:**
- ❌ Why the real kernel shows 7,168 and 3,072 conflicts
- ❌ Where those conflicts come from
- ❌ Why our isolated tests always show 0

## Hypothesis: Vector Load Behavior

**Our tests use:**
```cpp
for (int m = 0; m < 8; m++) {
    _Float16 val = lds[m * 32 + k];  // Individual scalar loads
}
```

**Real kernel might use:**
```
ds_read_b128  // 128-bit vector load (8 FP16 elements at once)
```

**Possibility:** Vector loads might:
- Access LDS differently than scalar loads
- Create different conflict patterns
- Have internal serialization we don't model

## Next Steps

1. **Profile the ACTUAL pure_read_no_xor/pure_read_xor kernels** to verify they still show 7,168/3,072
2. **Disassemble the real kernel** to see actual LDS instructions
3. **Test with explicit vector loads** (ds_read_b128 inline assembly)
4. **Compare compiler output** between our simple tests and real kernel

## Conclusion

Our isolated tests are **too simple** to recreate the real conflict pattern. The conflicts must come from:
- How CK's load_tile actually accesses LDS
- Vector load behavior
- Multiple passes or hidden accesses
- Compiler-generated LDS operations

**The calculator approach (analyzing individual offsets) may not capture the full picture.**
