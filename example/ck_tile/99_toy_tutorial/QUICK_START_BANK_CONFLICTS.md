# Quick Start: Bank Conflict Analysis

This is a quick reference for profiling and analyzing LDS bank conflicts in Tutorial 11.

For comprehensive understanding, see [BANK_CONFLICT_TUTORIAL.md](BANK_CONFLICT_TUTORIAL.md).

## What Are Bank Conflicts?

**Simple explanation:** When multiple GPU threads try to access the same memory bank simultaneously, they must wait in line (serialize), reducing performance.

**Our case:** Matrix transpose reads columns from row-major data, causing severe bank conflicts.

**Solution:** XOR swizzling permutes physical addresses to spread accesses across all 32 banks.

## Quick Profile

**1. Build tutorials:**
```bash
cd relbuild
cmake --build . --target aa_tutorial_11_plain_transpose aa_tutorial_11_production_transpose -j$(nproc)
```

**2. Run automated profiling:**
```bash
bash ../example/ck_tile/99_toy_tutorial/scripts/profile_bank_conflicts.sh
```

This will:
- Build both versions (plain and XOR)
- Profile using AMD performance counters
- Generate comparison report
- Show 57% conflict reduction

## Expected Results

```
╔════════════════════════════════════════════════════════════╗
║          Bank Conflict Analysis Results                   ║
╚════════════════════════════════════════════════════════════╝

Metric                          Plain LDS           XOR LDS
──────────────────────────────────────────────────────────────
SQ_LDS_BANK_CONFLICT                7,168             3,072
SQ_INSTS_LDS                          608               608
Conflict Rate (%)                 1,244.0             533.0
Conflicts per Instruction           12.4               5.3
──────────────────────────────────────────────────────────────
Conflict Reduction                 4,096 (57.1%)
Rate Improvement                   711.0%

✓ XOR swizzling reduces conflicts by 57%
✓ Plain: ~12-way conflicts → XOR: ~5-way conflicts
✓ Theoretical minimum: ~2-way (64 threads / 32 banks)
✓ Gap to optimal: 2.5× away from theoretical best
```

## Understanding the Numbers

### Conflict Rate >100%?

Yes! This means multiple conflicts per LDS instruction.

```
Plain: 1,244% = 12.4 conflicts per instruction
→ Each LDS access serializes ~12 times
→ 12× slower than ideal!

XOR: 533% = 5.3 conflicts per instruction
→ Each LDS access serializes ~5 times
→ Much better, but still room for improvement
```

### Why Not Zero Conflicts?

**Pigeonhole principle:** 64 threads, 32 banks → minimum 2 threads per bank

```
Theoretical optimal: 2-way conflicts (100% rate)
Current XOR:         5-way conflicts (533% rate)
Gap:                 2.5× from optimal

But XOR is practical:
- Simple implementation
- No algorithm changes
- 57% improvement
- Good enough for production!
```

## Manual Profiling

If you want to profile manually:

**Profile plain transpose:**
```bash
cd relbuild
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/plain \
          -- ./bin/aa_tutorial_11_plain_transpose
```

**Profile XOR transpose:**
```bash
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS \
          -d /tmp/xor \
          -- ./bin/aa_tutorial_11_production_transpose
```

**Analyze results:**
```bash
python3 ../example/ck_tile/99_toy_tutorial/scripts/analyze_bank_conflicts.py \
        /tmp/plain /tmp/xor
```

## Key Takeaways

1. **Bank conflicts are serious:** Plain transpose has 12× slowdown from conflicts
2. **XOR helps significantly:** 57% reduction with simple implementation
3. **Perfect is impossible:** Mathematical limits prevent zero conflicts with 64×32 tiles
4. **Practical solution:** XOR swizzling is the best trade-off for production code
5. **Know your limits:** Understanding constraints helps make informed decisions

## Next Steps

- **Read the full tutorial:** [BANK_CONFLICT_TUTORIAL.md](BANK_CONFLICT_TUTORIAL.md)
- **Study the code:** See detailed comments in `xor_test_production_transpose.cpp`
- **Experiment:** Try 32×32 tiles for near-zero conflicts (change `kM = 32` in code)
- **Apply to your kernels:** Use XOR swizzling in your own transpose operations

## Quick Reference: Profiling Counters

| Counter | Meaning |
|---------|---------|
| `SQ_LDS_BANK_CONFLICT` | Total number of bank conflicts |
| `SQ_INSTS_LDS` | Total number of LDS instructions |
| Conflict rate (%) | `(conflicts / instructions) × 100` |
| Conflicts per inst | `conflicts / instructions` |

**Ideal:** 0 conflicts per instruction (0%)
**Theoretical min (64t/32b):** 1 conflict per instruction (100%)
**Plain:** 12.4 conflicts per instruction (1,244%)
**XOR:** 5.3 conflicts per instruction (533%)

## Troubleshooting

**Error: "rocprofv3 not found"**
- Install ROCm profiling tools: `sudo apt install rocprofiler-dev`
- Or use module system: `module load rocm`

**Error: "results.db not found"**
- Check profiling completed successfully
- Look for error messages in rocprofv3 output
- Verify GPU is accessible: `rocm-smi`

**Kernel fails to run:**
- Check GPU targets match your hardware
- Verify HIP runtime: `hipcc --version`
- Check build logs for compilation errors

## Resources

- **Full tutorial:** [BANK_CONFLICT_TUTORIAL.md](BANK_CONFLICT_TUTORIAL.md)
- **Tutorial README:** [README.md](README.md)
- **AMD GPU architecture:** Search for "MI300 architecture guide"
- **ROCm profiling:** [ROCm documentation](https://rocm.docs.amd.com/)

---

**Questions?** Open an issue on the composable_kernel repository.
