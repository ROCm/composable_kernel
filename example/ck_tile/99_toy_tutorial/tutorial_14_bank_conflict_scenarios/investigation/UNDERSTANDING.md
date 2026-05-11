# Final Understanding: How Bank Conflicts Really Work

## The Breakthrough

Bank conflicts occur when **multiple threads execute the SAME load instruction simultaneously** and access the **same bank** (with different slots).

## What We Proved

### Test That Showed Conflicts (7 conflicts):
```cpp
// 8 threads execute THIS SINGLE INSTRUCTION together:
if (tid < 8) {
    _Float16 val = lds[tid * 64];  // All 8 threads hit bank 0, different slots
}

Thread 0: offset 0   → bank 0, slot 0
Thread 1: offset 64  → bank 0, slot 32  ← DIFFERENT slot, SAME bank!
Thread 2: offset 128 → bank 0, slot 64
...
Result: 8 threads → (8-1) = 7 conflicts ✓
```

### Tests That Showed 0 Conflicts:
```cpp
// Each thread loops sequentially - NO simultaneous contention
for (int m = 0; m < 8; m++) {
    _Float16 val = lds[m * 32 + k];  // Sequential per thread
}
```

## The Real CK Pattern

**WITHOUT XOR:** ~8 threads/bank → 7-way conflict
- 7,168 total / 4 blocks = 1,792 per tile
- If each tile has ~256 simultaneous accesses: 1,792 / 256 = 7 conflicts/access ✓

**WITH XOR:** ~4 threads/bank → 3-way conflict
- 3,072 total / 4 blocks = 768 per tile
- 768 / 256 = 3 conflicts/access ✓

**Ratio:** 7,168 / 3,072 = 2.33 = 7/3 ✓

## What XOR Does

XOR swizzling **reduces thread grouping per bank**:
- WITHOUT XOR: 8 threads map to same bank
- WITH XOR: 4 threads map to same bank
- **Reduction: 7-way → 3-way (57% fewer conflicts)**

## Calculator Fix Needed

The calculator must:

1. **Identify simultaneous execution groups**
   - All threads in a phase execute together
   - Each "step" of the tile_window movement

2. **Count threads per bank per execution**
   - For each instruction, how many threads hit each bank?
   - NOT sequential accesses by one thread

3. **Apply conflict formula**
   - Same bank, different slots: (n_threads - 1) conflicts
   - Same bank, same slot: 0 conflicts (FP16 optimization)

4. **Consider tile_distribution**
   - The distribution maps threads to coordinates
   - Multiple threads → same bank during simultaneous read

## Why Our Simple Tests Failed

Our tests had threads looping sequentially:
```cpp
for (int m = 0; m < 8; m++) {
    val = lds[m * 32 + k];  // Thread 0 executes all 8 sequentially
}
```

Real CK unrolls or uses tile operations where:
```cpp
// Conceptually (not actual CK code):
load_tile() {
    // All threads execute TOGETHER:
    step0: all_threads_read_their_first_element();   // Simultaneous!
    step1: all_threads_read_their_second_element();  // Simultaneous!
    ...
}
```

## The Fix

We need to analyze at the **thread-level simultaneity**:
- How many threads access each bank at the same time?
- This depends on tile_distribution encoding
- NOT just individual offset calculations
