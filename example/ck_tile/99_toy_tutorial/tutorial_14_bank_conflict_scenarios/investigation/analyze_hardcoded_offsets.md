# Understanding the Hardcoded Offsets (+128, +128, +256)

## The Question

Why do 3 out of 8 `ds_read_u16` instructions have hardcoded offsets?

```assembly
ds_read_u16 v14, v28                  ← No offset
ds_read_u16 v15, v27                  ← No offset
ds_read_u16 v16, v24                  ← No offset
ds_read_u16 v17, v25                  ← No offset
ds_read_u16 v18, v29 offset:128       ← +128 bytes!
ds_read_u16 v19, v23                  ← No offset
ds_read_u16 v20, v26 offset:128       ← +128 bytes!
ds_read_u16 v21, v22 offset:256       ← +256 bytes!
```

## The Tile Distribution

For the transpose read (LDS → Registers), the distribution is:

```cpp
K0=1, K1=4, K2=8  (K dimension: 1 × 4 × 8 = 32)
M0=8, M1=8        (M dimension: 8 × 8 = 64)
```

LDS layout: `[M, K] = [64, 32]` in FP16 (2 bytes each)
- Row stride: 32 × 2 = 64 bytes

## What Each Thread Reads

Each thread reads **8 FP16 values** from a single K-column across different M-rows.

The 8 reads access different M-positions. Let me trace through what the compiler might be doing:

### Hypothesis: Loop Unrolling with Stride Optimization

If each thread reads from positions like:
- m = [0, 8, 16, 24, 32, 40, 48, 56] (every 8th row, for example)

Then in bytes:
- Row 0:  offset = 0 × 64 = 0
- Row 8:  offset = 8 × 64 = 512
- Row 16: offset = 16 × 64 = 1024
- etc.

But wait, that doesn't match +128/+256...

### Better Hypothesis: Vectorized Access Pattern

The +128 and +256 are:
- +128 bytes = 64 FP16 elements = **2 rows** worth of data
- +256 bytes = 128 FP16 elements = **4 rows** worth of data

So the compiler might be:
1. Calculating a base address with XOR (stored in v22-v29)
2. Reading from base, base+2rows, base+4rows

Example for a thread reading column k=0:
```
Read 1: v28 = XOR(m=0, k=0) = base address
Read 2: v27 = XOR(m=1, k=0)
Read 3: v24 = XOR(m=2, k=0)
Read 4: v25 = XOR(m=3, k=0)
Read 5: v29 + 128 = XOR(m=4, k=0) + 128  ← Offset instead of recalculating!
Read 6: v23 = XOR(m=5, k=0)
Read 7: v26 + 128 = XOR(m=6, k=0) + 128  ← Offset instead of recalculating!
Read 8: v22 + 256 = XOR(m=7, k=0) + 256  ← Offset instead of recalculating!
```

### Why Would Compiler Do This?

**Possible reasons:**

1. **Register pressure**: The XOR transformation requires calculations. Instead of computing 8 different XOR addresses, compute 5 and add offsets to 3.

2. **Instruction encoding**: `ds_read_u16 vX, vY offset:N` is a single instruction. The alternative would be:
   ```
   v_add_u32 vTemp, v29, 128
   ds_read_u16 v18, vTemp
   ```
   That's 2 instructions vs 1!

3. **Compiler optimization**: The compiler sees that some addresses are related by constant offsets and optimizes.

## The Problem

The XOR descriptor is applied to the **base address calculation** only:
```
v29 = XOR_transform(address)
```

When you add +128:
```
final = v29 + 128
```

The +128 is added AFTER XOR, so it doesn't benefit from the XOR transformation's bank spreading!

### Concrete Example

Assume XOR spreads addresses like this:
```
Thread 0: v29 = XOR(row 4) = 0x100 → bank 64 / 4 = bank 16
Thread 1: v29 = XOR(row 4) = 0x140 → bank 80 / 4 = bank 20
Thread 2: v29 = XOR(row 4) = 0x180 → bank 96 / 4 = bank 24
```

But with +128 added:
```
Thread 0: v29 + 128 = 0x180 → bank 96 / 4 = bank 24  ← Conflicts with Thread 2!
Thread 1: v29 + 128 = 0x1C0 → bank 112 / 4 = bank 28
Thread 2: v29 + 128 = 0x200 → bank 128 / 4 = bank 0
```

The +128 offset can cause threads that were spread apart by XOR to collide on the same bank!

## Where in C++ Code?

The C++ code doesn't explicitly request these offsets. They come from:

**File**: `tile_window.hpp`, line 259-261
```cpp
const vector_t vec_value =
    this->get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
        bottom_tensor_thread_coord, 0, bool_constant<oob_conditional_check>{});
```

This eventually calls `buffer_view::get()` which does:
```cpp
auto rtn = *c_style_pointer_cast<const buf_t*>(&p_data_[i + linear_offset]);
```

The `i + linear_offset` is calculated by the tile distribution. For 8 reads per thread, the compiler:
1. Unrolls the loop 8 times
2. Calculates some addresses with XOR
3. **Optimizes** others by adding constant offsets

The offsets are an **optimization** by the compiler, not explicit in the C++ code!

## How to Prove This?

We could:
1. Disable compiler optimizations (-O0) - would likely see all 8 with XOR
2. Use different tile distributions - might change which reads get offsets
3. Manually inspect LLVM IR - would show where offsets are introduced

## The Fix?

To avoid this, the compiler would need to:
```assembly
# Instead of:
ds_read_u16 v18, v29 offset:128

# Do:
v_add_u32 v_temp, v29, 128
ds_read_u16 v18, v_temp
```

But that's 2 instructions instead of 1, so the compiler "optimizes" to use the offset field.

The real fix would be if the XOR descriptor could be applied to the offset as well, but that's not how the hardware works.
