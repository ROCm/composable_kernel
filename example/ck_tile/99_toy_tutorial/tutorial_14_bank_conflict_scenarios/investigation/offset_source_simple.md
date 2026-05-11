# Where Hardcoded Offsets Come From - Simple Answer

## The C++ Code

**File:** `04_row_major_xor.cpp`

**Line 331:**
```cpp
constexpr index_t M1 = 16 / sizeof(DataType); // = 8 for FP16
```

**Line 337-345:** (Distribution using M1)
```cpp
constexpr auto dist_km = make_static_tile_distribution(
    tile_distribution_encoding<
        sequence<1>,
        tuple<sequence<K0, K1, K2>, sequence<M0, M1>>,  // ← M1 = 8 here
        ...
    >{});
```

**Line 351:**
```cpp
auto reg_final = load_tile(lds_window_km);  // Uses the distribution
```

---

## What Happens

### Step 1: Distribution Determines Access Pattern

The `M1 = 8` parameter tells the distribution:
- Divide the M dimension into groups of 8
- Each thread reads 8 values

### Step 2: Memory Layout

LDS is `[K=32, M=64]` with FP16 (2 bytes):
- Each row: 64 elements × 2 bytes = 128 bytes
- **Key fact: 64 elements = 128 bytes**

### Step 3: Load Unrolls to 8 Reads

The `load_tile` call becomes 8 separate memory reads.

Due to the M1=8 distribution, some reads access memory locations that differ by:
- **64 elements = 128 bytes** (1 row)
- **128 elements = 256 bytes** (2 rows)

### Step 4: Compiler Optimizes

Instead of calculating all 8 addresses separately:
```
Address 1 = XOR(...)
Address 2 = XOR(...)
Address 3 = XOR(...)
Address 4 = XOR(...)
Address 5 = XOR(...)  ← Expensive!
Address 6 = XOR(...)
Address 7 = XOR(...)  ← Expensive!
Address 8 = XOR(...)  ← Expensive!
```

Compiler notices pattern and reuses:
```
Address 1 = XOR(...)     → v28
Address 2 = XOR(...)     → v27
Address 3 = XOR(...)     → v24
Address 4 = XOR(...)     → v25
Address 5 = Address_X    → v29 (reuse)
Address 6 = XOR(...)     → v23
Address 7 = Address_Y    → v26 (reuse)
Address 8 = Address_Z    → v22 (reuse)
```

Then uses instruction offset field:
```assembly
ds_read_u16 v18, v29 offset:128  ← v29 + 128 bytes
ds_read_u16 v20, v26 offset:128  ← v26 + 128 bytes
ds_read_u16 v21, v22 offset:256  ← v22 + 256 bytes
```

---

## The Answer

**No single line of C++ code explicitly creates the offsets.**

The offsets arise from:
1. **M1 = 8** creates a stride pattern (line 331)
2. **Distribution encoding** uses M1 (line 337-345)
3. **load_tile** uses the distribution (line 351)
4. **Compiler optimization** notices the pattern and adds offsets

---

## To Verify

You could test by changing M1:

```cpp
// Original (causes offsets)
constexpr index_t M1 = 16 / sizeof(DataType); // = 8

// Try different value
constexpr index_t M1 = 16 / sizeof(DataType) / 2; // = 4
```

This would change the access pattern and might eliminate or change the offsets.

But it would also change the tile distribution semantics, so the kernel would need to be redesigned.

---

## Key Insight

**The offsets are a compiler optimization, not a C++ feature.**

The C++ code (lines 331, 337-345, 351) defines a memory access pattern.

The compiler analyzes this pattern during code generation and chooses to use instruction offsets to reduce the number of address calculations.

This is why you can't "see" offset:128 in the C++ source - it only appears in the compiled assembly!
