# Tutorial 07: Debugging buffer_view with rocgdb

## Purpose

`07_minimal_buffer_view_lds.cpp` provides the simplest possible example of using `buffer_view` with LDS memory. This is ideal for:
- Learning the `buffer_view` API
- Understanding LDS addressing
- Debugging with rocgdb (no heavy template abstractions)
- Stepping through LDS reads/writes in a debugger

## What This Example Does

1. Allocates 64 FP16 elements in LDS (128 bytes)
2. Creates a `buffer_view<address_space_enum::lds>` wrapping the LDS memory
3. Thread 0 writes sequential values [0, 1, 2, ..., 63] using `lds_buf(i) = value`
4. Thread 0 reads them back using `value = lds_buf[i]`
5. Copies results to global memory for verification

## Key API Usage

```cpp
// Create buffer_view for LDS
auto lds_buf = make_buffer_view<address_space_enum::lds>(
    lds_memory,           // __shared__ pointer
    number<kLdsSize>{}    // buffer size
);

// Write using operator()(index)
lds_buf(i) = DataType(value);

// Read using operator[](index)
DataType value = lds_buf[i];
```

## Running the Test

```bash
cd /data0/aghamari/composable_kernel/build
./bin/aa_tutorial_14_07_minimal_buffer_view_lds
```

Expected output:
```
✓ Test PASSED
```

## Debugging with rocgdb

### Basic Setup

```bash
cd /data0/aghamari/composable_kernel/build
rocgdb ./bin/aa_tutorial_14_07_minimal_buffer_view_lds
```

### Useful Commands

```gdb
# Break in the kernel
(gdb) break minimal_buffer_view_kernel
(gdb) run

# List threads (find GPU threads)
(gdb) info threads

# Switch to GPU thread 0 (wavefront 0, lane 0)
(gdb) thread <gpu_thread_number>

# Inspect buffer_view structure
(gdb) print lds_buf
(gdb) print lds_buf.p_data_         # LDS base address
(gdb) print lds_buf.buffer_size_    # Should be 64

# View LDS memory contents (64 half-words in hex)
(gdb) x/64hh lds_memory

# Step through the write loop and watch LDS update
(gdb) break 07_minimal_buffer_view_lds.cpp:54
(gdb) continue
(gdb) next
(gdb) x/10hh lds_memory             # See first 10 values

# Continue stepping and watch memory fill
(gdb) next
(gdb) x/10hh lds_memory
```

### Finding ds_read/ds_write Instructions

```gdb
# After breaking in kernel, disassemble
(gdb) disassemble

# Look for:
#   ds_write_b16  - Write 16-bit value to LDS
#   ds_read_b16   - Read 16-bit value from LDS
```

## Comparison with Production Kernels

### This Example (Minimal)
- Direct `buffer_view` usage
- No `tile_distribution`, `tile_window`, `tensor_view`
- Single thread operation
- Sequential access pattern
- Easy to debug: ~120 lines total

### Production Kernels (Complex)
- Heavily templated abstractions
- Tile distribution across warps/blocks
- Tile windows with sweeping
- Transpose patterns
- XOR addressing for bank conflict avoidance
- Hard to debug: thousands of lines of template expansions

## Next Steps

After understanding this minimal example:

1. **Multi-thread version**: Each thread writes/reads its own element
2. **Vector access version**: Use `get<half4>()` and `set<half4>()` for vector loads
3. **Transpose version**: Write row-major, read column-major
4. **Bank conflict analysis**: Calculate and print bank assignments
5. **XOR addressing**: Apply XOR transform to avoid bank conflicts

See the other tutorials (01-06) for these advanced patterns.

## Files

- `07_minimal_buffer_view_lds.cpp` - Main implementation
- `CMakeLists.txt` - Build configuration (line 172-173)
- `/include/ck_tile/core/tensor/buffer_view.hpp` - buffer_view API reference (lines 755-900)
