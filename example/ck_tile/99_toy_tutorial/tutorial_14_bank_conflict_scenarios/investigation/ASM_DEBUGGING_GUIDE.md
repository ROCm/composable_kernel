# Assembly-Level Debugging with rocgdb

## Finding the Generated Assembly

### Method 1: Use -save-temps during compilation

```bash
cd /data0/aghamari/composable_kernel/build

# Compile with -save-temps to keep intermediate files
hipcc --offload-arch=gfx942 -save-temps=obj \
    ../example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios/07_minimal_buffer_view_lds.cpp \
    -I../include -I../library/include -I../example/ck_tile -std=c++20 -o test_asm

# Look for GPU assembly file (gfx942)
ls -lh 07_minimal_buffer_view_lds-hip-amdgcn-amd-amdhsa-gfx942.s
```

### Method 2: Extract from existing build

```bash
cd /data0/aghamari/composable_kernel/build

# Rebuild with verbose output to see compilation commands
ninja -v aa_tutorial_14_07_minimal_buffer_view_lds

# Add -save-temps to the hipcc command and re-run it manually
```

## Analyzing the Assembly

For `07_minimal_buffer_view_lds.cpp`, the compiler optimizes scalar LDS operations into **vectorized** operations:

### Original Source Code Intent
```cpp
// Scalar writes (one FP16 at a time)
for(int i = 0; i < 64; i++)
    lds_buf(i) = DataType(i);

// Scalar reads (one FP16 at a time)
for(int i = 0; i < 64; i++)
    value = lds_buf[i];
```

### Actual Generated Assembly
```asm
; The compiler vectorizes into 128-bit (8x FP16) operations:

ds_write_b128 v20, v[0:3]           ; Write 8 FP16 values (16 bytes)
ds_write_b128 v20, v[0:3] offset:16 ; Write next 8 FP16 values
; ... (total 8 writes to cover 64 FP16 elements)

ds_read_b128 v[0:3], v20            ; Read 8 FP16 values (16 bytes)
ds_read_b128 v[4:7], v20 offset:16  ; Read next 8 FP16 values
; ... (total 8 reads to cover 64 FP16 elements)
```

### LDS Instruction Types

| Instruction | Size | Description |
|-------------|------|-------------|
| `ds_write_b16` | 2 bytes | Write single FP16 |
| `ds_write_b32` | 4 bytes | Write single FP32 or 2x FP16 |
| `ds_write_b64` | 8 bytes | Write 4x FP16 |
| `ds_write_b128` | 16 bytes | Write 8x FP16 |
| `ds_read_b16` | 2 bytes | Read single FP16 |
| `ds_read_b32` | 4 bytes | Read single FP32 or 2x FP16 |
| `ds_read_b64` | 8 bytes | Read 4x FP16 |
| `ds_read_b128` | 16 bytes | Read 8x FP16 |

## Setting Breakpoints in Assembly

### Step 1: Find the Kernel Symbol Name

```bash
# Search for your kernel function in the assembly
grep "minimal_buffer_view_kernel" 07_minimal_buffer_view_lds-hip-amdgcn-amd-amdhsa-gfx942.s

# Output shows the mangled name:
# _Z26minimal_buffer_view_kernelPDF16_
```

### Step 2: Set Breakpoint by Function Name

```gdb
rocgdb ./bin/aa_tutorial_14_07_minimal_buffer_view_lds

# Break at the kernel entry
(gdb) break _Z26minimal_buffer_view_kernelPDF16_
Breakpoint 1 at 0x...

(gdb) run
# Kernel launches, hits breakpoint
```

### Step 3: Set Breakpoint by Instruction Address

```gdb
# After hitting initial breakpoint, disassemble
(gdb) disassemble
Dump of assembler code for function _Z26minimal_buffer_view_kernelPDF16_:
=> 0x...: v_cmp_eq_u32_e32 vcc, 0, v0
   0x...: s_and_saveexec_b64 s[2:3], vcc
   ...
   0x...: ds_write_b128 v20, v[0:3]
   0x...: ds_write_b128 v20, v[0:3] offset:16
   ...
   0x...: ds_read_b128 v[0:3], v20
   0x...: ds_read_b128 v[4:7], v20 offset:16

# Set breakpoint at first ds_read instruction
(gdb) break *0x<address_of_first_ds_read>

# Or use relative line numbers (if debug info available)
(gdb) info line
Line 58 of "07_minimal_buffer_view_lds.cpp"
```

### Step 4: Set Breakpoint by Searching for Instruction Pattern

```gdb
(gdb) disassemble
# Count down to find ds_read instructions
# Example: if ds_read is at +10 instructions from start

# Calculate offset: (current_pc + 10*instruction_size)
# AMD GCN instructions are 4 or 8 bytes

(gdb) break *($pc + 40)  # Approximate offset
```

## Examining LDS Memory During Debugging

### View LDS Memory Contents

```gdb
# After breaking in kernel, LDS address is in v20
(gdb) info registers v20
v20            0x0      0

# LDS is relative to LDS base, view at base
(gdb) x/64hh $lds_base  # 64 half-words (FP16)

# Or examine the __shared__ variable directly
(gdb) print lds_memory
(gdb) x/64hh &lds_memory
```

### Watch LDS Writes

```gdb
# Break before first ds_write
(gdb) break *<address_before_ds_write>
(gdb) continue

# Step instruction by instruction
(gdb) stepi

# After each ds_write, view LDS
(gdb) x/16hh &lds_memory  # View first 16 FP16 values
(gdb) stepi
(gdb) x/16hh &lds_memory  # See new values written
```

### Watch LDS Reads

```gdb
# Break at first ds_read
(gdb) break *<address_of_ds_read>
(gdb) continue

# View VGPR destination registers before read
(gdb) info registers v0 v1 v2 v3

# Step over the ds_read instruction
(gdb) stepi

# View VGPR registers after read (should contain LDS data)
(gdb) info registers v0 v1 v2 v3
```

## Complete Example Session

```bash
cd /data0/aghamari/composable_kernel/build

# Generate assembly
hipcc --offload-arch=gfx942 -save-temps=obj \
    ../example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios/07_minimal_buffer_view_lds.cpp \
    -I../include -I../library/include -I../example/ck_tile -std=c++20 -g -o test_asm

# Find ds_read instructions
grep -n "ds_read" 07_minimal_buffer_view_lds-hip-amdgcn-amd-amdhsa-gfx942.s

# Output:
# 58:	ds_read_b128 v[0:3], v20
# 59:	ds_read_b128 v[4:7], v20 offset:16
# ...

# Start debugger
rocgdb ./test_asm
```

```gdb
# Inside rocgdb
(gdb) break _Z26minimal_buffer_view_kernelPDF16_
(gdb) run

# Hit breakpoint, now disassemble
(gdb) disassemble
# Find address of line 58 (first ds_read_b128)

(gdb) break *0x<address_of_line_58>
(gdb) continue

# Now at first ds_read
(gdb) info threads
# Find GPU thread (agent, queue, dispatch, wave, lane)

(gdb) thread <gpu_thread_id>

# View LDS memory before read
(gdb) x/64hh &lds_memory

# View destination VGPRs before read
(gdb) info registers v0 v1 v2 v3

# Execute the ds_read instruction
(gdb) stepi

# View VGPRs after read (should contain first 8 FP16 values from LDS)
(gdb) info registers v0 v1 v2 v3

# v0 contains 2x FP16, v1 contains 2x FP16, etc.
# Total: v[0:3] = 8x FP16 values
```

## Tips

1. **Compiler Vectorization**: Scalar code often becomes vectorized. Don't expect to see `ds_read_b16` for single FP16 reads.

2. **LDS Address Register**: Usually `v20` holds the LDS offset (often 0 for static shared memory).

3. **VGPR Packing**: Multiple FP16 values pack into 32-bit VGPRs (2x FP16 per VGPR).

4. **Barriers**: Look for `s_barrier` and `s_waitcnt lgkmcnt(N)` to understand synchronization.

5. **Unrolled Loops**: The compiler unrolls loops completely, so you see multiple identical instructions instead of a loop.

## Why This Matters

Understanding assembly is crucial for:
- **Bank Conflict Analysis**: Seeing actual LDS address patterns
- **Performance Debugging**: Understanding why operations are slow
- **Compiler Behavior**: Seeing what optimizations are applied
- **Production Kernels**: Complex templates expand to assembly - this is the ground truth

## Next: Bank Conflict Detection in Assembly

To see bank conflicts, look at the LDS addresses in `ds_read`/`ds_write` instructions across multiple lanes:
- Same bank, different address → **Bank conflict**
- Different banks → **No conflict**
- Same address (broadcast) → **No conflict** (hardware handles this)

See tutorials 01-04 for bank conflict patterns and XOR-based solutions.
