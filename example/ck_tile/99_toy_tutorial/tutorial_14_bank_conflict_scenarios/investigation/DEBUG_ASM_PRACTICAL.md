# Practical Assembly Debugging with rocgdb - Working Solution

## The Reality

**The .s file cannot be directly used as source in rocgdb** because:
1. The GPU code object is embedded in a fat binary format
2. Debug info for assembly doesn't propagate through HIP's compilation pipeline
3. rocgdb can debug the GPU code, but shows it as runtime disassembly, not .s source

## Practical Working Solution

Use **both** tools together:
- **The .s file** = ground truth showing what instructions exist
- **rocgdb** = runtime inspection of registers, memory, and execution

### Step 1: Generate and Study the .s File

```bash
cd /data0/aghamari/composable_kernel/build

# Generate assembly
hipcc --offload-arch=gfx942 -save-temps=obj \
    ../example/ck_tile/99_toy_tutorial/tutorial_14_bank_conflict_scenarios/08_force_lds_reads.cpp \
    -I../include -I../library/include -I../example/ck_tile -std=c++20 -o tut08

# View the GPU assembly
cat -n 08_force_lds_reads-hip-amdgcn-amd-amdhsa-gfx942.s | less
```

Key lines in the .s file:
```asm
 8 _Z22force_lds_reads_kernelPDF16_:   ; Kernel entry
10 	v_cmp_gt_u32_e32 vcc, 64, v0       ; Check if tid < 64
11 	v_lshlrev_b32_e32 v1, 1, v0        ; v1 = tid * 2 (byte address)
15 	v_cvt_f32_u32_e32 v2, v0            ; Convert tid to float
16 	v_cvt_f16_f32_e32 v2, v2            ; Convert to FP16
17 	ds_write_b16 v1, v2                 ; <-- WRITE TO LDS
21 	s_barrier                           ; Barrier/sync
25 	v_add_u32_e32 v0, 1, v0             ; v0 = tid + 1
26 	v_and_b32_e32 v0, 63, v0            ; v0 = (tid + 1) % 64
27 	v_lshlrev_b32_e32 v0, 1, v0         ; v0 = address * 2
29 	ds_read_u16 v0, v0                  ; <-- READ FROM LDS
```

### Step 2: Debug with rocgdb (Using Instruction Counting)

```bash
rocgdb ./tut08
```

Inside rocgdb:

```gdb
# Break at kernel
(gdb) break force_lds_reads_kernel
(gdb) run

# Hit breakpoint - now at kernel entry
# The .s file shows ds_write at line 17, which is ~7 instructions from entry

# Disassemble to see addresses
(gdb) disassemble
Dump of assembler code for function _Z22force_lds_reads_kernelPDF16_:
=> 0x... <+0>:  v_cmp_gt_u32_e32 vcc, 64, v0
   0x... <+4>:  v_lshlrev_b32_e32 v1, 1, v0
   0x... <+8>:  s_and_saveexec_b64 s[2:3], vcc
   ...

# Count instructions - ds_write should be around instruction 7-8
# Set breakpoint by stepping
(gdb) stepi
(gdb) stepi
... (repeat until you hit ds_write)

# Or estimate the offset
(gdb) break *_Z22force_lds_reads_kernelPDF16_+28
```

### Step 3: Inspect at ds_write

```gdb
# When you hit ds_write_b16
(gdb) x/i $pc
=> ds_write_b16 v1, v2

# Check registers (thread 0, lane 0)
(gdb) info registers v0 v1 v2
v0 = 0     # thread ID
v1 = 0     # LDS address (tid * 2)
v2 = 0x0   # FP16 value (0.0)

# Switch to lane 5
(gdb) thread apply all info registers v1 | grep "Thread.*12"
# (Thread 12 is usually lane 5)

# Step over the ds_write
(gdb) stepi
```

### Step 4: Verify LDS Contents After Barrier

```gdb
# Continue to s_barrier
# From .s file, barrier is at line 21, a few instructions after ds_write

(gdb) stepi  # Keep stepping
(gdb) x/i $pc
# When you see s_barrier

# Step past barrier
(gdb) stepi

# Now LDS should have all writes completed
# View LDS memory (if you can find the address)
(gdb) x/64hh <lds_address>
```

## Alternative: Use Print Statements for Verification

Since debugging LDS in rocgdb is difficult, add debug output to your kernel:

```cpp
__global__ void force_lds_reads_kernel(DataType* output)
{
    __shared__ DataType lds_memory[kLdsSize];
    int tid = threadIdx.x;

    auto lds_buf = make_buffer_view<address_space_enum::lds>(lds_memory, number<kLdsSize>{});

    if(tid < kLdsSize)
    {
        lds_buf(tid) = DataType(tid);

        // DEBUG: Print address being written
        if(tid < 4)  // Only first 4 threads
        {
            printf("Thread %d writes to LDS[%d]\n", tid, tid);
        }
    }

    __syncthreads();

    if(tid < kLdsSize)
    {
        int read_idx = (tid + 1) % kLdsSize;
        DataType value = lds_buf[read_idx];

        // DEBUG: Print what was read
        if(tid < 4)
        {
            printf("Thread %d reads LDS[%d] = %f\n", tid, read_idx, (float)value);
        }

        output[tid] = value;
    }
}
```

Run and see the output - much easier than rocgdb for verification!

## Recommended Workflow

**For understanding LDS access patterns:**
1. Read the .s file - it shows EXACTLY what instructions execute
2. Manually analyze:
   - Which lanes access which LDS addresses
   - Calculate bank assignments: `(address / 4) % 32`
   - Identify bank conflicts

**For runtime verification:**
1. Use printf in kernel (easiest)
2. Use rocgdb to inspect specific registers at breakpoints
3. Use the .s file to know WHICH instructions to break at

## Example: Analyzing Bank Conflicts from .s File

From tutorial 08's .s file:

```asm
Line 17: ds_write_b16 v1, v2
  v1 = tid * 2 (each thread writes to its own address)
  Addresses: 0, 2, 4, 6, ..., 126

  Bank calculation:
  Lane 0: addr=0, bank=(0/4)%32=0
  Lane 1: addr=2, bank=(2/4)%32=0  <- SAME BANK!
  Lane 2: addr=4, bank=(4/4)%32=1
  Lane 3: addr=6, bank=(6/4)%32=1  <- SAME BANK!

  Result: 2-way bank conflicts (pairs of lanes hit same bank)
```

**You can do this analysis WITHOUT running rocgdb** - just by reading the .s file!

## Summary

- ❌ Can't use .s file as source in rocgdb directly
- ✅ Use .s file for understanding instructions
- ✅ Use rocgdb for runtime register/memory inspection
- ✅ Use printf for quick verification
- ✅ Calculate bank conflicts manually from .s file

The .s file IS your debuggable source - you just read it with your eyes instead of rocgdb showing it!
