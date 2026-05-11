# Setting Breakpoints on Assembly in rocgdb

## Yes, you can break on assembly addresses!

### Method 1: Break on specific address
```gdb
# Use asterisk (*) before address
b *0x00007fbfd8346xyz

# Example:
b *0x00007fbfd83464d0
```

### Method 2: Break on instruction pattern
```gdb
# Unfortunately rocgdb doesn't support regex breakpoints on instruction mnemonics
# So you need to find the address first, then break on it
```

### Method 3: Find and break on ds_read instructions

**Step 1:** Get into GPU kernel context
```gdb
b 04_row_major_xor.cpp:351
run
```

**Step 2:** Step into nested call
```gdb
stepi
stepi
stepi
```

**Step 3:** Disassemble large range
```gdb
x/500i $pc
```

**Step 4:** Find ds_read_u16 instructions in output
Look for lines like:
```
0x00007fbfd83464xx:  ds_read_u16 v14, v28
0x00007fbfd83464yy:  ds_read_u16 v18, v29 offset:128
```

**Step 5:** Set breakpoint on FIRST ds_read address
```gdb
b *0x00007fbfd83464xx
c  # continue to that breakpoint
```

**Step 6:** Examine registers
```gdb
info registers v22 v23 v24 v25 v26 v27 v28 v29
```

**Step 7:** Calculate banks
```gdb
# For each register vNN containing LDS address:
p/x ($vNN >> 2) & 0x1f
```

## What you'll find:

The 8 ds_read_u16 instructions use these registers:
- v28, v27, v24, v25 (no offset) → XOR working correctly
- v29 + 128 → Hardcoded offset AFTER XOR
- v23 (no offset) → XOR working
- v26 + 128 → Hardcoded offset AFTER XOR
- v22 + 256 → Hardcoded offset AFTER XOR

The registers v22, v29, v26 will have their XOR-transformed addresses, but then +128/+256 shifts them into conflicting banks!

## Example session:

```gdb
(rocgdb) b *0x00007fbfd83464d0
(rocgdb) c
(rocgdb) info lanes
(rocgdb) lane 0
(rocgdb) info registers v28
v28            0x0
(rocgdb) p/x (0x0 >> 2) & 0x1f
$1 = 0x0  # Bank 0
```
