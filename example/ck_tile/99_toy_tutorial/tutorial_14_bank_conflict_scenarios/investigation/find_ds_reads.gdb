# GDB script to find and analyze the ds_read_u16 instructions
# Run with: rocgdb -x find_ds_reads.gdb ./04_row_major_xor_asm

set breakpoint pending on
set pagination off
set scheduler-locking off

# Break at the load_tile call
break 04_row_major_xor.cpp:351
run

# Should now be at the breakpoint in load_tile
# Step into the s_swappc_b64 function call to find ds_read_u16
echo \n=== Stepping into nested function call ===\n
stepi
stepi
stepi
stepi
stepi

# Now disassemble to see where we are
echo \n=== Current location after stepping ===\n
disassemble

# Try to search for ds_read in the disassembly
echo \n=== Searching for ds_read instructions in wider range ===\n
# Disassemble 300 instructions from current location
x/300i $pc

echo \n=== If you see ds_read_u16 instructions above, set breakpoint before first one ===\n
echo \n=== Then examine registers with: info registers v22 v23 v24 v25 v26 v27 v28 v29 ===\n

# Keep session open for manual inspection
set confirm off
