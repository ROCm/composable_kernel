# GDB script to search for ds_read_u16 in a large memory range
set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

# We're at the s_swappc_b64 call
# Let's step into it
echo \n=== Stepping into function call ===\n
stepi

# Now disassemble 1000 instructions forward
echo \n=== Searching 1000 instructions for ds_read ===\n
set logging file ds_read_search.txt
set logging on
x/1000i $pc
set logging off

echo \n=== Saved disassembly to ds_read_search.txt ===\n
echo Now grep for ds_read:\n
shell grep -n "ds_read" ds_read_search.txt | head -30

quit
