# Break when GPU kernel is actually executing

set breakpoint pending on
set pagination off
set scheduler-locking off

# Break at the C++ source line where load_tile is called
break 04_row_major_xor.cpp:351
run

echo \n=== At breakpoint, switching to GPU mode ===\n
# Continue execution - this will launch the kernel
continue

# The kernel should now be running. Let's check if we're on GPU
echo \n=== Checking current thread/lane ===\n
info threads

echo \n=== Trying to switch to a GPU lane ===\n
info lanes

echo \n=== Current location ===\n
where

echo \n=== Let's disassemble and pipe to grep ===\n
set logging file disasm_output.txt
set logging overwrite on
set logging on
disassemble
set logging off

shell echo "=== Searching for ds_read in disassembly ===" && grep -i ds_read disasm_output.txt

echo \n=== If no ds_read found, let's try disassembling more ===\n
set logging file disasm_large.txt
set logging overwrite on
set logging on
x/2000i $pc
set logging off

shell echo "=== Searching 2000 instructions ===" && grep -i ds_read disasm_large.txt | head -20

quit
