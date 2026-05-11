# GDB script to examine register values and calculate LDS banks
# Based on assembly at xor_kernel_lds_reads.asm lines 21-28

set breakpoint pending on
set pagination off

# Run the program
break 04_row_major_xor.cpp:351
run

echo \n=================================================================\n
echo Analyzing LDS Bank Conflicts - Live Register Values\n
echo =================================================================\n

# We need to find the actual runtime address of instruction 0x26BC
# The address in the disassembly (0x26BC) is relative, we need the absolute address
# Let's step into the GPU function and find it

echo \nStepping into GPU kernel...\n
stepi
stepi
stepi

# Now let's find where we are
echo \nCurrent PC:\n
p/x $pc

# Disassemble current location to find ds_read instructions
echo \n=== Searching for ds_read_u16 instructions ===\n
set logging file ds_instruction_search.txt
set logging overwrite on
set logging on
x/500i $pc
set logging off

echo \nSearching for ds_read in disassembly...\n
shell grep -B1 "ds_read_u16" ds_instruction_search.txt | head -40 | tee ds_read_locations.txt

echo \n=================================================================\n
echo NEXT STEPS:\n
echo 1. Look at ds_read_locations.txt above for the addresses\n
echo 2. Set breakpoint: b *0xADDRESS (use first ds_read_u16 address)\n
echo 3. Continue: c\n
echo 4. Check which lane: info lanes\n
echo 5. Switch to lane 0: lane 0\n
echo 6. Examine registers: info registers v22 v23 v24 v25 v26 v27 v28 v29\n
echo 7. Calculate banks: p/x (\$v28 >> 2) & 0x1f\n
echo =================================================================\n

# Keep session open
set confirm off
