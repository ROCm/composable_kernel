# Simple script to find ds_read instructions

set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

echo \n=== At breakpoint line 351 ===\n

# Disassemble current function and search
pipe disassemble | grep -i ds

# If nothing found, try stepping and disassembling
echo \n=== Stepping into code ===\n
stepi
stepi
stepi
stepi
stepi

pipe disassemble | grep -i ds

echo \n=== Disassembling 2000 instructions from PC ===\n
set logging file large_disasm.txt
set logging overwrite on
set logging on
x/2000i $pc
set logging off

shell grep -i "ds_read\|ds_write" large_disasm.txt | head -30

quit
