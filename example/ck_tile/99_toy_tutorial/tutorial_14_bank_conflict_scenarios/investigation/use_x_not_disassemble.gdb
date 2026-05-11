# Use x/i instead of disassemble to see across function boundaries

set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

echo \n=== At load_tile, using x/i to see ALL instructions ===\n

# Use x/1000i instead of disassemble - this ignores function boundaries
set logging file x_disasm.txt
set logging overwrite on
set logging on
x/1000i $pc
set logging off

echo \n=== Searching for ds_read_u16 ===\n
shell grep -n "ds_read_u16" x_disasm.txt | head -30

echo \n=== If found, the line numbers show addresses ===\n
echo === Use: b *0xADDRESS to break at that instruction ===\n

quit
