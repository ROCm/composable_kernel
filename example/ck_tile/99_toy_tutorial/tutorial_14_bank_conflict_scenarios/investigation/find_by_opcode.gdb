# Search for ds_read_u16 by opcode pattern
# ds_read_u16 opcode starts with 0xD878xxxx

set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

echo \n=== Searching for ds_read_u16 opcode pattern (0xD878) ===\n

# Step into the code
stepi
stepi
stepi

# Search memory for the opcode pattern
# ds_read_u16 has encoding 0xD878xxxx
set $search_addr = $pc - 0x10000
set $end_addr = $pc + 0x10000

echo Searching from:
p/x $search_addr
echo to:
p/x $end_addr

echo \nSearching for opcode 0xD878... (this may take a moment)\n

# Use find command to search for the byte pattern
# ds_read_u16 v14, v28 encodes as: D8780000 0E00001C
find /w $pc-0x10000, $pc+0x10000, 0xD878

echo \nIf found, use 'x/10i ADDRESS' to disassemble from that location\n

quit
