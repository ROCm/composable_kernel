# Break at the ACTUAL LDS memory read - the pointer dereference that becomes ds_read_u16

set breakpoint pending on
set pagination off

# Break at line 831 of buffer_view.hpp - the pointer dereference
break buffer_view.hpp:831

run

echo \n=== At the actual LDS pointer dereference (line 831) ===\n
echo === This should compile to ds_read_u16 ===\n

# Show current location
where

# Disassemble current location
echo \n=== Disassembling around current instruction ===\n
x/20i $pc-0x20

echo \n=== Current instruction ===\n
x/1i $pc

echo \n=== Next 50 instructions ===\n
set logging file actual_lds_read.txt
set logging overwrite on
set logging on
x/50i $pc
set logging off

shell echo "=== Searching for ds_read ===" && grep -i "ds_read\|ds_write" actual_lds_read.txt

echo \n=== Examining address calculation ===\n
echo i + linear_offset should be in a register\n
info registers

quit
