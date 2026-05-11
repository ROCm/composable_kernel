# Search ALL GPU memory for ds_read instructions

set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

echo \n=== Searching entire address space for ds_read patterns ===\n

# Step into GPU code
stepi
stepi
stepi

# Get current PC
set $start = $pc

# Search backwards and forwards from current location
set $addr = $start - 0x10000

echo \nSearching memory ranges for ds_read instruction patterns...\n
echo This will take a moment...\n\n

set logging file memory_search.txt
set logging overwrite on
set logging on

# Search 128KB before current PC
x/32000i $start - 0x10000

set logging off

shell echo "=== Searching for ds_read ===" && grep -n "ds_read\|ds_write" memory_search.txt | head -50

quit
