# Find ds_read by starting from store_tile and continuing forward

set breakpoint pending on
set pagination off

# Break AFTER the store to LDS
break 04_row_major_xor.cpp:322
run

echo \n=== At block_sync after store_tile ===\n
echo === The next load_tile should contain ds_read_u16 ===\n

# Continue to the next breakpoint (should hit load_tile)
break 04_row_major_xor.cpp:351
continue

echo \n=== At load_tile from LDS ===\n

# Search for ds instructions from here
set logging file from_lds_load.txt
set logging overwrite on
set logging on
x/2000i $pc
set logging off

shell echo "=== ds_read search ===" && grep -n "ds_read" from_lds_load.txt | head -20

# Also search a wider range
shell echo "=== ANY ds instruction ===" && grep -n "^\s*0x.*:\s*ds_" from_lds_load.txt | head -30

quit
