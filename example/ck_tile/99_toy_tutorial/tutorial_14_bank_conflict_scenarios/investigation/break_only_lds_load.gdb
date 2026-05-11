# Break ONLY at the LDS load_tile (line 351), skip the global load

set breakpoint pending on
set pagination off

# Skip the first load_tile at line 319 by breaking at 322
break 04_row_major_xor.cpp:322
run

echo \n=== Passed the global load (line 319), now at block_sync ===\n

# Now break at the LDS load_tile
break 04_row_major_xor.cpp:351
continue

echo \n=== At LDS load_tile (line 351) ===\n
echo === This should lead to ds_read_u16 ===\n

# Step into the load_tile
step
step
step

echo \n=== After stepping into load_tile ===\n
where

# Now break inside buffer_view::get
break buffer_view.hpp:831
continue

echo \n=== Inside buffer_view::get for LDS ===\n

# Disassemble from here
set logging file lds_buffer_view.txt
set logging overwrite on
set logging on
x/1000i $pc
set logging off

shell echo "=== Searching for ds_read ===" && grep -i "ds_read" lds_buffer_view.txt | head -20

shell echo "=== Searching for ANY ds instruction ===" && grep "^\s*0x.*:\s*ds_" lds_buffer_view.txt | head -30

quit
