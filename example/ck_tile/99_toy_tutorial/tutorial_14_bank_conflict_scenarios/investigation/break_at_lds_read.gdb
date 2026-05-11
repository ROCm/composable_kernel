# Break at the actual LDS read in tile_window.hpp

set breakpoint pending on
set pagination off

# Break at line 259 of tile_window.hpp where get_vectorized_elements is called
break tile_window.hpp:259

run

echo \n=== At get_vectorized_elements call - THE ACTUAL LDS READ ===\n

# Step into the function
step

echo \n=== Now disassemble to find ds_read ===\n

set logging file lds_read_disasm.txt
set logging overwrite on
set logging on
x/500i $pc
set logging off

shell echo "=== Searching for ds_read ===" && grep -n "ds_read\|ds_write" lds_read_disasm.txt | head -30

echo \n=== If found, break at that address and examine registers ===\n

quit
