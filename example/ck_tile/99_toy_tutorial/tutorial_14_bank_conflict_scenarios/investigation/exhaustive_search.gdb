# Exhaustive search for ds_read_u16 in all reachable memory

set breakpoint pending on
set pagination off

break buffer_view.hpp:831
run

echo \n=== Searching BACKWARDS 50000 instructions ===\n
set logging file search_backward.txt
set logging overwrite on
set logging on
x/50000i $pc-0x20000
set logging off

shell echo "=== Backward search results ===" && grep -c "ds_read_u16" search_backward.txt || echo "Not found going backward"

echo \n=== Searching FORWARD 50000 instructions ===\n
set logging file search_forward_big.txt
set logging overwrite on
set logging on
x/50000i $pc
set logging off

shell echo "=== Forward search results ===" && grep -c "ds_read_u16" search_forward_big.txt || echo "Not found going forward"

echo \n=== If found, showing context ===\n
shell grep -B2 -A2 "ds_read_u16" search_backward.txt search_forward_big.txt | head -30

quit
