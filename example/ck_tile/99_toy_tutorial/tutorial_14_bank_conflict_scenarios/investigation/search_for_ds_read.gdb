# Search for ds_read instructions in GPU memory

set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

echo \n=== At breakpoint, stepping into GPU code ===\n
stepi
stepi
stepi

echo \n=== Current PC ===\n
p/x $pc

echo \n=== Searching 40KB around current PC for ds_read ===\n

set logging file full_search.txt
set logging overwrite on
set logging on
x/10000i $pc - 0x5000
set logging off

shell echo "=== Found ds instructions ===" && grep -n "ds_read\|ds_write" full_search.txt | head -50

echo \n=== If found, check full_search.txt for complete listing ===\n

quit
