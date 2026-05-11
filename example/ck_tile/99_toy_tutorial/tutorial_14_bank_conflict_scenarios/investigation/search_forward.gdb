# Search FORWARD from PC for ds_read instructions

set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

echo \n=== Stepping into GPU code ===\n
stepi
stepi
stepi

echo \n=== Current PC ===\n
p/x $pc

echo \n=== Searching FORWARD from PC for ds_read ===\n

set logging file forward_search.txt
set logging overwrite on
set logging on
x/5000i $pc
set logging off

shell echo "=== Found ds instructions ===" && grep -n "ds_read\|ds_write" forward_search.txt | head -50

quit
