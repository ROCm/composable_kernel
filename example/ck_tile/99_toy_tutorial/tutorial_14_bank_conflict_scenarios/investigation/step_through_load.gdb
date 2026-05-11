# Step through load_tile to find ds_read instructions

set breakpoint pending on
set pagination off

break 04_row_major_xor.cpp:351
run

echo \n=== At load_tile call, now stepping through it ===\n

# Step into load_tile
step

# Now we should be inside load_tile
echo \n=== Inside load_tile, disassembling ===\n
pipe disassemble | head -100

echo \n=== Searching for ds in disassembly ===\n
pipe disassemble | grep ds

# Step instruction by instruction and check for ds_read
echo \n=== Stepping 50 instructions looking for ds_read ===\n

set logging file step_by_step.txt
set logging overwrite on
set logging on

define step_and_show
    stepi
    x/1i $pc
end

step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show
step_and_show

set logging off

shell grep -i "ds_read" step_by_step.txt

quit
