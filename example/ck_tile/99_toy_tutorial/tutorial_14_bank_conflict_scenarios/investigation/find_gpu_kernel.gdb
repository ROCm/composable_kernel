# Find the GPU kernel function and search for ds_read instructions

set breakpoint pending on
set pagination off

run

echo \n=== Searching for GPU kernel functions ===\n
# List all functions with "Transpose" in the name
info functions Transpose

echo \n\n=== Searching for functions with "kentry" ===\n
info functions kentry

echo \n\n=== Let's try to find where GPU code is loaded ===\n
info files

echo \n\n=== Try maintenance info sections ===\n
maintenance info sections

quit
