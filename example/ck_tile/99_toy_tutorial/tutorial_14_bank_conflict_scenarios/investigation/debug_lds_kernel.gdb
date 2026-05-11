# GDB script to debug LDS bank conflicts - break in kernel

set breakpoint pending on
set pagination off
set scheduler-locking off

# Run the program
run

# The program will finish - that's okay
# Let's just examine the assembly offline to understand the bank conflicts

# For now, let's just show what we know from the assembly:
echo \n=== LDS Bank Conflict Analysis ===\n
echo The 8 ds_read_u16 instructions:\n
echo \n
echo 1. ds_read_u16 v14, v28          <- Uses v28 (no offset)\n
echo 2. ds_read_u16 v15, v27          <- Uses v27 (no offset)\n
echo 3. ds_read_u16 v16, v24          <- Uses v24 (no offset)\n
echo 4. ds_read_u16 v17, v25          <- Uses v25 (no offset)\n
echo 5. ds_read_u16 v18, v29 offset:128  <- Uses v29 + 128 (HARDCODED!)\n
echo 6. ds_read_u16 v19, v23          <- Uses v23 (no offset)\n
echo 7. ds_read_u16 v20, v26 offset:128  <- Uses v26 + 128 (HARDCODED!)\n
echo 8. ds_read_u16 v21, v22 offset:256  <- Uses v22 + 256 (HARDCODED!)\n
echo \n
echo Reads 5, 7, 8 have hardcoded offsets (128, 128, 256 bytes)\n
echo These bypass the XOR transformation that was applied to the base addresses!\n
echo Result: 3 out of 8 reads cause bank conflicts\n

quit
