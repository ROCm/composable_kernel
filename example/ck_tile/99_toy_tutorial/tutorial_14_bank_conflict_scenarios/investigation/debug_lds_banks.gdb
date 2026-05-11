# GDB script to debug LDS bank conflicts in XOR transpose kernel

# Set AMD GPU debugging
set breakpoint pending on
set pagination off

# Run to main
break main
run

# Continue to kernel launch
# We'll break in the kernel after it starts
set scheduler-locking on

# Break on any GPU kernel
catch syscall

continue

# At this point we should be in the kernel
# Let's check which thread we're on
info threads

# Switch to a GPU thread if available
# We want to examine the LDS address registers (v22-v29)
# These are the registers used in the 8 ds_read_u16 instructions

# Print instruction pointer
info registers pc

# The 8 LDS reads use these registers:
# ds_read_u16 v14, v28
# ds_read_u16 v15, v27
# ds_read_u16 v16, v24
# ds_read_u16 v17, v25
# ds_read_u16 v18, v29 offset:128      <- +128
# ds_read_u16 v19, v23
# ds_read_u16 v20, v26 offset:128      <- +128
# ds_read_u16 v21, v22 offset:256      <- +256

# Try to print the vector registers
info registers v22 v23 v24 v25 v26 v27 v28 v29

# Calculate banks (address >> 2) & 0x1F
# Show which banks each read will hit

quit
