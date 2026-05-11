# Proper rocgdb script for GPU kernel debugging

set breakpoint pending on
set pagination off

# Start the program
start

# Set a breakpoint on the kernel
# We need to find the kernel name first
info functions

# For now, let's just run and see what happens
continue

# When kernel launches, we should see GPU threads
info threads

# Show GPU lanes (AMD terminology for wavefront threads)
info lanes

# We want to examine VGPR registers v22-v29 which hold LDS addresses
# These are used in the ds_read_u16 instructions

info registers v22
info registers v23  
info registers v24
info registers v25
info registers v26
info registers v27
info registers v28
info registers v29

quit
