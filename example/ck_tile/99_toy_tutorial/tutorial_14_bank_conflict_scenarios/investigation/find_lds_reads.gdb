# Find the ds_read_u16 instructions in the kernel

# You're already in the kernel, so let's search for ds_read instructions
# In rocgdb, use 'disassemble /m' to show source + assembly

# First, let's see the current PC
info registers pc

# Search for ds_read instructions around the current location
# The ds_read_u16 instructions should be near the load_tile() call

# Try disassembling from a different starting point
# Look for the pattern: ds_read_u16 followed by s_waitcnt lgkmcnt(0)

# You can also search memory for the ds_read instruction opcode
# ds_read_u16 has opcode 0xD878xxxx

echo \nSearching for ds_read_u16 instructions...\n
echo Look for instruction pattern: D8 78 00 00\n
echo \n
echo In rocgdb, try:\n
echo   x/100i $pc-1000\n
echo   x/100i $pc+1000\n
echo \n
echo Or search for "ds_" instructions:\n
echo   disassemble $pc-2000,$pc+2000\n

