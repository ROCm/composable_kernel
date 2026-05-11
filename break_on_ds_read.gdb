# GDB script to break on ds_read_u16 instructions directly
# Run with: rocgdb -x break_on_ds_read.gdb ./04_row_major_xor_asm

set breakpoint pending on
set pagination off

# First, let's find the ds_read instructions by searching the binary
# We'll break at load_tile first to get into GPU context
#break 08_xor_cross_warp_window_reinterpret.cpp:118
break 08_xor_cross_warp_window_reinterpret.cpp:309
run

#echo \n=== At load_tile, now searching for ds_read_u16 instructions ===\n
#
## Step into the function call (s_swappc_b64)
#stepi
#stepi
#stepi
#
## Disassemble a large range to find ds_read
#echo \n=== Disassembling 500 instructions to find ds_read_u16 ===\n
#set disassemble-next-line on
#x/500i $pc
#
#echo \n\n=== INSTRUCTIONS ===\n
#echo Look for lines like: "ds_read_u16 v14, v28" or "ds_read_u16 v18, v29 offset:128"\n
#echo \n
#echo To set breakpoint on assembly address:\n
#echo   b *0x00007fbfd8346xyz    (use the exact address you see)\n
#echo \n
#echo To examine GPU registers at that point:\n
#echo   info registers v22 v23 v24 v25 v26 v27 v28 v29\n
#echo \n
#echo To calculate bank from address in register:\n
#echo   p/x ($v28 >> 2) & 0x1f\n
#echo \n
