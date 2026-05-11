# Break at the kentry function where ds_read should be

set breakpoint pending on
set pagination off

# Try to break at the mangled kentry function name
break _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_

run

echo \n=== In kentry function ===\n
echo Current PC:
p/x $pc

# The ds_read instructions should be at offset +0x26BC through +0x26F4
# Calculate the address
echo \n=== Looking for ds_read at offsets 0x26BC-0x26F4 ===\n

# Disassemble large range
set logging file kentry_disasm.txt
set logging overwrite on
set logging on
x/5000i $pc
set logging off

shell echo "=== Searching for ds_read ===" && grep -n "ds_read" kentry_disasm.txt | head -30

# If found, show the address
shell echo "=== First ds_read address ===" && grep -m1 "ds_read_u16" kentry_disasm.txt | awk '{print $1}'

quit
