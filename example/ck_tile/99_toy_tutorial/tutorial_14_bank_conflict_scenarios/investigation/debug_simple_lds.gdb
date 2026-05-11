# Debug simple LDS test to see if we can find ds_read

set breakpoint pending on
set pagination off

# Break at the LDS read line
break simple_lds_test.cpp:21
run

echo \n=== At LDS read: float value = lds[read_idx] ===\n

# Disassemble current location
echo \n=== Current function disassembly ===\n
disassemble

echo \n=== Searching with x/i for ds instructions ===\n
set logging file simple_lds_disasm.txt
set logging overwrite on
set logging on
x/2000i $pc
set logging off

shell echo "=== Searching for ds_ instructions ===" && grep -i "ds_" simple_lds_disasm.txt | head -30

echo \n=== If found, showing ds_read lines ===\n
shell grep -i "ds_read" simple_lds_disasm.txt

quit
