#!/bin/bash
export KMDUMPISA=1
export KMDUMPLLVM=1
export KMOPTLLC=-mattr=+enable-ds128

make -j driver
/opt/rocm/hcc/bin/llvm-objdump -mcpu=gfx906 -source -line-numbers driver/dump-gfx906.isabin > driver/dump-gfx906.isabin.asm
