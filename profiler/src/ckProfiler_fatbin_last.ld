/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates. */
/* SPDX-License-Identifier: MIT */

/*
 * ckProfiler links every device instance library, so its loadable image is
 * larger than the 2 GB reach of x86-64 PC-relative relocations.
 *
 * By default the linker emits .hip_fatbin, which is read-only and holds the
 * embedded device images for every target, between .rodata and .text. That
 * pushed .text about 1.7 GB above .rodata and put each object's own constant
 * pool out of reach of its own code, which is where the "relocation
 * R_X86_64_PC32 out of range" failures came from.
 *
 * Emitting the fat binary after .bss instead keeps code and constants together
 * at the front of the image and leaves the device images, which nothing
 * addresses PC-relative, at the end.
 *
 * This is an INSERT script, so it adds one placement rule and leaves the rest
 * of the default layout alone.
 */

SECTIONS
{
  .hip_fatbin : { *(.hip_fatbin) }
}
INSERT AFTER .bss;
