#!/usr/bin/env python3

import sys

WaveSize64 = 64  # Assume gfx942, gfx950, or similar with wave size 64.

def GetNXdlPerWave(BlockSize, MPerBlock, MPerXdl, MXdlPerWave, NPerBlock, NPerXdl) -> int:
    Waves  = BlockSize // 64
    MWaves = MPerBlock // (MXdlPerWave * MPerXdl)
    if (MWaves == 0):
        raise ValueError("MWaves cannot be zero.")
    NWaves = Waves // MWaves
    if (NWaves == 0):
        raise ValueError("NWaves cannot be zero.")
    if NPerBlock % (NPerXdl * NWaves) == 0:
        return NPerBlock / (NWaves * NPerXdl)
    else:
        raise ValueError("NPerBlock is not divisible by (NPerXdl * NWaves).")
    

def IsValidGemmCompilationParameter(BlockSize, MPerBlock, MPerXdl, MXdlPerWave, NPerBlock, NPerXdl, NXdlPerWave) -> bool:
    if MXdlPerWave > 0 and NXdlPerWave > 0:
        MWaves = MPerBlock // (MXdlPerWave * MPerXdl)
        NWaves = NPerBlock // (NXdlPerWave * NPerXdl)
        if MWaves > 0 and NWaves > 0:
            WaveSize = BlockSize // (MWaves * NWaves)
                #(BlockSize * MXdlPerWave * MPerXdl * NXdlPerWave * NPerXdl) // (MPerBlock * NPerBlock)
            if WaveSize == WaveSize64:
                return True
            else:
                # Print debug info
                print(f"Invalid WaveSize: {WaveSize}, expected 64.")
                print(f"Computed MWaves: {MWaves}, NWaves: {NWaves}")
    return False

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: check_block_tiling.py <BlockSize> <MPerBlock> <MPerXdl> <MXdlPerWave> <NPerBlock> <NPerXdl> <NXdlPerWave>")
        sys.exit(1)

    BlockSize = int(sys.argv[1])
    MPerBlock = int(sys.argv[2])
    MPerXdl = int(sys.argv[3])
    MXdlPerWave = int(sys.argv[4])
    NPerBlock = int(sys.argv[5])
    NPerXdl = int(sys.argv[6])
    NXdlPerWave_init = int(sys.argv[7])

    print(f"NXdlPerWave (input): {NXdlPerWave_init}")
    NXdlPerWave = GetNXdlPerWave(BlockSize, MPerBlock, MPerXdl, MXdlPerWave, NPerBlock, NPerXdl)
    print(f"NXdlPerWave (computed): {NXdlPerWave}")

    if IsValidGemmCompilationParameter(BlockSize, MPerBlock, MPerXdl, MXdlPerWave, NPerBlock, NPerXdl, NXdlPerWave):
        print("Valid GEMM compilation parameters.")
        sys.exit(0)
    else:
        print("Invalid GEMM compilation parameters.")
        sys.exit(1)