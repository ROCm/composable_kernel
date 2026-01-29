def get_xdl_per_wave_2(block_size: int, 
                       nn_per_block: int, 
                       mm_per_block: int, 
                       nn_per_xdl: int, 
                       mm_per_xdl: int, 
                       nn_xdl_per_wave: int, 
                       is_wave64: bool) -> int:
    """
    Calculate the number of XDL operations per wave in the N dimension.
    
    Returns 0 if configuration is invalid.
    """
    waves = block_size // 64 if is_wave64 else block_size // 32
    
    if nn_xdl_per_wave == 0 or nn_per_xdl == 0:
        return 0
    
    m_waves = nn_per_block // (nn_xdl_per_wave * nn_per_xdl)
    assert m_waves > 0, "MWaves must be greater than 0"
    
    n_waves = waves // m_waves
    if n_waves == 0:
        return 0
    
    if mm_per_block % (mm_per_xdl * n_waves) == 0:
        return mm_per_block // (n_waves * mm_per_xdl)
    else:
        return 0

BlockSize = 256
MPerBlock = 128
NPerBlock = 32 #64
MPerXdl = 32
NPerXdl = 32
MXdlPerWave = 2
NXdlPerWave = 1

result = get_xdl_per_wave_2(BlockSize, NPerBlock, MPerBlock, NPerXdl, MPerXdl, NXdlPerWave, is_wave64=True)
print(f"XDL operations per wave in M dimension: {result}")