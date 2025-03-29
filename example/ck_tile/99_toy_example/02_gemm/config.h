
#if defined(KERNEL_A)
    #define USING_MFMA_32x32x_8x2
#elif defined(KERNEL_B)
    #define USING_MFMA_16x16x16
#elif defined(KERNEL_C)
    #define USING_MFMA_16x16x_16x2
#elif defined(KERNEL_D)
    #define USING_MFMA_16x16x_16x2
    #define USING_XOR_BASED_BANK_CONFLICT_FREE
#elif defined(KERNEL_E)
    #define USING_MFMA_16x16x_16x2
    #define USING_XOR_BASED_BANK_CONFLICT_FREE
    #define ADJUST_BLOCK_TILE_SHAPE
#elif defined(KERNEL_F)
    #define USING_MFMA_16x16x_16x2
    #define USING_XOR_BASED_BANK_CONFLICT_FREE
    #define ADJUST_BLOCK_TILE_SHAPE
    #define ENABLE_PREFETCH
    #define ENABLE_INSTRUCTION_SCH
#elif defined(KERNEL_G)
    #define USING_MFMA_16x16x_16x2
    #define USING_XOR_BASED_BANK_CONFLICT_FREE
    #define ADJUST_BLOCK_TILE_SHAPE
    #define ENABLE_PREFETCH
    #define ENABLE_INSTRUCTION_SCH
    #define ENABLE_CACHE_AWARE_WG_SCH
#else
    #define NAIVE_IMPLEMENTATION 
#endif

