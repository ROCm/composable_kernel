#ifndef CK_TILE_FLATMM_UK_MFMA
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_FP8
#endif

#if CK_TILE_FLATMM_UK_MFMA == CK_TILE_FLATMM_UK_MFMA_BF8
#define _UK_MFMA_ "v_mfma_f32_16x16x32_bf8"
#elif CK_TILE_FLATMM_UK_MFMA == CK_TILE_FLATMM_UK_MFMA_FP8
#define _UK_MFMA_ "v_mfma_f32_16x16x32_fp8_fp8"
#endif

#define _FF_SYNC_ " s_waitcnt vmcnt(0) \n" \
                  " s_waitcnt lgkmcnt(0) \n" \
                  " s_barrier \n" \
                  " s_nop 16 \n"


    "s_mov_b32 s16,    %[s_res_a0] \n"
    "s_mov_b32 s17,    %[s_res_a1] \n"
    "s_mov_b32 s18,    %[s_res_a2] \n"
    "s_mov_b32 s19,    %[s_res_a3] \n"
    "s_mov_b32 s20,    %[s_res_b0] \n"
    "s_mov_b32 s21,    %[s_res_b1] \n"
    "s_mov_b32 s22,    %[s_res_b2] \n"
    "s_mov_b32 s23,    %[s_res_b3] \n"
    
#if 0
    // ----------------------------------------------------------------
    "   s_waitcnt vmcnt(0) & lgkmcnt(0)                             \n"
    "   s_barrier                                                   \n"
    // debug A vmem (4*8 = 32) -----------------------------------------
    " buffer_load_dword  %[v_dbg_0], %[v_os_a0],  s[16:19], 0 offen \n"
    " buffer_load_dword  %[v_dbg_1], %[v_os_a1],  s[16:19], 0 offen \n"
    " buffer_load_dword  %[v_dbg_2], %[v_os_a2],  s[16:19], 0 offen \n"
    " buffer_load_dword  %[v_dbg_3], %[v_os_a3],  s[16:19], 0 offen \n"
    " buffer_load_dword  %[v_dbg_4], %[v_os_a4],  s[16:19], 0 offen \n"
    " buffer_load_dword  %[v_dbg_5], %[v_os_a5],  s[16:19], 0 offen \n"
    " buffer_load_dword  %[v_dbg_6], %[v_os_a6],  s[16:19], 0 offen \n"
    " buffer_load_dword  %[v_dbg_7], %[v_os_a7],  s[16:19], 0 offen \n"
    "   s_waitcnt vmcnt(0) & lgkmcnt(0)                             \n"
    "   s_barrier                                                   \n" 
    // debug A lds (128/8 * 8 = 128) -----------------------------------
    " s_add_u32     m0, 0, %[s_m0_init]                             \n"
    " buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds         \n"
    " s_add_u32     m0,   %[s_size_per_issue], m0                   \n"
    " buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds         \n"
    " s_add_u32     m0,   %[s_size_per_issue], m0                   \n"
    " buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds         \n"
    " s_add_u32     m0,   %[s_size_per_issue], m0                   \n"
    " buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds         \n"
    " s_add_u32     m0,   %[s_size_per_issue], m0                   \n"
    " buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds         \n"
    " s_add_u32     m0,   %[s_size_per_issue], m0                   \n"
    " buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds         \n"
    " s_add_u32     m0,   %[s_size_per_issue], m0                   \n"
    " buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds         \n"
    " s_add_u32     m0,   %[s_size_per_issue], m0                   \n"
    " buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds         \n"
    "   s_waitcnt vmcnt(0) & lgkmcnt(0)                             \n"
    "   s_barrier                                                   \n" 
    " ds_read_b128 %[v_dbg4_0], %[v_os_slda] offset: 0              \n"
    " ds_read_b128 %[v_dbg4_1], %[v_os_slda] offset: 64             \n"
    " ds_read_b128 %[v_dbg4_2], %[v_os_slda] offset: 128            \n"
    " ds_read_b128 %[v_dbg4_3], %[v_os_slda] offset: 192            \n"
    " ds_read_b128 %[v_dbg4_4], %[v_os_slda] offset: 4352           \n"
    " ds_read_b128 %[v_dbg4_5], %[v_os_slda] offset: 4416           \n"
    " ds_read_b128 %[v_dbg4_6], %[v_os_slda] offset: 4480           \n"
    " ds_read_b128 %[v_dbg4_7], %[v_os_slda] offset: 4544           \n"
    "   s_waitcnt vmcnt(0) & lgkmcnt(0)                             \n"
    "   s_barrier                                                   \n"
    // debug B vmem (4*4 * 8 = 128) ------------------------------------
    " buffer_load_dwordx4  %[v_dbg4_0], %[v_os_b0], s[20:23], 0 offen               \n"
    " buffer_load_dwordx4  %[v_dbg4_1], %[v_os_b0], s[20:23], 0 offen offset:1024   \n"
    " buffer_load_dwordx4  %[v_dbg4_2], %[v_os_b0], s[20:23], 0 offen offset:2048   \n"
    " buffer_load_dwordx4  %[v_dbg4_3], %[v_os_b0], s[20:23], 0 offen offset:3072   \n"
    " buffer_load_dwordx4  %[v_dbg4_4], %[v_os_b1], s[20:23], 0 offen               \n"
    " buffer_load_dwordx4  %[v_dbg4_5], %[v_os_b1], s[20:23], 0 offen offset:1024   \n"
    " buffer_load_dwordx4  %[v_dbg4_6], %[v_os_b1], s[20:23], 0 offen offset:2048   \n"
    " buffer_load_dwordx4  %[v_dbg4_7], %[v_os_b1], s[20:23], 0 offen offset:3072   \n"
    "   s_waitcnt vmcnt(0) & lgkmcnt(0)                                             \n"
    "   s_barrier                                                                   \n"
    // ----------------------------------------------------------------

#else

#pragma region PREFETCH_A
    // A0: vmem->lds
    "s_add_u32 m0, %[s_m0_init], 0*%[smem_sz]                                       \n" // set to A0 lds
    "buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds                          \n" // A0: vmem->lds  (vmcnt)

    "s_cmp_gt_i32  %[s_loop_cnt] 1             ; move a with cond                   \n"
    "s_cselect_b32 s86, %[s_tile_os_a], 0      ; move a with cond                   \n" // tile_offset_a_bytes = 256 = 128(bf16)K
    "s_add_u32     s16, s86, s16               ; move a with cond                   \n"
    "s_addc_u32    s17, 0, s17                 ; move a with cond                   \n"

    // A1: vmem->lds
    "s_add_u32 m0, %[s_m0_init], 1*%[smem_sz]                                       \n" // set to A1 lds
    "buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)
    "s_add_u32     m0,   %[s_size_per_issue], m0                                    \n"
    "buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds                          \n" // A1: vmem->lds  (vmcnt)

    "s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond                   \n"
    "s_cselect_b32 s86, %[s_tile_os_a], 0      ; move a with cond                   \n"
    "s_add_u32     s16, s86, s16               ; move a with cond                   \n"
    "s_addc_u32    s17, 0, s17                 ; move a with cond                   \n"

    "  s_waitcnt     vmcnt(8)                                                       \n" // wait A0 vmem->lds
    "  s_barrier                                                                    \n" // 

    // A0: lds->vgpr;
    "  ds_read_b128  v[64:67], %[v_os_slda] offset:0*%[smem_sz] + 0                 \n" // A0: lds->vgpr
    "  ds_read_b128  v[68:71], %[v_os_slda] offset:0*%[smem_sz] + 64                \n" // A0: lds->vgpr
    "  ds_read_b128  v[72:75], %[v_os_slda] offset:0*%[smem_sz] + 128               \n" // A0: lds->vgpr
    "  ds_read_b128  v[76:79], %[v_os_slda] offset:0*%[smem_sz] + 192               \n" // A0: lds->vgpr
    "  ds_read_b128  v[80:83], %[v_os_slda] offset:0*%[smem_sz] + 4352              \n" // A0: lds->vgpr
    "  ds_read_b128  v[84:87], %[v_os_slda] offset:0*%[smem_sz] + 4416              \n" // A0: lds->vgpr
    "  ds_read_b128  v[88:91], %[v_os_slda] offset:0*%[smem_sz] + 4480              \n" // A0: lds->vgpr
    "  ds_read_b128  v[92:95], %[v_os_slda] offset:0*%[smem_sz] + 4544              \n" // A0: lds->vgpr    
#pragma endregion

#pragma region PREFETCH_B
    // B0: vmem->acc
    "buffer_load_dwordx4  acc[0:3],     %[v_os_b0], s[20:23], 0 offen               \n" // B0: vmem->acc  (vmcnt)
    "buffer_load_dwordx4  acc[4:7],     %[v_os_b0], s[20:23], 0 offen offset:1024   \n" // B0: vmem->acc  (vmcnt)
    "buffer_load_dwordx4  acc[8:11],    %[v_os_b0], s[20:23], 0 offen offset:2048   \n" // B0: vmem->acc  (vmcnt)
    "buffer_load_dwordx4  acc[12:15],   %[v_os_b0], s[20:23], 0 offen offset:3072   \n" // B0: vmem->acc  (vmcnt)
    "buffer_load_dwordx4  acc[16:19],   %[v_os_b1], s[20:23], 0 offen               \n" // B0: vmem->acc  (vmcnt)
    "buffer_load_dwordx4  acc[20:23],   %[v_os_b1], s[20:23], 0 offen offset:1024   \n" // B0: vmem->acc  (vmcnt)
    "buffer_load_dwordx4  acc[24:27],   %[v_os_b1], s[20:23], 0 offen offset:2048   \n" // B0: vmem->acc  (vmcnt)
    "buffer_load_dwordx4  acc[28:31],   %[v_os_b1], s[20:23], 0 offen offset:3072   \n" // B0: vmem->acc  (vmcnt)

    "s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond                   \n"
    "s_cselect_b32 s86, %[s_tile_os_b], 0      ; move b with cond                   \n" // s_tile_os_b = 4096B = 2048(bf16)K
    "s_add_u32     s20, s86, s20               ; move b with cond                   \n"
    "s_addc_u32    s21, 0, s21                 ; move b with cond                   \n"

    "  s_waitcnt     lgkmcnt(0)                                                     \n" // wait B0
#pragma endregion

"L_start%=:                                                             \n"
    //////////////////////////////////////////////////////////////////////
#pragma region LOOP_1
    // A0: vmem->lds
    // A1: lds->vgppr
    // B1: vmem->acc
    // mfma: B0 * A0
#if 0
    _FF_SYNC_
    // A0: vmem->lds
    "  s_add_u32           m0, %[s_m0_init], 0*%[smem_sz]                           \n" // set to A0 lds
    "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    _FF_SYNC_

    // A1: lds->vgpr
    "  ds_read_b128  v[96:99],   %[v_os_slda], offset:1*%[smem_sz] + 0              \n" // A1: lds->vgpr
    "  ds_read_b128  v[100:103], %[v_os_slda], offset:1*%[smem_sz] + 64             \n" // A1: lds->vgpr
    "  ds_read_b128  v[104:107], %[v_os_slda], offset:1*%[smem_sz] + 128            \n" // A1: lds->vgpr
    "  ds_read_b128  v[108:111], %[v_os_slda], offset:1*%[smem_sz] + 192            \n" // A1: lds->vgpr
    "  ds_read_b128  v[112:115], %[v_os_slda], offset:1*%[smem_sz] + 4352           \n" // A1: lds->vgpr
    "  ds_read_b128  v[116:119], %[v_os_slda], offset:1*%[smem_sz] + 4416           \n" // A1: lds->vgpr
    "  ds_read_b128  v[120:123], %[v_os_slda], offset:1*%[smem_sz] + 4480           \n" // A1: lds->vgpr
    "  ds_read_b128  v[124:127], %[v_os_slda], offset:1*%[smem_sz] + 4544           \n" // A1: lds->vgpr
    _FF_SYNC_

    // B1: vmem->acc
    "  buffer_load_dwordx4  acc[128:131], %[v_os_b0], s[20:23], 0 offen offset:0    \n" // B1: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[132:135], %[v_os_b0], s[20:23], 0 offen offset:1024 \n" // B1: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[136:139], %[v_os_b0], s[20:23], 0 offen offset:2048 \n" // B1: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[140:143], %[v_os_b0], s[20:23], 0 offen offset:3072 \n" // B1: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[144:147], %[v_os_b1], s[20:23], 0 offen offset:0    \n" // B1: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[148:151], %[v_os_b1], s[20:23], 0 offen offset:1024 \n" // B1: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[152:155], %[v_os_b1], s[20:23], 0 offen offset:2048 \n" // B1: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[156:159], %[v_os_b1], s[20:23], 0 offen offset:3072 \n" // B1: vmem->acc  (vmcnt)
    _FF_SYNC_
    
    // B0[0:3] * A0[0:3]
    _UK_MFMA_    "  %[v_acc_0], acc[0:1],   v[64:65], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[4:5],   v[68:69], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[6:7],   v[70:71], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[2:3],   v[66:67], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[8:9],   v[72:73], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[10:11], v[74:75], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[12:13], v[76:77], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[14:15], v[78:79], %[v_acc_0]                    \n"
    // B0[0:3] * A0[4:7]
    _UK_MFMA_    "  %[v_acc_1], acc[0:1],   v[80:81], %[v_acc_1]                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[2:3],   v[82:83], %[v_acc_1]                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[4:5],   v[84:85], %[v_acc_1]                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[6:7],   v[86:87], %[v_acc_1]                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[8:9],   v[88:89], %[v_acc_1]                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[10:11], v[90:91], %[v_acc_1]                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[12:13], v[92:93], %[v_acc_1]                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[14:15], v[94:95], %[v_acc_1]                    \n"
    // B0[4:7] * A0[0:3]
    _UK_MFMA_    "  %[v_acc_2], acc[16:17], v[64:65], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[18:19], v[66:67], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[20:21], v[68:69], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[22:23], v[70:71], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[24:25], v[72:73], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[26:27], v[74:75], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[28:29], v[76:77], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[30:31], v[78:79], %[v_acc_2]                    \n"
    // B0[4:7] * A0[4:7]
    _UK_MFMA_    "  %[v_acc_3], acc[16:17], v[80:81], %[v_acc_3]                    \n"
    _UK_MFMA_    "  %[v_acc_3], acc[18:19], v[82:83], %[v_acc_3]                    \n"
    _UK_MFMA_    "  %[v_acc_3], acc[20:21], v[84:85], %[v_acc_3]                    \n"
    _UK_MFMA_    "  %[v_acc_3], acc[22:23], v[86:87], %[v_acc_3]                    \n"
    _UK_MFMA_    "  %[v_acc_3], acc[24:25], v[88:89], %[v_acc_3]                    \n"
    _UK_MFMA_    "  %[v_acc_3], acc[26:27], v[90:91], %[v_acc_3]                    \n"
    _UK_MFMA_    "  %[v_acc_3], acc[28:29], v[92:93], %[v_acc_3]                    \n"
    _UK_MFMA_    "  %[v_acc_3], acc[30:31], v[94:95], %[v_acc_3]                    \n"
    _FF_SYNC_    
#else
    "  s_waitcnt     vmcnt(16)                                                      \n" // wait A1 vmem->lds
    "  s_barrier                                                                    \n"

    "  s_waitcnt     vmcnt(4)                                                       \n" // wait B0[0:3]
    //"  s_waitcnt     lgkmcnt(4)                                                   \n" // wait A0[0:3]
    "  s_waitcnt     lgkmcnt(0)                                                     \n" // wait A0: lds->vgpr
    
    "  s_add_u32 m0, %[s_m0_init], 0*%[smem_sz]                                     \n" // set to A0 lds

    // A0[0:3] * B0[0:3]
    "  buffer_load_dwordx4  acc[128:131], %[v_os_b0], s[20:23], 0 offen offset:0    \n" // B1: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[0:1],   v[64:65], %[v_acc_0]                    \n"
    "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32     m0,   %[s_size_per_issue], m0                                  \n"
    _UK_MFMA_    "  %[v_acc_0], acc[4:5],   v[68:69], %[v_acc_0]                    \n"
    "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    "  s_add_u32     m0,   %[s_size_per_issue], m0                                  \n"
    _UK_MFMA_    "  %[v_acc_0], acc[6:7],   v[70:71], %[v_acc_0]                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[2:3],   v[66:67], %[v_acc_0]                    \n"
    "  buffer_load_dwordx4  acc[132:135], %[v_os_b0], s[20:23], 0 offen offset:1024 \n" // B1: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[8:9],   v[72:73], %[v_acc_0]                    \n"
    "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[10:11], v[74:75], %[v_acc_0]                    \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[12:13], v[76:77], %[v_acc_0]                    \n"
    "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[14:15], v[78:79], %[v_acc_0]                    \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"

    //"  s_waitcnt     lgkmcnt(0)                                                   \n" // wait A0[4:7]

    // A0[4:7] * B0[0:3]
    "  buffer_load_dwordx4  acc[136:139], %[v_os_b0], s[20:23], 0 offen offset:2048 \n" // B1: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[0:1],   v[80:81], %[v_acc_1]                    \n"
    "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[2:3],   v[82:83], %[v_acc_1]                    \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[4:5],   v[84:85], %[v_acc_1]                    \n"
    "  buffer_load_dwordx4  acc[140:143], %[v_os_b0], s[20:23], 0 offen offset:3072 \n" // B1: vmem->acc  (vmcnt)   
    _UK_MFMA_    "  %[v_acc_1], acc[6:7],   v[86:87], %[v_acc_1]                    \n"
    "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[8:9],   v[88:89], %[v_acc_1]                    \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[10:11], v[90:91], %[v_acc_1]                    \n"
    "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[12:13], v[92:93], %[v_acc_1]                    \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[14:15], v[94:95], %[v_acc_1]                    \n" 
    "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds                        \n" // A0: vmem->lds  (vmcnt)

    "  s_waitcnt     vmcnt(12)                                                      \n" // wait B0[4:7]

    // A0[0:3] * B0[4:7]
    _UK_MFMA_    "  %[v_acc_2], acc[16:17], v[64:65], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[18:19], v[66:67], %[v_acc_2]                    \n"
    "  buffer_load_dwordx4  acc[144:147], %[v_os_b1], s[20:23], 0 offen offset:0    \n" // B1: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_2], acc[20:21], v[68:69], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[22:23], v[70:71], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[24:25], v[72:73], %[v_acc_2]                    \n"
    _UK_MFMA_    "  %[v_acc_2], acc[26:27], v[74:75], %[v_acc_2]                    \n"
    "  buffer_load_dwordx4  acc[148:151], %[v_os_b1], s[20:23], 0 offen offset:1024 \n" // B1: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_2], acc[28:29], v[76:77], %[v_acc_2]                    \n"
    "  ds_read_b128  v[96:99],   %[v_os_slda], offset:1*%[smem_sz] + 0              \n" // A1: lds->vgpr
    _UK_MFMA_    "  %[v_acc_2], acc[30:31], v[78:79], %[v_acc_2]                    \n"
    "  ds_read_b128  v[100:103], %[v_os_slda], offset:1*%[smem_sz] + 64             \n" // A1: lds->vgpr

    // A0[4:7] * B0[4:7]
    _UK_MFMA_    "  %[v_acc_3], acc[16:17], v[80:81], %[v_acc_3]                    \n"
    "  buffer_load_dwordx4  acc[152:155], %[v_os_b1], s[20:23], 0 offen offset:2048 \n" // B1: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_3], acc[18:19], v[82:83], %[v_acc_3]                    \n"
    "  ds_read_b128  v[104:107], %[v_os_slda], offset:1*%[smem_sz] + 128            \n" // A1: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[20:21], v[84:85], %[v_acc_3]                    \n"
    "  ds_read_b128  v[108:111], %[v_os_slda], offset:1*%[smem_sz] + 192            \n" // A1: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[22:23], v[86:87], %[v_acc_3]                    \n"
    "  buffer_load_dwordx4  acc[156:159], %[v_os_b1], s[20:23], 0 offen offset:3072 \n" // B1: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_3], acc[24:25], v[88:89], %[v_acc_3]                    \n"
    "  ds_read_b128  v[112:115], %[v_os_slda], offset:1*%[smem_sz] + 4352           \n" // A1: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[26:27], v[90:91], %[v_acc_3]                    \n"
    "  ds_read_b128  v[116:119], %[v_os_slda], offset:1*%[smem_sz] + 4416           \n" // A1: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[28:29], v[92:93], %[v_acc_3]                    \n"
    "  ds_read_b128  v[120:123], %[v_os_slda], offset:1*%[smem_sz] + 4480           \n" // A1: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[30:31], v[94:95], %[v_acc_3]                    \n"
    "  ds_read_b128  v[124:127], %[v_os_slda], offset:1*%[smem_sz] + 4544           \n" // A1: lds->vgpr
#endif
    //////////////////////////////////////////////////////////////////////
    "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1                    \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 0                                    \n"
    "  s_cbranch_scc0 L_end%=                                           \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond     \n"
    "  s_cselect_b32 s86, %[s_tile_os_a], 0                             \n"
    "  s_add_u32     s16, s86, s16                                      \n"
    "  s_addc_u32    s17, 0, s17                                        \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond     \n"
    "  s_cselect_b32 s86, %[s_tile_os_b], 0                             \n"
    "  s_add_u32     s20, s86, s20                                      \n"
    "  s_addc_u32    s21, 0, s21                                        \n"
    //////////////////////////////////////////////////////////////////////
#pragma endregion

#pragma region LOOP_2
    // A1: vmem->lds
    // A0: lds->vgppr
    // B0: vmem->acc
    // mfma: B1 * A1
#if 0
    _FF_SYNC_
    // A1: vmem->lds
    "  s_add_u32           m0, %[s_m0_init], 1*%[smem_sz]                           \n" // set to A2 lds
    "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    "  s_add_u32           m0, %[s_size_per_issue], m0                              \n"
    "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds                        \n" // A2: vmem->lds  (vmcnt)
    _FF_SYNC_

    // A0: lds->vgpr
    "  ds_read_b128  v[64:67],  %[v_os_slda], offset:0*%[smem_sz] + 0               \n" // A0: lds->vgpr
    "  ds_read_b128  v[68:71],  %[v_os_slda], offset:0*%[smem_sz] + 64              \n" // A0: lds->vgpr
    "  ds_read_b128  v[72:75],  %[v_os_slda], offset:0*%[smem_sz] + 128             \n" // A0: lds->vgpr
    "  ds_read_b128  v[76:79],  %[v_os_slda], offset:0*%[smem_sz] + 192             \n" // A0: lds->vgpr
    "  ds_read_b128  v[80:83],  %[v_os_slda], offset:0*%[smem_sz] + 4352            \n" // A0: lds->vgpr
    "  ds_read_b128  v[84:87],  %[v_os_slda], offset:0*%[smem_sz] + 4416            \n" // A0: lds->vgpr
    "  ds_read_b128  v[88:91],  %[v_os_slda], offset:0*%[smem_sz] + 4480            \n" // A0: lds->vgpr
    "  ds_read_b128  v[92:95],  %[v_os_slda], offset:0*%[smem_sz] + 4544            \n" // A0: lds->vgpr
    _FF_SYNC_
    
    // B0: vmem->acc
    "  buffer_load_dwordx4  acc[0:3],   %[v_os_b0], s[20:23], 0 offen offset:0      \n" // B0: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[4:7],   %[v_os_b0], s[20:23], 0 offen offset:1024   \n" // B0: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[8:11],  %[v_os_b0], s[20:23], 0 offen offset:2048   \n" // B0: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[20:23], 0 offen offset:3072   \n" // B0: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[20:23], 0 offen offset:0      \n" // B0: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[20:23], 0 offen offset:1024   \n" // B0: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[20:23], 0 offen offset:2048   \n" // B0: vmem->acc  (vmcnt)
    "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[20:23], 0 offen offset:3072   \n" // B0: vmem->acc  (vmcnt)
    _FF_SYNC_
        
    // B1[0:3] * A1[0:3]
    _UK_MFMA_    "  %[v_acc_0], acc[128:129], v[96:97],   %[v_acc_0]                \n"
    _UK_MFMA_    "  %[v_acc_0], acc[130:131], v[98:99],   %[v_acc_0]                \n"
    _UK_MFMA_    "  %[v_acc_0], acc[132:133], v[100:101], %[v_acc_0]                \n"
    _UK_MFMA_    "  %[v_acc_0], acc[134:135], v[102:103], %[v_acc_0]                \n"
    _UK_MFMA_    "  %[v_acc_0], acc[136:137], v[104:105], %[v_acc_0]                \n"
    _UK_MFMA_    "  %[v_acc_0], acc[138:139], v[106:107], %[v_acc_0]                \n"
    _UK_MFMA_    "  %[v_acc_0], acc[140:141], v[108:109], %[v_acc_0]                \n"
    _UK_MFMA_    "  %[v_acc_0], acc[142:143], v[110:111], %[v_acc_0]                \n"
    // B1[0:3] * A1[4:7]
    _UK_MFMA_    "  %[v_acc_1], acc[128:129], v[112:113], %[v_acc_1]                \n"
    _UK_MFMA_    "  %[v_acc_1], acc[130:131], v[114:115], %[v_acc_1]                \n"
    _UK_MFMA_    "  %[v_acc_1], acc[132:133], v[116:117], %[v_acc_1]                \n"
    _UK_MFMA_    "  %[v_acc_1], acc[134:135], v[118:119], %[v_acc_1]                \n"
    _UK_MFMA_    "  %[v_acc_1], acc[136:137], v[120:121], %[v_acc_1]                \n"
    _UK_MFMA_    "  %[v_acc_1], acc[138:139], v[122:123], %[v_acc_1]                \n"
    _UK_MFMA_    "  %[v_acc_1], acc[140:141], v[124:125], %[v_acc_1]                \n"
    _UK_MFMA_    "  %[v_acc_1], acc[142:143], v[126:127], %[v_acc_1]                \n"    
    // B1[4:7] * A1[0:3]
    _UK_MFMA_    "  %[v_acc_2], acc[144:145], v[96:97],   %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[146:147], v[98:99],   %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[148:149], v[100:101], %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[150:151], v[102:103], %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[152:153], v[104:105], %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[154:155], v[106:107], %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[156:157], v[108:109], %[v_acc_2]                \n" 
    _UK_MFMA_    "  %[v_acc_2], acc[158:159], v[110:111], %[v_acc_2]                \n"
    // B1[4:7] * A1[4:7]
    _UK_MFMA_    "  %[v_acc_3], acc[144:145], v[112:113], %[v_acc_3]                \n"
    _UK_MFMA_    "  %[v_acc_3], acc[146:147], v[114:115], %[v_acc_3]                \n"
    _UK_MFMA_    "  %[v_acc_3], acc[148:149], v[116:117], %[v_acc_3]                \n"
    _UK_MFMA_    "  %[v_acc_3], acc[150:151], v[118:119], %[v_acc_3]                \n"
    _UK_MFMA_    "  %[v_acc_3], acc[152:153], v[120:121], %[v_acc_3]                \n"
    _UK_MFMA_    "  %[v_acc_3], acc[154:155], v[122:123], %[v_acc_3]                \n"
    _UK_MFMA_    "  %[v_acc_3], acc[156:157], v[124:125], %[v_acc_3]                \n"
    _UK_MFMA_    "  %[v_acc_3], acc[158:159], v[126:127], %[v_acc_3]                \n"
    _FF_SYNC_
#else
    "  s_waitcnt     vmcnt(16)                                                      \n" // wait A0 vmem->lds
    "  s_barrier                                                                    \n"

    "  s_waitcnt     vmcnt(4)                                                       \n" // wait B0[0:3]
    //"  s_waitcnt     lgkmcnt(4)                                                   \n" // wait A0[0:7]
    "  s_waitcnt     lgkmcnt(0)                                                     \n" // wait A1 lds->vgpr

    "  s_add_u32 m0, %[s_m0_init], 1*%[smem_sz]                                     \n" // set to A1 lds

    // A1[0:3] * B1[0:3]
    "  buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[20:23], 0 offen offset:0        \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[128:129], v[96:97], %[v_acc_0]                  \n" 
    "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[130:131], v[98:99], %[v_acc_0]                  \n"
    "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[132:133], v[100:101], %[v_acc_0]                \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[134:135], v[102:103], %[v_acc_0]                \n"
    "  buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[20:23], 0 offen offset:1024     \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[136:137], v[104:105], %[v_acc_0]                \n"
    "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[138:139], v[106:107], %[v_acc_0]                \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_0], acc[140:141], v[108:109], %[v_acc_0]                \n"
    "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_0], acc[142:143], v[110:111], %[v_acc_0]                \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"

    //"  s_waitcnt     lgkmcnt(0)                                                   \n" // wait A1[4:7]

    // A1[4:7] * B1[0:3]
    "  buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[20:23], 0 offen offset:2048    \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[128:129], v[112:113], %[v_acc_1]                \n"
    "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[130:131], v[114:115], %[v_acc_1]                \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[132:133], v[116:117], %[v_acc_1]                \n"
    "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[20:23], 0 offen offset:3072   \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[134:135], v[118:119], %[v_acc_1]                \n"
    "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[136:137], v[120:121], %[v_acc_1]                \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[138:139], v[122:123], %[v_acc_1]                \n"
    "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)
    _UK_MFMA_    "  %[v_acc_1], acc[140:141], v[124:125], %[v_acc_1]                \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                                    \n"
    _UK_MFMA_    "  %[v_acc_1], acc[142:143], v[126:127], %[v_acc_1]                \n"
    "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds                        \n" // A1: vmem->lds  (vmcnt)

    "  s_waitcnt     vmcnt(12)                                                      \n" // wait B1[4:7]

    // A1[0:3] * B1[4:7]
    _UK_MFMA_    "  %[v_acc_2], acc[144:145], v[96:97], %[v_acc_2]                  \n"
    _UK_MFMA_    "  %[v_acc_2], acc[146:147], v[98:99], %[v_acc_2]                  \n"
    "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[20:23], 0 offen offset:0      \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_2], acc[148:149], v[100:101], %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[150:151], v[102:103], %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[152:153], v[104:105], %[v_acc_2]                \n"
    _UK_MFMA_    "  %[v_acc_2], acc[154:155], v[106:107], %[v_acc_2]                \n"
    "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[20:23], 0 offen offset:1024   \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_2], acc[156:157], v[108:109], %[v_acc_2]                \n" 
    "  ds_read_b128  v[64:67], %[v_os_slda] offset:0*%[smem_sz] + 0                 \n" // A0: lds->vgpr
    _UK_MFMA_    "  %[v_acc_2], acc[158:159], v[110:111], %[v_acc_2]                \n"
    "  ds_read_b128  v[68:71], %[v_os_slda] offset:0*%[smem_sz] + 64                \n" // A0: lds->vgpr

    // A1[4:7] * B1[4:7]
    _UK_MFMA_    "  %[v_acc_3], acc[144:145], v[112:113], %[v_acc_3]                \n"
    "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[20:23], 0 offen offset:2048   \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_3], acc[146:147], v[114:115], %[v_acc_3]                \n"
    "  ds_read_b128  v[72:75], %[v_os_slda] offset:0*%[smem_sz] + 128               \n" // A0: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[148:149], v[116:117], %[v_acc_3]                \n"
    "  ds_read_b128  v[76:79], %[v_os_slda] offset:0*%[smem_sz] + 192               \n" // A0: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[150:151], v[118:119], %[v_acc_3]                \n"
    "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[20:23], 0 offen offset:3072   \n" // B0: vmem->acc  (vmcnt)
    _UK_MFMA_    "  %[v_acc_3], acc[152:153], v[120:121], %[v_acc_3]                \n"
    "  ds_read_b128  v[80:83], %[v_os_slda] offset:0*%[smem_sz] + 4352              \n" // A0: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[154:155], v[122:123], %[v_acc_3]                \n"
    "  ds_read_b128  v[84:87], %[v_os_slda] offset:0*%[smem_sz] + 4416              \n" // A0: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[156:157], v[124:125], %[v_acc_3]                \n"
    "  ds_read_b128  v[88:91], %[v_os_slda] offset:0*%[smem_sz] + 4480              \n" // A0: lds->vgpr
    _UK_MFMA_    "  %[v_acc_3], acc[158:159], v[126:127], %[v_acc_3]                \n"
    "  ds_read_b128  v[92:95], %[v_os_slda] offset:0*%[smem_sz] + 4544              \n" // A0: lds->vgpr
#endif   
    //////////////////////////////////////////////////////////////////////
    "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1                    \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 0                                    \n"
    "  s_cbranch_scc0 L_end%=                                           \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond     \n"
    "  s_cselect_b32 s86, %[s_tile_os_a], 0                             \n"
    "  s_add_u32     s16, s86, s16                                      \n"
    "  s_addc_u32    s17, 0, s17                                        \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond     \n"
    "  s_cselect_b32 s86, %[s_tile_os_b], 0                             \n"
    "  s_add_u32     s20, s86, s20                                      \n"
    "  s_addc_u32    s21, 0, s21                                        \n"
    "  s_branch     L_start%=                                           \n"
    //////////////////////////////////////////////////////////////////////
#pragma endregion

    //////////////////////////////////////////////////////////////////////
"L_end%=:                                                               \n"
    "  s_nop 2                                                          \n"

#endif

#undef _UK_MFMA_
#undef _FF_SYNC_
