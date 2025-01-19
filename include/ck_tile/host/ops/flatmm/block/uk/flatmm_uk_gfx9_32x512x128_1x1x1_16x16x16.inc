#ifndef CK_TILE_FLATMM_UK_MFMA
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_BF16
#endif

#if CK_TILE_FLATMM_UK_MFMA == CK_TILE_FLATMM_UK_MFMA_BF16
#define _UK_MFMA_ "v_mfma_f32_16x16x16_bf16"
#elif CK_TILE_FLATMM_UK_MFMA == CK_TILE_FLATMM_UK_MFMA_FP16
#define _UK_MFMA_ "v_mfma_f32_16x16x16_f16"
#endif

"s_mov_b32 s16,    %[s_res_a0] \n"
    "s_mov_b32 s17,    %[s_res_a1] \n"
    "s_mov_b32 s18,    %[s_res_a2] \n"
    "s_mov_b32 s19,    %[s_res_a3] \n"
    "s_mov_b32 s20,    %[s_res_b0] \n"
    "s_mov_b32 s21,    %[s_res_b1] \n"
    "s_mov_b32 s22,    %[s_res_b2] \n"
    "s_mov_b32 s23,    %[s_res_b3] \n"
    // "s_nop  4\n"
    "; -- prefetch A0\n"
    "s_add_u32     m0, 0, %[s_m0_init]                        \n"
    "buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                \n"
    "buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                \n"
    "buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                \n"
    "buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                \n"
    "buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                \n"
    "buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                \n"
    "buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                \n"
    "buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds    \n"
    "s_add_u32 m0, %[smem_sz], %[s_m0_init]                       \n"
    "s_cmp_gt_i32  %[s_loop_cnt] 1             ; move a with cond \n"
    "s_cselect_b32 s86, %[s_tile_os_a], 0      ; move a with cond  \n"
    "s_add_u32     s16, s86, s16               ; move a with cond \n"
    "s_addc_u32    s17, 0, s17                 ; move a with cond \n"
    "; -- prefetch A1\n"
    "buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
    "buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
    "buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
    "buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
    "buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
    "buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
    "buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds    \n"
    "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
    "buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds    \n"
    "s_add_u32 m0, 0, %[s_m0_init]                                \n"
    "s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond \n"
    "s_cselect_b32 s86, %[s_tile_os_a], 0      ; move a with cond  \n"
    "s_add_u32     s16, s86, s16               ; move a with cond \n"
    "s_addc_u32    s17, 0, s17                 ; move a with cond \n"
    "; -- prefetch B0\n"
    "buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[20:23], 0 offen \n"
    "buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[20:23], 0 offen offset:1024  \n"
    "buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[20:23], 0 offen offset:2048  \n"
    "buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[20:23], 0 offen offset:3072  \n"
    "buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[20:23], 0 offen  \n"
    "buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[20:23], 0 offen offset:1024  \n"
    "buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[20:23], 0 offen offset:2048  \n"
    "buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[20:23], 0 offen offset:3072  \n"
    "buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[20:23], 0 offen  \n"
    "buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[20:23], 0 offen offset:1024  \n"
    "buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[20:23], 0 offen offset:2048  \n"
    "buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[20:23], 0 offen offset:3072  \n"
    "buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[20:23], 0 offen  \n"
    "buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[20:23], 0 offen offset:1024    \n"
    "buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[20:23], 0 offen offset:2048    \n"
    "buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[20:23], 0 offen offset:3072    \n"
    "buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[20:23], 0 offen                \n"
    "buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[20:23], 0 offen offset:1024    \n"
    "buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[20:23], 0 offen offset:2048    \n"
    "buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[20:23], 0 offen offset:3072    \n"
    "buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[20:23], 0 offen                \n"
    "buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[20:23], 0 offen offset:1024    \n"
    "buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[20:23], 0 offen offset:2048    \n"
    "buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[20:23], 0 offen offset:3072    \n"
    "buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[20:23], 0 offen                \n"
    "buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[20:23], 0 offen offset:1024  \n"
    "buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[20:23], 0 offen offset:2048  \n"
    "buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[20:23], 0 offen offset:3072  \n"
    "buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[20:23], 0 offen              \n"
    "buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[20:23], 0 offen offset:1024  \n"
    "buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[20:23], 0 offen offset:2048  \n"
    "buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[20:23], 0 offen offset:3072  \n"
    "s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
    "s_cselect_b32 s86, %[s_tile_os_b], 0      ; move b with cond \n"
    "s_add_u32     s20, s86, s20               ; move b with cond \n"
    "s_addc_u32    s21, 0, s21                 ; move b with cond \n"
    "s_waitcnt     vmcnt(40)                        \n"
    "s_barrier                                      \n"
    "ds_read_b128  v[64:67], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_0]\n" // 1024: N stride, 64
                                                                               // K stride
    "ds_read_b128  v[68:71], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_1]\n"
    "ds_read_b128  v[72:75], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_2]\n"
    "ds_read_b128  v[76:79], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_3]\n"
    "ds_read_b128  v[80:83], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_4]\n"
    "ds_read_b128  v[84:87], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_5]\n"
    "ds_read_b128  v[88:91], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_6]\n"
    "ds_read_b128  v[92:95], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_7]\n"
    "L_start%=:                                                         \n"
    "  s_waitcnt     vmcnt(24) & lgkmcnt(0)                             \n"
    "  s_barrier                                                        \n" _UK_MFMA_
    "  %[v_acc_0], acc[0:1], v[64:65], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[2:3], v[66:67], %[v_acc_0] \n"
    "  buffer_load_dwordx4  acc[128:131], %[v_os_b0], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_0], acc[4:5], v[68:69], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[6:7], v[70:71], %[v_acc_0] \n"
    "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_0], acc[8:9], v[72:73], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[10:11], v[74:75], %[v_acc_0] \n"
    "  buffer_load_dwordx4  acc[132:135], %[v_os_b0], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_0], acc[12:13], v[76:77], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[14:15], v[78:79], %[v_acc_0] \n"
    "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_1], acc[0:1], v[80:81], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[2:3], v[82:83], %[v_acc_1] \n"
    "  buffer_load_dwordx4  acc[136:139], %[v_os_b0], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_1], acc[4:5], v[84:85], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[6:7], v[86:87], %[v_acc_1] \n"
    "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_1], acc[8:9], v[88:89], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[10:11], v[90:91], %[v_acc_1] \n"
    "  buffer_load_dwordx4  acc[140:143], %[v_os_b0], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_1], acc[12:13], v[92:93], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[14:15], v[94:95], %[v_acc_1] \n"
    "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_2], acc[16:17], v[64:65], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[18:19], v[66:67], %[v_acc_2] \n"
    "  buffer_load_dwordx4  acc[144:147], %[v_os_b1], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_2], acc[20:21], v[68:69], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[22:23], v[70:71], %[v_acc_2] \n"
    "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_2], acc[24:25], v[72:73], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[26:27], v[74:75], %[v_acc_2] \n"
    "  buffer_load_dwordx4  acc[148:151], %[v_os_b1], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_2], acc[28:29], v[76:77], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[30:31], v[78:79], %[v_acc_2] \n"
    "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_3], acc[16:17], v[80:81], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[18:19], v[82:83], %[v_acc_3] \n"
    "  buffer_load_dwordx4  acc[152:155], %[v_os_b1], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_3], acc[20:21], v[84:85], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[22:23], v[86:87], %[v_acc_3] \n"
    "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_3], acc[24:25], v[88:89], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[26:27], v[90:91], %[v_acc_3] \n"
    "  buffer_load_dwordx4  acc[156:159], %[v_os_b1], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_3], acc[28:29], v[92:93], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[30:31], v[94:95], %[v_acc_3] \n"
    "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[smem_sz], %[s_m0_init]                  \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  %[v_acc_4], acc[32:33], v[64:65], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[34:35], v[66:67], %[v_acc_4] \n"
    "  buffer_load_dwordx4  acc[160:163], %[v_os_b2], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_4], acc[36:37], v[68:69], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[38:39], v[70:71], %[v_acc_4] \n"
    "  ds_read_b128  v[96:99], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_0]                "
    "\n" _UK_MFMA_ "  %[v_acc_4], acc[40:41], v[72:73], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[42:43], v[74:75], %[v_acc_4] \n"
    "  buffer_load_dwordx4  acc[164:167], %[v_os_b2], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_4], acc[44:45], v[76:77], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[46:47], v[78:79], %[v_acc_4] \n"
    "  ds_read_b128  v[100:103], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_1]              "
    "\n" _UK_MFMA_ "  %[v_acc_5], acc[32:33], v[80:81], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[34:35], v[82:83], %[v_acc_5] \n"
    "  buffer_load_dwordx4  acc[168:171], %[v_os_b2], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_5], acc[36:37], v[84:85], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[38:39], v[86:87], %[v_acc_5] \n"
    "  ds_read_b128  v[104:107], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_2]              "
    "\n" _UK_MFMA_ "  %[v_acc_5], acc[40:41], v[88:89], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[42:43], v[90:91], %[v_acc_5] \n"
    "  buffer_load_dwordx4  acc[172:175], %[v_os_b2], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_5], acc[44:45], v[92:93], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[46:47], v[94:95], %[v_acc_5] \n"
    "  ds_read_b128  v[108:111], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_3]              "
    "\n" _UK_MFMA_ "  %[v_acc_6], acc[48:49], v[64:65], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[50:51], v[66:67], %[v_acc_6] \n"
    "  buffer_load_dwordx4  acc[176:179], %[v_os_b3], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_6], acc[52:53], v[68:69], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[54:55], v[70:71], %[v_acc_6] \n"
    "  ds_read_b128  v[112:115], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_4]              "
    "\n" _UK_MFMA_ "  %[v_acc_6], acc[56:57], v[72:73], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[58:59], v[74:75], %[v_acc_6] \n"
    "  buffer_load_dwordx4  acc[180:183], %[v_os_b3], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_6], acc[60:61], v[76:77], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[62:63], v[78:79], %[v_acc_6] \n"
    "  ds_read_b128  v[116:119], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_5]              "
    "\n" _UK_MFMA_ "  %[v_acc_7], acc[48:49], v[80:81], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[50:51], v[82:83], %[v_acc_7] \n"
    "  buffer_load_dwordx4  acc[184:187], %[v_os_b3], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_7], acc[52:53], v[84:85], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[54:55], v[86:87], %[v_acc_7] \n"
    "  ds_read_b128  v[120:123], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_6]              "
    "\n" _UK_MFMA_ "  %[v_acc_7], acc[56:57], v[88:89], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[58:59], v[90:91], %[v_acc_7] \n"
    "  buffer_load_dwordx4  acc[188:191], %[v_os_b3], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_7], acc[60:61], v[92:93], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[62:63], v[94:95], %[v_acc_7] \n"
    "  ds_read_b128  v[124:127], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_7]              \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  %[v_acc_8], acc[64:65], v[64:65], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[66:67], v[66:67], %[v_acc_8] \n"
    "  buffer_load_dwordx4  acc[192:195], %[v_os_b4], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_8], acc[68:69], v[68:69], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[70:71], v[70:71], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[72:73], v[72:73], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[74:75], v[74:75], %[v_acc_8] \n"
    "  buffer_load_dwordx4  acc[196:199], %[v_os_b4], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_8], acc[76:77], v[76:77], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[78:79], v[78:79], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_9], acc[64:65], v[80:81], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[66:67], v[82:83], %[v_acc_9] \n"
    "  buffer_load_dwordx4  acc[200:203], %[v_os_b4], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_9], acc[68:69], v[84:85], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[70:71], v[86:87], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[72:73], v[88:89], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[74:75], v[90:91], %[v_acc_9] \n"
    "  buffer_load_dwordx4  acc[204:207], %[v_os_b4], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_9], acc[76:77], v[92:93], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[78:79], v[94:95], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_10], acc[80:81], v[64:65], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[82:83], v[66:67], %[v_acc_10] \n"
    "  buffer_load_dwordx4  acc[208:211], %[v_os_b5], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_10], acc[84:85], v[68:69], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[86:87], v[70:71], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[88:89], v[72:73], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[90:91], v[74:75], %[v_acc_10] \n"
    "  buffer_load_dwordx4  acc[212:215], %[v_os_b5], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_10], acc[92:93], v[76:77], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[94:95], v[78:79], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_11], acc[80:81], v[80:81], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[82:83], v[82:83], %[v_acc_11] \n"
    "  buffer_load_dwordx4  acc[216:219], %[v_os_b5], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_11], acc[84:85], v[84:85], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[86:87], v[86:87], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[88:89], v[88:89], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[90:91], v[90:91], %[v_acc_11] \n"
    "  buffer_load_dwordx4  acc[220:223], %[v_os_b5], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_11], acc[92:93], v[92:93], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[94:95], v[94:95], %[v_acc_11] \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  %[v_acc_12], acc[96:97], v[64:65], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[98:99], v[66:67], %[v_acc_12] \n"
    "  buffer_load_dwordx4  acc[224:227], %[v_os_b6], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_12], acc[100:101], v[68:69], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[102:103], v[70:71], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[104:105], v[72:73], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[106:107], v[74:75], %[v_acc_12] \n"
    "  buffer_load_dwordx4  acc[228:231], %[v_os_b6], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_12], acc[108:109], v[76:77], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[110:111], v[78:79], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_13], acc[96:97], v[80:81], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[98:99], v[82:83], %[v_acc_13] \n"
    "  buffer_load_dwordx4  acc[232:235], %[v_os_b6], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_13], acc[100:101], v[84:85], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[102:103], v[86:87], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[104:105], v[88:89], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[106:107], v[90:91], %[v_acc_13] \n"
    "  buffer_load_dwordx4  acc[236:239], %[v_os_b6], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_13], acc[108:109], v[92:93], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[110:111], v[94:95], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_14], acc[112:113], v[64:65], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[114:115], v[66:67], %[v_acc_14] \n"
    "  buffer_load_dwordx4  acc[240:243], %[v_os_b7], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_14], acc[116:117], v[68:69], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[118:119], v[70:71], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[120:121], v[72:73], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[122:123], v[74:75], %[v_acc_14] \n"
    "  buffer_load_dwordx4  acc[244:247], %[v_os_b7], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_14], acc[124:125], v[76:77], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[126:127], v[78:79], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_15], acc[112:113], v[80:81], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[114:115], v[82:83], %[v_acc_15] \n"
    "  buffer_load_dwordx4  acc[248:251], %[v_os_b7], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_15], acc[116:117], v[84:85], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[118:119], v[86:87], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[120:121], v[88:89], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[122:123], v[90:91], %[v_acc_15] \n"
    "  buffer_load_dwordx4  acc[252:255], %[v_os_b7], s[20:23], 0 offen offset:3072\n" _UK_MFMA_
    "  %[v_acc_15], acc[124:125], v[92:93], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[126:127], v[94:95], %[v_acc_15] \n"
    "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1                \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
    "  s_cbranch_scc0 L_end%=                                       \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond \n"
    "  s_cselect_b32 s86, %[s_tile_os_a], 0                          \n"
    "  s_add_u32     s16, s86, s16                                  \n"
    "  s_addc_u32    s17, 0, s17                                    \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
    "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
    "  s_add_u32     s20, s86, s20                                  \n"
    "  s_addc_u32    s21, 0, s21                                    \n"
    "  ;------------------------------------------                  \n"
    "  s_waitcnt     vmcnt(24) & lgkmcnt(0)                  \n"
    "  s_barrier                                             \n" _UK_MFMA_
    "  %[v_acc_0], acc[128:129], v[96:97], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[130:131], v[98:99], %[v_acc_0] \n"
    "  buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_0], acc[132:133], v[100:101], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[134:135], v[102:103], %[v_acc_0] \n"
    "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_0], acc[136:137], v[104:105], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[138:139], v[106:107], %[v_acc_0] \n"
    "  buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_0], acc[140:141], v[108:109], %[v_acc_0] \n" _UK_MFMA_
    "  %[v_acc_0], acc[142:143], v[110:111], %[v_acc_0] \n"
    "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_1], acc[128:129], v[112:113], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[130:131], v[114:115], %[v_acc_1] \n"
    "  buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_1], acc[132:133], v[116:117], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[134:135], v[118:119], %[v_acc_1] \n"
    "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_1], acc[136:137], v[120:121], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[138:139], v[122:123], %[v_acc_1] \n"
    "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_1], acc[140:141], v[124:125], %[v_acc_1] \n" _UK_MFMA_
    "  %[v_acc_1], acc[142:143], v[126:127], %[v_acc_1] \n"
    "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_2], acc[144:145], v[96:97], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[146:147], v[98:99], %[v_acc_2] \n"
    "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_2], acc[148:149], v[100:101], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[150:151], v[102:103], %[v_acc_2] \n"
    "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_2], acc[152:153], v[104:105], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[154:155], v[106:107], %[v_acc_2] \n"
    "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_2], acc[156:157], v[108:109], %[v_acc_2] \n" _UK_MFMA_
    "  %[v_acc_2], acc[158:159], v[110:111], %[v_acc_2] \n"
    "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_3], acc[144:145], v[112:113], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[146:147], v[114:115], %[v_acc_3] \n"
    "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_3], acc[148:149], v[116:117], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[150:151], v[118:119], %[v_acc_3] \n"
    "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, %[s_size_per_issue], m0                  \n" _UK_MFMA_
    "  %[v_acc_3], acc[152:153], v[120:121], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[154:155], v[122:123], %[v_acc_3] \n"
    "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_3], acc[156:157], v[124:125], %[v_acc_3] \n" _UK_MFMA_
    "  %[v_acc_3], acc[158:159], v[126:127], %[v_acc_3] \n"
    "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds     \n"
    "  s_add_u32     m0, 0, %[s_m0_init]                  \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  %[v_acc_4], acc[160:161], v[96:97], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[162:163], v[98:99], %[v_acc_4] \n"
    "  buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_4], acc[164:165], v[100:101], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[166:167], v[102:103], %[v_acc_4] \n"
    "  ds_read_b128  v[64:67], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_0]  \n" _UK_MFMA_
    "  %[v_acc_4], acc[168:169], v[104:105], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[170:171], v[106:107], %[v_acc_4] \n"
    "  buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_4], acc[172:173], v[108:109], %[v_acc_4] \n" _UK_MFMA_
    "  %[v_acc_4], acc[174:175], v[110:111], %[v_acc_4] \n"
    "  ds_read_b128  v[68:71], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_1]  \n" _UK_MFMA_
    "  %[v_acc_5], acc[160:161], v[112:113], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[162:163], v[114:115], %[v_acc_5] \n"
    "  buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_5], acc[164:165], v[116:117], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[166:167], v[118:119], %[v_acc_5] \n"
    "  ds_read_b128  v[72:75], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_2]                 "
    "\n" _UK_MFMA_ "  %[v_acc_5], acc[168:169], v[120:121], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[170:171], v[122:123], %[v_acc_5] \n"
    "  buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_5], acc[172:173], v[124:125], %[v_acc_5] \n" _UK_MFMA_
    "  %[v_acc_5], acc[174:175], v[126:127], %[v_acc_5] \n"
    "  ds_read_b128  v[76:79], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_3]                "
    "\n" _UK_MFMA_ "  %[v_acc_6], acc[176:177], v[96:97], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[178:179], v[98:99], %[v_acc_6] \n"
    "  buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_6], acc[180:181], v[100:101], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[182:183], v[102:103], %[v_acc_6] \n"
    "  ds_read_b128  v[80:83], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_4]               "
    "\n" _UK_MFMA_ "  %[v_acc_6], acc[184:185], v[104:105], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[186:187], v[106:107], %[v_acc_6] \n"
    "  buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_6], acc[188:189], v[108:109], %[v_acc_6] \n" _UK_MFMA_
    "  %[v_acc_6], acc[190:191], v[110:111], %[v_acc_6] \n"
    "  ds_read_b128  v[84:87], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_5]            "
    "\n" _UK_MFMA_ "  %[v_acc_7], acc[176:177], v[112:113], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[178:179], v[114:115], %[v_acc_7] \n"
    "  buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_7], acc[180:181], v[116:117], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[182:183], v[118:119], %[v_acc_7] \n"
    "  ds_read_b128  v[88:91], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_6]                "
    "\n" _UK_MFMA_ "  %[v_acc_7], acc[184:185], v[120:121], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[186:187], v[122:123], %[v_acc_7] \n"
    "  buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_7], acc[188:189], v[124:125], %[v_acc_7] \n" _UK_MFMA_
    "  %[v_acc_7], acc[190:191], v[126:127], %[v_acc_7] \n"
    "  ds_read_b128  v[92:95], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_7]           \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  %[v_acc_8], acc[192:193], v[96:97], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[194:195], v[98:99], %[v_acc_8] \n"
    "  buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_8], acc[196:197], v[100:101], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[198:199], v[102:103], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[200:201], v[104:105], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[202:203], v[106:107], %[v_acc_8] \n"
    "  buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_8], acc[204:205], v[108:109], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_8], acc[206:207], v[110:111], %[v_acc_8] \n" _UK_MFMA_
    "  %[v_acc_9], acc[192:193], v[112:113], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[194:195], v[114:115], %[v_acc_9] \n"
    "  buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_9], acc[196:197], v[116:117], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[198:199], v[118:119], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[200:201], v[120:121], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[202:203], v[122:123], %[v_acc_9] \n"
    "  buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_9], acc[204:205], v[124:125], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_9], acc[206:207], v[126:127], %[v_acc_9] \n" _UK_MFMA_
    "  %[v_acc_10], acc[208:209], v[96:97], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[210:211], v[98:99], %[v_acc_10] \n"
    "  buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_10], acc[212:213], v[100:101], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[214:215], v[102:103], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[216:217], v[104:105], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[218:219], v[106:107], %[v_acc_10] \n"
    "  buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_10], acc[220:221], v[108:109], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_10], acc[222:223], v[110:111], %[v_acc_10] \n" _UK_MFMA_
    "  %[v_acc_11], acc[208:209], v[112:113], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[210:211], v[114:115], %[v_acc_11] \n"
    "  buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_11], acc[212:213], v[116:117], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[214:215], v[118:119], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[216:217], v[120:121], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[218:219], v[122:123], %[v_acc_11] \n"
    "  buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_11], acc[220:221], v[124:125], %[v_acc_11] \n" _UK_MFMA_
    "  %[v_acc_11], acc[222:223], v[126:127], %[v_acc_11] \n"
    "  s_waitcnt     vmcnt(32)                               \n" _UK_MFMA_
    "  %[v_acc_12], acc[224:225], v[96:97], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[226:227], v[98:99], %[v_acc_12] \n"
    "  buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_12], acc[228:229], v[100:101], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[230:231], v[102:103], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[232:233], v[104:105], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[234:235], v[106:107], %[v_acc_12] \n"
    "  buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_12], acc[236:237], v[108:109], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_12], acc[238:239], v[110:111], %[v_acc_12] \n" _UK_MFMA_
    "  %[v_acc_13], acc[224:225], v[112:113], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[226:227], v[114:115], %[v_acc_13] \n"
    "  buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_13], acc[228:229], v[116:117], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[230:231], v[118:119], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[232:233], v[120:121], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[234:235], v[122:123], %[v_acc_13] \n"
    "  buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_13], acc[236:237], v[124:125], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_13], acc[238:239], v[126:127], %[v_acc_13] \n" _UK_MFMA_
    "  %[v_acc_14], acc[240:241], v[96:97], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[242:243], v[98:99], %[v_acc_14] \n"
    "  buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[20:23], 0 offen \n" _UK_MFMA_
    "  %[v_acc_14], acc[244:245], v[100:101], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[246:247], v[102:103], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[248:249], v[104:105], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[250:251], v[106:107], %[v_acc_14] \n"
    "  buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[20:23], 0 offen offset:1024 \n" _UK_MFMA_
    "  %[v_acc_14], acc[252:253], v[108:109], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_14], acc[254:255], v[110:111], %[v_acc_14] \n" _UK_MFMA_
    "  %[v_acc_15], acc[240:241], v[112:113], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[242:243], v[114:115], %[v_acc_15] \n"
    "  buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[20:23], 0 offen offset:2048 \n" _UK_MFMA_
    "  %[v_acc_15], acc[244:245], v[116:117], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[246:247], v[118:119], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[248:249], v[120:121], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[250:251], v[122:123], %[v_acc_15] \n"
    "  buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[20:23], 0 offen offset:3072 \n" _UK_MFMA_
    "  %[v_acc_15], acc[252:253], v[124:125], %[v_acc_15] \n" _UK_MFMA_
    "  %[v_acc_15], acc[254:255], v[126:127], %[v_acc_15] \n"
    "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1                \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
    "  s_cbranch_scc0 L_end%=                                       \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond \n"
    "  s_cselect_b32 s86, %[s_tile_os_a], 0                          \n"
    "  s_add_u32     s16, s86, s16                                  \n"
    "  s_addc_u32    s17, 0, s17                                    \n"
    "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
    "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
    "  s_add_u32     s20, s86, s20                                  \n"
    "  s_addc_u32    s21, 0, s21                                    \n"
    "  s_branch     L_start%=                                       \n"
    "L_end%=:                                                       \n"
    "  s_nop 2                                                      \n"

#undef _UK_MFMA_
