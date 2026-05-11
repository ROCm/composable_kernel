
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx942
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z33test_intra_lane_transpose_patternPKDF16_Pf ; -- Begin function _Z33test_intra_lane_transpose_patternPKDF16_Pf
	.globl	_Z33test_intra_lane_transpose_patternPKDF16_Pf
	.p2align	8
	.type	_Z33test_intra_lane_transpose_patternPKDF16_Pf,@function
_Z33test_intra_lane_transpose_patternPKDF16_Pf: ; @_Z33test_intra_lane_transpose_patternPKDF16_Pf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x1c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s8, s2, 0xffff
	v_cvt_f32_u32_e32 v2, s8
	s_movk_i32 s2, 0x800
	v_add_u32_e32 v1, s8, v0
	v_mov_b32_e32 v3, s8
	v_rcp_iflag_f32_e32 v2, v2
	v_cmp_gt_u32_e32 vcc, s2, v1
	s_cmp_eq_u32 s8, 1
	s_cselect_b64 s[4:5], -1, 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_addc_co_u32_e64 v3, s[2:3], v0, v3, vcc
	v_max_u32_e32 v4, 0x800, v1
	s_sub_i32 s2, 0, s8
	v_sub_u32_e32 v3, v4, v3
	v_mul_lo_u32 v4, s2, v2
	v_mul_hi_u32 v4, v2, v4
	v_add_u32_e32 v2, v2, v4
	v_mul_hi_u32 v2, v3, v2
	v_mul_lo_u32 v4, v2, s8
	v_sub_u32_e32 v3, v3, v4
	v_add_u32_e32 v5, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	v_subrev_u32_e32 v4, s8, v3
	s_nop 0
	v_cndmask_b32_e64 v2, v2, v5, s[2:3]
	v_cndmask_b32_e64 v3, v3, v4, s[2:3]
	v_add_u32_e32 v4, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	s_nop 1
	v_cndmask_b32_e64 v3, v2, v4, s[2:3]
	v_addc_co_u32_e64 v8, s[2:3], 1, v3, vcc
	v_cmp_lt_u32_e64 s[2:3], 3, v8
	s_and_b64 s[6:7], s[2:3], s[4:5]
	s_mov_b64 s[4:5], -1
	v_mov_b32_e32 v2, v0
	s_and_saveexec_b64 s[2:3], s[6:7]
	s_cbranch_execz .LBB0_9
; %bb.1:                                ; %vector.ph
	v_addc_co_u32_e32 v6, vcc, 0, v3, vcc
	v_mad_u64_u32 v[4:5], s[4:5], s8, 3, v[0:1]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, -3, v6
	v_lshl_add_u32 v2, s8, 1, v0
	v_lshrrev_b32_e32 v5, 2, v4
	v_add_u32_e32 v9, 1, v5
	v_cmp_lt_u32_e32 vcc, 11, v4
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b32_e32 v12, 0
	v_mov_b64_e32 v[4:5], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_5
; %bb.2:                                ; %vector.ph.new
	v_mov_b64_e32 v[6:7], v[2:3]
	v_and_b32_e32 v10, 0x7ffffffc, v9
	s_lshl_b32 s9, s8, 3
	v_lshlrev_b32_e32 v11, 1, v0
	s_lshl_b32 s10, s8, 5
	s_lshl_b32 s11, s8, 4
	s_mul_i32 s12, s8, 24
	s_mov_b32 s13, 0
	s_mov_b64 s[6:7], 0
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_3:                                ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v1, v7
	v_cvt_f32_u32_e32 v2, v6
	v_cvt_f32_u32_e32 v3, v5
	v_cvt_f32_u32_e32 v13, v4
	v_add_u32_e32 v12, 4, v4
	v_add_u32_e32 v14, 4, v5
	v_add_u32_e32 v15, 4, v6
	v_add_u32_e32 v16, 4, v7
	v_add_u32_e32 v17, 8, v4
	v_add_u32_e32 v19, 8, v5
	v_add_u32_e32 v20, 8, v6
	v_add_u32_e32 v21, 8, v7
	v_add_u32_e32 v23, 12, v4
	v_add_u32_e32 v24, 12, v5
	v_add_u32_e32 v25, 12, v6
	v_add_u32_e32 v26, 12, v7
	v_cvt_f32_u32_e32 v16, v16
	v_cvt_f32_u32_e32 v15, v15
	v_cvt_f32_u32_e32 v14, v14
	v_cvt_f32_u32_e32 v28, v12
	v_cvt_f32_u32_e32 v21, v21
	v_cvt_f32_u32_e32 v20, v20
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v17, v17
	v_cvt_f32_u32_e32 v26, v26
	v_cvt_f32_u32_e32 v25, v25
	v_cvt_f32_u32_e32 v24, v24
	v_cvt_f32_u32_e32 v23, v23
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v29, v3
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v26, v26
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v23, v23
	v_add_u32_e32 v10, -4, v10
	s_add_i32 s13, s13, 16
	v_cmp_eq_u32_e32 vcc, 0, v10
	v_pack_b32_f16 v3, v2, v1
	v_pack_b32_f16 v2, v13, v29
	v_add_u32_e32 v18, s9, v11
	v_add_u32_e32 v22, s11, v11
	v_add_u32_e32 v27, s12, v11
	v_add_u32_e32 v7, 16, v7
	v_add_u32_e32 v6, 16, v6
	v_add_u32_e32 v5, 16, v5
	v_add_u32_e32 v4, 16, v4
	v_mov_b32_e32 v12, s13
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v11, v[2:3]
	v_pack_b32_f16 v3, v15, v16
	v_pack_b32_f16 v2, v28, v14
	v_add_u32_e32 v11, s10, v11
	v_pack_b32_f16 v15, v20, v21
	v_pack_b32_f16 v14, v17, v19
	v_pack_b32_f16 v17, v25, v26
	v_pack_b32_f16 v16, v23, v24
	ds_write_b64 v18, v[2:3]
	ds_write_b64 v22, v[14:15]
	ds_write_b64 v27, v[16:17]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_3
; %bb.4:                                ; %Flow47
	s_or_b64 exec, exec, s[6:7]
.LBB0_5:                                ; %Flow48
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, 3, v9
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_8
; %bb.6:                                ; %vector.body.epil.preheader
	v_mul_lo_u32 v2, v12, s8
	v_add_lshl_u32 v2, v0, v2, 1
	s_lshl_b32 s9, s8, 3
	s_mov_b64 s[6:7], 0
.LBB0_7:                                ; %vector.body.epil
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v7
	v_cvt_f32_u32_e32 v9, v6
	v_cvt_f32_u32_e32 v10, v5
	v_cvt_f32_u32_e32 v11, v4
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v12, v11
	v_add_u32_e32 v1, -1, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	v_pack_b32_f16 v11, v9, v3
	v_pack_b32_f16 v10, v12, v10
	v_add_u32_e32 v7, 4, v7
	v_add_u32_e32 v6, 4, v6
	v_add_u32_e32 v5, 4, v5
	v_add_u32_e32 v4, 4, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v2, v[10:11]
	v_add_u32_e32 v2, s9, v2
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_7
.LBB0_8:                                ; %Flow46
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB0_9:                                ; %Flow49
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_12
; %bb.10:                               ; %for.body.preheader
	v_lshlrev_b32_e32 v1, 1, v2
	s_lshl_b32 s6, s8, 1
	s_mov_b64 s[4:5], 0
	s_movk_i32 s7, 0x7ff
.LBB0_11:                               ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v2
	v_add_u32_e32 v2, s8, v2
	v_cmp_lt_u32_e32 vcc, s7, v2
	s_or_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v3, v3
	ds_write_b16 v1, v3
	v_add_u32_e32 v1, s6, v1
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB0_11
.LBB0_12:                               ; %Flow50
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:                               ; %for.body10.preheader
	v_mov_b32_e32 v0, 0
	ds_read_u16 v1, v0
	ds_read_u16 v2, v0 offset:64
	ds_read_u16 v3, v0 offset:128
	ds_read_u16 v4, v0 offset:192
	ds_read_u16 v5, v0 offset:256
	ds_read_u16 v6, v0 offset:320
	ds_read_u16 v7, v0 offset:384
	ds_read_u16 v8, v0 offset:448
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v1, v1
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v2, v2
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v3, v3
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v4, v4
	v_add_f32_e32 v1, 0, v1
	v_add_f32_e32 v1, v1, v2
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v2, v5
	v_add_f32_e32 v1, v1, v3
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v3, v6
	v_add_f32_e32 v1, v1, v4
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v4, v7
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v5, v8
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	global_store_dword v0, v1, s[0:1]
.LBB0_14:                               ; %if.end
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z33test_intra_lane_transpose_patternPKDF16_Pf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z33test_intra_lane_transpose_patternPKDF16_Pf, .Lfunc_end0-_Z33test_intra_lane_transpose_patternPKDF16_Pf
                                        ; -- End function
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.num_vgpr, 30
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.num_agpr, 0
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.numbered_sgpr, 14
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.private_seg_size, 0
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.uses_vcc, 1
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.uses_flat_scratch, 0
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.has_dyn_sized_stack, 0
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.has_recursion, 0
	.set _Z33test_intra_lane_transpose_patternPKDF16_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1104
; TotalNumSgprs: 20
; NumVgprs: 30
; NumAgprs: 0
; TotalNumVgprs: 30
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 30
; AccumOffset: 32
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z28test_intra_lane_no_conflictsPKDF16_Pf ; -- Begin function _Z28test_intra_lane_no_conflictsPKDF16_Pf
	.globl	_Z28test_intra_lane_no_conflictsPKDF16_Pf
	.p2align	8
	.type	_Z28test_intra_lane_no_conflictsPKDF16_Pf,@function
_Z28test_intra_lane_no_conflictsPKDF16_Pf: ; @_Z28test_intra_lane_no_conflictsPKDF16_Pf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x1c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s8, s2, 0xffff
	v_cvt_f32_u32_e32 v2, s8
	s_movk_i32 s2, 0x800
	v_add_u32_e32 v1, s8, v0
	v_mov_b32_e32 v3, s8
	v_rcp_iflag_f32_e32 v2, v2
	v_cmp_gt_u32_e32 vcc, s2, v1
	s_cmp_eq_u32 s8, 1
	s_cselect_b64 s[4:5], -1, 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_addc_co_u32_e64 v3, s[2:3], v0, v3, vcc
	v_max_u32_e32 v4, 0x800, v1
	s_sub_i32 s2, 0, s8
	v_sub_u32_e32 v3, v4, v3
	v_mul_lo_u32 v4, s2, v2
	v_mul_hi_u32 v4, v2, v4
	v_add_u32_e32 v2, v2, v4
	v_mul_hi_u32 v2, v3, v2
	v_mul_lo_u32 v4, v2, s8
	v_sub_u32_e32 v3, v3, v4
	v_add_u32_e32 v5, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	v_subrev_u32_e32 v4, s8, v3
	s_nop 0
	v_cndmask_b32_e64 v2, v2, v5, s[2:3]
	v_cndmask_b32_e64 v3, v3, v4, s[2:3]
	v_add_u32_e32 v4, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	s_nop 1
	v_cndmask_b32_e64 v3, v2, v4, s[2:3]
	v_addc_co_u32_e64 v8, s[2:3], 1, v3, vcc
	v_cmp_lt_u32_e64 s[2:3], 3, v8
	s_and_b64 s[6:7], s[2:3], s[4:5]
	s_mov_b64 s[4:5], -1
	v_mov_b32_e32 v2, v0
	s_and_saveexec_b64 s[2:3], s[6:7]
	s_cbranch_execz .LBB1_9
; %bb.1:                                ; %vector.ph
	v_addc_co_u32_e32 v6, vcc, 0, v3, vcc
	v_mad_u64_u32 v[4:5], s[4:5], s8, 3, v[0:1]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, -3, v6
	v_lshl_add_u32 v2, s8, 1, v0
	v_lshrrev_b32_e32 v5, 2, v4
	v_add_u32_e32 v9, 1, v5
	v_cmp_lt_u32_e32 vcc, 11, v4
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b32_e32 v12, 0
	v_mov_b64_e32 v[4:5], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_5
; %bb.2:                                ; %vector.ph.new
	v_mov_b64_e32 v[6:7], v[2:3]
	v_and_b32_e32 v10, 0x7ffffffc, v9
	s_lshl_b32 s9, s8, 3
	v_lshlrev_b32_e32 v11, 1, v0
	s_lshl_b32 s10, s8, 5
	s_lshl_b32 s11, s8, 4
	s_mul_i32 s12, s8, 24
	s_mov_b32 s13, 0
	s_mov_b64 s[6:7], 0
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB1_3:                                ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v1, v7
	v_cvt_f32_u32_e32 v2, v6
	v_cvt_f32_u32_e32 v3, v5
	v_cvt_f32_u32_e32 v13, v4
	v_add_u32_e32 v12, 4, v4
	v_add_u32_e32 v14, 4, v5
	v_add_u32_e32 v15, 4, v6
	v_add_u32_e32 v16, 4, v7
	v_add_u32_e32 v17, 8, v4
	v_add_u32_e32 v19, 8, v5
	v_add_u32_e32 v20, 8, v6
	v_add_u32_e32 v21, 8, v7
	v_add_u32_e32 v23, 12, v4
	v_add_u32_e32 v24, 12, v5
	v_add_u32_e32 v25, 12, v6
	v_add_u32_e32 v26, 12, v7
	v_cvt_f32_u32_e32 v16, v16
	v_cvt_f32_u32_e32 v15, v15
	v_cvt_f32_u32_e32 v14, v14
	v_cvt_f32_u32_e32 v28, v12
	v_cvt_f32_u32_e32 v21, v21
	v_cvt_f32_u32_e32 v20, v20
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v17, v17
	v_cvt_f32_u32_e32 v26, v26
	v_cvt_f32_u32_e32 v25, v25
	v_cvt_f32_u32_e32 v24, v24
	v_cvt_f32_u32_e32 v23, v23
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v29, v3
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v26, v26
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v23, v23
	v_add_u32_e32 v10, -4, v10
	s_add_i32 s13, s13, 16
	v_cmp_eq_u32_e32 vcc, 0, v10
	v_pack_b32_f16 v3, v2, v1
	v_pack_b32_f16 v2, v13, v29
	v_add_u32_e32 v18, s9, v11
	v_add_u32_e32 v22, s11, v11
	v_add_u32_e32 v27, s12, v11
	v_add_u32_e32 v7, 16, v7
	v_add_u32_e32 v6, 16, v6
	v_add_u32_e32 v5, 16, v5
	v_add_u32_e32 v4, 16, v4
	v_mov_b32_e32 v12, s13
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v11, v[2:3]
	v_pack_b32_f16 v3, v15, v16
	v_pack_b32_f16 v2, v28, v14
	v_add_u32_e32 v11, s10, v11
	v_pack_b32_f16 v15, v20, v21
	v_pack_b32_f16 v14, v17, v19
	v_pack_b32_f16 v17, v25, v26
	v_pack_b32_f16 v16, v23, v24
	ds_write_b64 v18, v[2:3]
	ds_write_b64 v22, v[14:15]
	ds_write_b64 v27, v[16:17]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_3
; %bb.4:                                ; %Flow47
	s_or_b64 exec, exec, s[6:7]
.LBB1_5:                                ; %Flow48
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, 3, v9
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_8
; %bb.6:                                ; %vector.body.epil.preheader
	v_mul_lo_u32 v2, v12, s8
	v_add_lshl_u32 v2, v0, v2, 1
	s_lshl_b32 s9, s8, 3
	s_mov_b64 s[6:7], 0
.LBB1_7:                                ; %vector.body.epil
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v7
	v_cvt_f32_u32_e32 v9, v6
	v_cvt_f32_u32_e32 v10, v5
	v_cvt_f32_u32_e32 v11, v4
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v12, v11
	v_add_u32_e32 v1, -1, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	v_pack_b32_f16 v11, v9, v3
	v_pack_b32_f16 v10, v12, v10
	v_add_u32_e32 v7, 4, v7
	v_add_u32_e32 v6, 4, v6
	v_add_u32_e32 v5, 4, v5
	v_add_u32_e32 v4, 4, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v2, v[10:11]
	v_add_u32_e32 v2, s9, v2
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB1_7
.LBB1_8:                                ; %Flow46
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB1_9:                                ; %Flow49
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB1_12
; %bb.10:                               ; %for.body.preheader
	v_lshlrev_b32_e32 v1, 1, v2
	s_lshl_b32 s6, s8, 1
	s_mov_b64 s[4:5], 0
	s_movk_i32 s7, 0x7ff
.LBB1_11:                               ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v2
	v_add_u32_e32 v2, s8, v2
	v_cmp_lt_u32_e32 vcc, s7, v2
	s_or_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v3, v3
	ds_write_b16 v1, v3
	v_add_u32_e32 v1, s6, v1
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB1_11
.LBB1_12:                               ; %Flow50
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB1_14
; %bb.13:                               ; %for.body11.preheader
	v_mov_b32_e32 v0, 0
	ds_read_u16 v1, v0
	ds_read_u16 v2, v0 offset:4
	ds_read_u16 v3, v0 offset:8
	ds_read_u16 v4, v0 offset:12
	ds_read_u16 v5, v0 offset:16
	ds_read_u16 v6, v0 offset:20
	ds_read_u16 v7, v0 offset:24
	ds_read_u16 v8, v0 offset:28
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v1, v1
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v2, v2
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v3, v3
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v4, v4
	v_add_f32_e32 v1, 0, v1
	v_add_f32_e32 v1, v1, v2
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v2, v5
	v_add_f32_e32 v1, v1, v3
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v3, v6
	v_add_f32_e32 v1, v1, v4
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v4, v7
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v5, v8
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	global_store_dword v0, v1, s[0:1]
.LBB1_14:                               ; %if.end
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z28test_intra_lane_no_conflictsPKDF16_Pf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z28test_intra_lane_no_conflictsPKDF16_Pf, .Lfunc_end1-_Z28test_intra_lane_no_conflictsPKDF16_Pf
                                        ; -- End function
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.num_vgpr, 30
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.num_agpr, 0
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.numbered_sgpr, 14
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.private_seg_size, 0
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.uses_vcc, 1
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.uses_flat_scratch, 0
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.has_dyn_sized_stack, 0
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.has_recursion, 0
	.set _Z28test_intra_lane_no_conflictsPKDF16_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1104
; TotalNumSgprs: 20
; NumVgprs: 30
; NumAgprs: 0
; TotalNumVgprs: 30
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 30
; AccumOffset: 32
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf ; -- Begin function _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.globl	_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.p2align	8
	.type	_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf,@function
_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf: ; @_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x1c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s8, s2, 0xffff
	v_cvt_f32_u32_e32 v2, s8
	s_movk_i32 s2, 0x800
	v_add_u32_e32 v1, s8, v0
	v_mov_b32_e32 v3, s8
	v_rcp_iflag_f32_e32 v2, v2
	v_cmp_gt_u32_e32 vcc, s2, v1
	s_cmp_eq_u32 s8, 1
	s_cselect_b64 s[4:5], -1, 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_addc_co_u32_e64 v3, s[2:3], v0, v3, vcc
	v_max_u32_e32 v4, 0x800, v1
	s_sub_i32 s2, 0, s8
	v_sub_u32_e32 v3, v4, v3
	v_mul_lo_u32 v4, s2, v2
	v_mul_hi_u32 v4, v2, v4
	v_add_u32_e32 v2, v2, v4
	v_mul_hi_u32 v2, v3, v2
	v_mul_lo_u32 v4, v2, s8
	v_sub_u32_e32 v3, v3, v4
	v_add_u32_e32 v5, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	v_subrev_u32_e32 v4, s8, v3
	s_nop 0
	v_cndmask_b32_e64 v2, v2, v5, s[2:3]
	v_cndmask_b32_e64 v3, v3, v4, s[2:3]
	v_add_u32_e32 v4, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	s_nop 1
	v_cndmask_b32_e64 v3, v2, v4, s[2:3]
	v_addc_co_u32_e64 v8, s[2:3], 1, v3, vcc
	v_cmp_lt_u32_e64 s[2:3], 3, v8
	s_and_b64 s[6:7], s[2:3], s[4:5]
	s_mov_b64 s[4:5], -1
	v_mov_b32_e32 v2, v0
	s_and_saveexec_b64 s[2:3], s[6:7]
	s_cbranch_execz .LBB2_9
; %bb.1:                                ; %vector.ph
	v_addc_co_u32_e32 v6, vcc, 0, v3, vcc
	v_mad_u64_u32 v[4:5], s[4:5], s8, 3, v[0:1]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, -3, v6
	v_lshl_add_u32 v2, s8, 1, v0
	v_lshrrev_b32_e32 v5, 2, v4
	v_add_u32_e32 v9, 1, v5
	v_cmp_lt_u32_e32 vcc, 11, v4
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b32_e32 v12, 0
	v_mov_b64_e32 v[4:5], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB2_5
; %bb.2:                                ; %vector.ph.new
	v_mov_b64_e32 v[6:7], v[2:3]
	v_and_b32_e32 v10, 0x7ffffffc, v9
	s_lshl_b32 s9, s8, 3
	v_lshlrev_b32_e32 v11, 1, v0
	s_lshl_b32 s10, s8, 5
	s_lshl_b32 s11, s8, 4
	s_mul_i32 s12, s8, 24
	s_mov_b32 s13, 0
	s_mov_b64 s[6:7], 0
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB2_3:                                ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v1, v7
	v_cvt_f32_u32_e32 v2, v6
	v_cvt_f32_u32_e32 v3, v5
	v_cvt_f32_u32_e32 v13, v4
	v_add_u32_e32 v12, 4, v4
	v_add_u32_e32 v14, 4, v5
	v_add_u32_e32 v15, 4, v6
	v_add_u32_e32 v16, 4, v7
	v_add_u32_e32 v17, 8, v4
	v_add_u32_e32 v19, 8, v5
	v_add_u32_e32 v20, 8, v6
	v_add_u32_e32 v21, 8, v7
	v_add_u32_e32 v23, 12, v4
	v_add_u32_e32 v24, 12, v5
	v_add_u32_e32 v25, 12, v6
	v_add_u32_e32 v26, 12, v7
	v_cvt_f32_u32_e32 v16, v16
	v_cvt_f32_u32_e32 v15, v15
	v_cvt_f32_u32_e32 v14, v14
	v_cvt_f32_u32_e32 v28, v12
	v_cvt_f32_u32_e32 v21, v21
	v_cvt_f32_u32_e32 v20, v20
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v17, v17
	v_cvt_f32_u32_e32 v26, v26
	v_cvt_f32_u32_e32 v25, v25
	v_cvt_f32_u32_e32 v24, v24
	v_cvt_f32_u32_e32 v23, v23
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v29, v3
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v26, v26
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v23, v23
	v_add_u32_e32 v10, -4, v10
	s_add_i32 s13, s13, 16
	v_cmp_eq_u32_e32 vcc, 0, v10
	v_pack_b32_f16 v3, v2, v1
	v_pack_b32_f16 v2, v13, v29
	v_add_u32_e32 v18, s9, v11
	v_add_u32_e32 v22, s11, v11
	v_add_u32_e32 v27, s12, v11
	v_add_u32_e32 v7, 16, v7
	v_add_u32_e32 v6, 16, v6
	v_add_u32_e32 v5, 16, v5
	v_add_u32_e32 v4, 16, v4
	v_mov_b32_e32 v12, s13
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v11, v[2:3]
	v_pack_b32_f16 v3, v15, v16
	v_pack_b32_f16 v2, v28, v14
	v_add_u32_e32 v11, s10, v11
	v_pack_b32_f16 v15, v20, v21
	v_pack_b32_f16 v14, v17, v19
	v_pack_b32_f16 v17, v25, v26
	v_pack_b32_f16 v16, v23, v24
	ds_write_b64 v18, v[2:3]
	ds_write_b64 v22, v[14:15]
	ds_write_b64 v27, v[16:17]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB2_3
; %bb.4:                                ; %Flow47
	s_or_b64 exec, exec, s[6:7]
.LBB2_5:                                ; %Flow48
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, 3, v9
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB2_8
; %bb.6:                                ; %vector.body.epil.preheader
	v_mul_lo_u32 v2, v12, s8
	v_add_lshl_u32 v2, v0, v2, 1
	s_lshl_b32 s9, s8, 3
	s_mov_b64 s[6:7], 0
.LBB2_7:                                ; %vector.body.epil
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v7
	v_cvt_f32_u32_e32 v9, v6
	v_cvt_f32_u32_e32 v10, v5
	v_cvt_f32_u32_e32 v11, v4
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v12, v11
	v_add_u32_e32 v1, -1, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	v_pack_b32_f16 v11, v9, v3
	v_pack_b32_f16 v10, v12, v10
	v_add_u32_e32 v7, 4, v7
	v_add_u32_e32 v6, 4, v6
	v_add_u32_e32 v5, 4, v5
	v_add_u32_e32 v4, 4, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v2, v[10:11]
	v_add_u32_e32 v2, s9, v2
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB2_7
.LBB2_8:                                ; %Flow46
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB2_9:                                ; %Flow49
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB2_12
; %bb.10:                               ; %for.body.preheader
	v_lshlrev_b32_e32 v1, 1, v2
	s_lshl_b32 s6, s8, 1
	s_mov_b64 s[4:5], 0
	s_movk_i32 s7, 0x7ff
.LBB2_11:                               ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v2
	v_add_u32_e32 v2, s8, v2
	v_cmp_lt_u32_e32 vcc, s7, v2
	s_or_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v3, v3
	ds_write_b16 v1, v3
	v_add_u32_e32 v1, s6, v1
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB2_11
.LBB2_12:                               ; %Flow50
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_14
; %bb.13:                               ; %for.body11.preheader
	v_mov_b32_e32 v0, 0
	ds_read_u16 v1, v0
	ds_read_u16 v2, v0 offset:128
	ds_read_u16 v3, v0 offset:256
	ds_read_u16 v4, v0 offset:384
	ds_read_u16 v5, v0 offset:512
	ds_read_u16 v6, v0 offset:640
	ds_read_u16 v7, v0 offset:768
	ds_read_u16 v8, v0 offset:896
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v1, v1
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v2, v2
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v3, v3
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v4, v4
	v_add_f32_e32 v1, 0, v1
	v_add_f32_e32 v1, v1, v2
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v2, v5
	v_add_f32_e32 v1, v1, v3
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v3, v6
	v_add_f32_e32 v1, v1, v4
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v4, v7
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v5, v8
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v3
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	global_store_dword v0, v1, s[0:1]
.LBB2_14:                               ; %if.end
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end2:
	.size	_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf, .Lfunc_end2-_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
                                        ; -- End function
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.num_vgpr, 30
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.num_agpr, 0
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.numbered_sgpr, 14
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.private_seg_size, 0
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.uses_vcc, 1
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.uses_flat_scratch, 0
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.has_dyn_sized_stack, 0
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.has_recursion, 0
	.set _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1104
; TotalNumSgprs: 20
; NumVgprs: 30
; NumAgprs: 0
; TotalNumVgprs: 30
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 30
; AccumOffset: 32
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z24test_full_phase0_patternPKDF16_Pf ; -- Begin function _Z24test_full_phase0_patternPKDF16_Pf
	.globl	_Z24test_full_phase0_patternPKDF16_Pf
	.p2align	8
	.type	_Z24test_full_phase0_patternPKDF16_Pf,@function
_Z24test_full_phase0_patternPKDF16_Pf:  ; @_Z24test_full_phase0_patternPKDF16_Pf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x1c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s8, s2, 0xffff
	v_cvt_f32_u32_e32 v2, s8
	s_movk_i32 s2, 0x800
	v_add_u32_e32 v1, s8, v0
	v_mov_b32_e32 v3, s8
	v_rcp_iflag_f32_e32 v2, v2
	v_cmp_gt_u32_e32 vcc, s2, v1
	s_cmp_eq_u32 s8, 1
	s_cselect_b64 s[4:5], -1, 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_addc_co_u32_e64 v3, s[2:3], v0, v3, vcc
	v_max_u32_e32 v4, 0x800, v1
	s_sub_i32 s2, 0, s8
	v_sub_u32_e32 v3, v4, v3
	v_mul_lo_u32 v4, s2, v2
	v_mul_hi_u32 v4, v2, v4
	v_add_u32_e32 v2, v2, v4
	v_mul_hi_u32 v2, v3, v2
	v_mul_lo_u32 v4, v2, s8
	v_sub_u32_e32 v3, v3, v4
	v_add_u32_e32 v5, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	v_subrev_u32_e32 v4, s8, v3
	s_nop 0
	v_cndmask_b32_e64 v2, v2, v5, s[2:3]
	v_cndmask_b32_e64 v3, v3, v4, s[2:3]
	v_add_u32_e32 v4, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	s_nop 1
	v_cndmask_b32_e64 v3, v2, v4, s[2:3]
	v_addc_co_u32_e64 v8, s[2:3], 1, v3, vcc
	v_cmp_lt_u32_e64 s[2:3], 3, v8
	s_and_b64 s[6:7], s[2:3], s[4:5]
	s_mov_b64 s[4:5], -1
	v_mov_b32_e32 v2, v0
	s_and_saveexec_b64 s[2:3], s[6:7]
	s_cbranch_execz .LBB3_9
; %bb.1:                                ; %vector.ph
	v_addc_co_u32_e32 v6, vcc, 0, v3, vcc
	v_mad_u64_u32 v[4:5], s[4:5], s8, 3, v[0:1]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, -3, v6
	v_lshl_add_u32 v2, s8, 1, v0
	v_lshrrev_b32_e32 v5, 2, v4
	v_add_u32_e32 v9, 1, v5
	v_cmp_lt_u32_e32 vcc, 11, v4
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b32_e32 v12, 0
	v_mov_b64_e32 v[4:5], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %vector.ph.new
	v_mov_b64_e32 v[6:7], v[2:3]
	v_and_b32_e32 v10, 0x7ffffffc, v9
	s_lshl_b32 s9, s8, 3
	v_lshlrev_b32_e32 v11, 1, v0
	s_lshl_b32 s10, s8, 5
	s_lshl_b32 s11, s8, 4
	s_mul_i32 s12, s8, 24
	s_mov_b32 s13, 0
	s_mov_b64 s[6:7], 0
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB3_3:                                ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v1, v7
	v_cvt_f32_u32_e32 v2, v6
	v_cvt_f32_u32_e32 v3, v5
	v_cvt_f32_u32_e32 v13, v4
	v_add_u32_e32 v12, 4, v4
	v_add_u32_e32 v14, 4, v5
	v_add_u32_e32 v15, 4, v6
	v_add_u32_e32 v16, 4, v7
	v_add_u32_e32 v17, 8, v4
	v_add_u32_e32 v19, 8, v5
	v_add_u32_e32 v20, 8, v6
	v_add_u32_e32 v21, 8, v7
	v_add_u32_e32 v23, 12, v4
	v_add_u32_e32 v24, 12, v5
	v_add_u32_e32 v25, 12, v6
	v_add_u32_e32 v26, 12, v7
	v_cvt_f32_u32_e32 v16, v16
	v_cvt_f32_u32_e32 v15, v15
	v_cvt_f32_u32_e32 v14, v14
	v_cvt_f32_u32_e32 v28, v12
	v_cvt_f32_u32_e32 v21, v21
	v_cvt_f32_u32_e32 v20, v20
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v17, v17
	v_cvt_f32_u32_e32 v26, v26
	v_cvt_f32_u32_e32 v25, v25
	v_cvt_f32_u32_e32 v24, v24
	v_cvt_f32_u32_e32 v23, v23
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v29, v3
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v26, v26
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v23, v23
	v_add_u32_e32 v10, -4, v10
	s_add_i32 s13, s13, 16
	v_cmp_eq_u32_e32 vcc, 0, v10
	v_pack_b32_f16 v3, v2, v1
	v_pack_b32_f16 v2, v13, v29
	v_add_u32_e32 v18, s9, v11
	v_add_u32_e32 v22, s11, v11
	v_add_u32_e32 v27, s12, v11
	v_add_u32_e32 v7, 16, v7
	v_add_u32_e32 v6, 16, v6
	v_add_u32_e32 v5, 16, v5
	v_add_u32_e32 v4, 16, v4
	v_mov_b32_e32 v12, s13
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v11, v[2:3]
	v_pack_b32_f16 v3, v15, v16
	v_pack_b32_f16 v2, v28, v14
	v_add_u32_e32 v11, s10, v11
	v_pack_b32_f16 v15, v20, v21
	v_pack_b32_f16 v14, v17, v19
	v_pack_b32_f16 v17, v25, v26
	v_pack_b32_f16 v16, v23, v24
	ds_write_b64 v18, v[2:3]
	ds_write_b64 v22, v[14:15]
	ds_write_b64 v27, v[16:17]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB3_3
; %bb.4:                                ; %Flow89
	s_or_b64 exec, exec, s[6:7]
.LBB3_5:                                ; %Flow90
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, 3, v9
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB3_8
; %bb.6:                                ; %vector.body.epil.preheader
	v_mul_lo_u32 v2, v12, s8
	v_add_lshl_u32 v2, v0, v2, 1
	s_lshl_b32 s9, s8, 3
	s_mov_b64 s[6:7], 0
.LBB3_7:                                ; %vector.body.epil
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v7
	v_cvt_f32_u32_e32 v9, v6
	v_cvt_f32_u32_e32 v10, v5
	v_cvt_f32_u32_e32 v11, v4
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v12, v11
	v_add_u32_e32 v1, -1, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	v_pack_b32_f16 v11, v9, v3
	v_pack_b32_f16 v10, v12, v10
	v_add_u32_e32 v7, 4, v7
	v_add_u32_e32 v6, 4, v6
	v_add_u32_e32 v5, 4, v5
	v_add_u32_e32 v4, 4, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v2, v[10:11]
	v_add_u32_e32 v2, s9, v2
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB3_7
.LBB3_8:                                ; %Flow88
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB3_9:                                ; %Flow91
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB3_12
; %bb.10:                               ; %for.body.preheader
	v_lshlrev_b32_e32 v1, 1, v2
	s_lshl_b32 s6, s8, 1
	s_mov_b64 s[4:5], 0
	s_movk_i32 s7, 0x7ff
.LBB3_11:                               ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v2
	v_add_u32_e32 v2, s8, v2
	v_cmp_lt_u32_e32 vcc, s7, v2
	s_or_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v3, v3
	ds_write_b16 v1, v3
	v_add_u32_e32 v1, s6, v1
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB3_11
.LBB3_12:                               ; %Flow92
	s_or_b64 exec, exec, s[2:3]
	v_and_b32_e32 v1, 7, v0
	v_cmp_eq_u32_e32 vcc, 0, v0
	v_mov_b32_e32 v2, 0
	v_lshlrev_b32_e32 v1, 1, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_14
; %bb.13:                               ; %for.body17.preheader
	v_mov_b32_e32 v2, 0
	ds_read_u16 v3, v1
	ds_read_u16 v4, v2 offset:64
	ds_read_u16 v5, v2 offset:128
	ds_read_u16 v6, v2 offset:192
	ds_read_u16 v7, v2 offset:256
	ds_read_u16 v8, v2 offset:320
	ds_read_u16 v9, v2 offset:384
	ds_read_u16 v2, v2 offset:448
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v3, v3
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v6, v6
	v_add_f32_e32 v3, 0, v3
	v_add_f32_e32 v3, v3, v4
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v4, v7
	v_add_f32_e32 v3, v3, v5
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v5, v8
	v_add_f32_e32 v3, v3, v6
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v6, v9
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v2, v2
	v_add_f32_e32 v3, v3, v4
	v_add_f32_e32 v3, v3, v5
	v_add_f32_e32 v3, v3, v6
	v_add_f32_e32 v2, v3, v2
.LBB3_14:                               ; %for.inc25
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 1, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_16
; %bb.15:                               ; %for.body17.preheader.1
	v_mov_b32_e32 v3, 0
	ds_read_u16 v4, v1
	ds_read_u16 v5, v3 offset:66
	ds_read_u16 v6, v3 offset:130
	ds_read_u16 v7, v3 offset:194
	ds_read_u16 v8, v3 offset:258
	ds_read_u16 v9, v3 offset:322
	ds_read_u16 v10, v3 offset:386
	ds_read_u16 v3, v3 offset:450
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v7, v7
	v_add_f32_e32 v2, v2, v4
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v4, v8
	v_add_f32_e32 v2, v2, v5
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v5, v9
	v_add_f32_e32 v2, v2, v6
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v6, v10
	v_add_f32_e32 v2, v2, v7
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v2, v2, v6
	v_add_f32_e32 v2, v2, v3
.LBB3_16:                               ; %for.inc25.1
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 2, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_18
; %bb.17:                               ; %for.body17.preheader.2
	v_mov_b32_e32 v3, 0
	ds_read_u16 v4, v1
	ds_read_u16 v5, v3 offset:68
	ds_read_u16 v6, v3 offset:132
	ds_read_u16 v7, v3 offset:196
	ds_read_u16 v8, v3 offset:260
	ds_read_u16 v9, v3 offset:324
	ds_read_u16 v10, v3 offset:388
	ds_read_u16 v3, v3 offset:452
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v7, v7
	v_add_f32_e32 v2, v2, v4
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v4, v8
	v_add_f32_e32 v2, v2, v5
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v5, v9
	v_add_f32_e32 v2, v2, v6
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v6, v10
	v_add_f32_e32 v2, v2, v7
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v2, v2, v6
	v_add_f32_e32 v2, v2, v3
.LBB3_18:                               ; %for.inc25.2
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 3, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_20
; %bb.19:                               ; %for.body17.preheader.3
	v_mov_b32_e32 v3, 0
	ds_read_u16 v4, v1
	ds_read_u16 v5, v3 offset:70
	ds_read_u16 v6, v3 offset:134
	ds_read_u16 v7, v3 offset:198
	ds_read_u16 v8, v3 offset:262
	ds_read_u16 v9, v3 offset:326
	ds_read_u16 v10, v3 offset:390
	ds_read_u16 v3, v3 offset:454
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v7, v7
	v_add_f32_e32 v2, v2, v4
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v4, v8
	v_add_f32_e32 v2, v2, v5
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v5, v9
	v_add_f32_e32 v2, v2, v6
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v6, v10
	v_add_f32_e32 v2, v2, v7
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v2, v2, v6
	v_add_f32_e32 v2, v2, v3
.LBB3_20:                               ; %for.inc25.3
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 20, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_22
; %bb.21:                               ; %for.body17.preheader.4
	v_mov_b32_e32 v3, 0
	ds_read_u16 v4, v1
	ds_read_u16 v5, v3 offset:72
	ds_read_u16 v6, v3 offset:136
	ds_read_u16 v7, v3 offset:200
	ds_read_u16 v8, v3 offset:264
	ds_read_u16 v9, v3 offset:328
	ds_read_u16 v10, v3 offset:392
	ds_read_u16 v3, v3 offset:456
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v7, v7
	v_add_f32_e32 v2, v2, v4
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v4, v8
	v_add_f32_e32 v2, v2, v5
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v5, v9
	v_add_f32_e32 v2, v2, v6
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v6, v10
	v_add_f32_e32 v2, v2, v7
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v2, v2, v6
	v_add_f32_e32 v2, v2, v3
.LBB3_22:                               ; %for.inc25.4
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 21, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_24
; %bb.23:                               ; %for.body17.preheader.5
	v_mov_b32_e32 v3, 0
	ds_read_u16 v4, v1
	ds_read_u16 v5, v3 offset:74
	ds_read_u16 v6, v3 offset:138
	ds_read_u16 v7, v3 offset:202
	ds_read_u16 v8, v3 offset:266
	ds_read_u16 v9, v3 offset:330
	ds_read_u16 v10, v3 offset:394
	ds_read_u16 v3, v3 offset:458
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v7, v7
	v_add_f32_e32 v2, v2, v4
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v4, v8
	v_add_f32_e32 v2, v2, v5
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v5, v9
	v_add_f32_e32 v2, v2, v6
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v6, v10
	v_add_f32_e32 v2, v2, v7
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v2, v2, v6
	v_add_f32_e32 v2, v2, v3
.LBB3_24:                               ; %for.inc25.5
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 22, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_26
; %bb.25:                               ; %for.body17.preheader.6
	v_mov_b32_e32 v3, 0
	ds_read_u16 v4, v1
	ds_read_u16 v5, v3 offset:76
	ds_read_u16 v6, v3 offset:140
	ds_read_u16 v7, v3 offset:204
	ds_read_u16 v8, v3 offset:268
	ds_read_u16 v9, v3 offset:332
	ds_read_u16 v10, v3 offset:396
	ds_read_u16 v3, v3 offset:460
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v7, v7
	v_add_f32_e32 v2, v2, v4
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v4, v8
	v_add_f32_e32 v2, v2, v5
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v5, v9
	v_add_f32_e32 v2, v2, v6
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v6, v10
	v_add_f32_e32 v2, v2, v7
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v2, v2, v6
	v_add_f32_e32 v2, v2, v3
.LBB3_26:                               ; %for.inc25.6
	s_or_b64 exec, exec, s[2:3]
	v_cmp_eq_u32_e32 vcc, 23, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_28
; %bb.27:                               ; %for.body17.preheader.7
	v_mov_b32_e32 v3, 0
	ds_read_u16 v1, v1
	ds_read_u16 v4, v3 offset:78
	ds_read_u16 v5, v3 offset:142
	ds_read_u16 v6, v3 offset:206
	ds_read_u16 v7, v3 offset:270
	ds_read_u16 v8, v3 offset:334
	ds_read_u16 v9, v3 offset:398
	ds_read_u16 v3, v3 offset:462
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v1, v1
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v6, v6
	v_add_f32_e32 v1, v2, v1
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v2, v7
	v_add_f32_e32 v1, v1, v4
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v4, v8
	v_add_f32_e32 v1, v1, v5
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v5, v9
	v_add_f32_e32 v1, v1, v6
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v3, v3
	v_add_f32_e32 v1, v1, v2
	v_add_f32_e32 v1, v1, v4
	v_add_f32_e32 v1, v1, v5
	v_add_f32_e32 v2, v1, v3
.LBB3_28:                               ; %for.inc25.7
	s_or_b64 exec, exec, s[2:3]
	v_cmp_gt_u32_e32 vcc, 8, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_30
; %bb.29:                               ; %if.then29
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v2, s[0:1]
.LBB3_30:                               ; %if.end32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z24test_full_phase0_patternPKDF16_Pf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end3:
	.size	_Z24test_full_phase0_patternPKDF16_Pf, .Lfunc_end3-_Z24test_full_phase0_patternPKDF16_Pf
                                        ; -- End function
	.set _Z24test_full_phase0_patternPKDF16_Pf.num_vgpr, 30
	.set _Z24test_full_phase0_patternPKDF16_Pf.num_agpr, 0
	.set _Z24test_full_phase0_patternPKDF16_Pf.numbered_sgpr, 14
	.set _Z24test_full_phase0_patternPKDF16_Pf.private_seg_size, 0
	.set _Z24test_full_phase0_patternPKDF16_Pf.uses_vcc, 1
	.set _Z24test_full_phase0_patternPKDF16_Pf.uses_flat_scratch, 0
	.set _Z24test_full_phase0_patternPKDF16_Pf.has_dyn_sized_stack, 0
	.set _Z24test_full_phase0_patternPKDF16_Pf.has_recursion, 0
	.set _Z24test_full_phase0_patternPKDF16_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2400
; TotalNumSgprs: 20
; NumVgprs: 30
; NumAgprs: 0
; TotalNumVgprs: 30
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 30
; AccumOffset: 32
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z27test_scaled_intra_conflictsPKDF16_Pf ; -- Begin function _Z27test_scaled_intra_conflictsPKDF16_Pf
	.globl	_Z27test_scaled_intra_conflictsPKDF16_Pf
	.p2align	8
	.type	_Z27test_scaled_intra_conflictsPKDF16_Pf,@function
_Z27test_scaled_intra_conflictsPKDF16_Pf: ; @_Z27test_scaled_intra_conflictsPKDF16_Pf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x1c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s8, s2, 0xffff
	v_cvt_f32_u32_e32 v2, s8
	s_movk_i32 s2, 0x800
	v_add_u32_e32 v1, s8, v0
	v_mov_b32_e32 v3, s8
	v_rcp_iflag_f32_e32 v2, v2
	v_cmp_gt_u32_e32 vcc, s2, v1
	s_cmp_eq_u32 s8, 1
	s_cselect_b64 s[4:5], -1, 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_addc_co_u32_e64 v3, s[2:3], v0, v3, vcc
	v_max_u32_e32 v4, 0x800, v1
	s_sub_i32 s2, 0, s8
	v_sub_u32_e32 v3, v4, v3
	v_mul_lo_u32 v4, s2, v2
	v_mul_hi_u32 v4, v2, v4
	v_add_u32_e32 v2, v2, v4
	v_mul_hi_u32 v2, v3, v2
	v_mul_lo_u32 v4, v2, s8
	v_sub_u32_e32 v3, v3, v4
	v_add_u32_e32 v5, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	v_subrev_u32_e32 v4, s8, v3
	s_nop 0
	v_cndmask_b32_e64 v2, v2, v5, s[2:3]
	v_cndmask_b32_e64 v3, v3, v4, s[2:3]
	v_add_u32_e32 v4, 1, v2
	v_cmp_le_u32_e64 s[2:3], s8, v3
	s_nop 1
	v_cndmask_b32_e64 v3, v2, v4, s[2:3]
	v_addc_co_u32_e64 v8, s[2:3], 1, v3, vcc
	v_cmp_lt_u32_e64 s[2:3], 3, v8
	s_and_b64 s[6:7], s[2:3], s[4:5]
	s_mov_b64 s[4:5], -1
	v_mov_b32_e32 v2, v0
	s_and_saveexec_b64 s[2:3], s[6:7]
	s_cbranch_execz .LBB4_9
; %bb.1:                                ; %vector.ph
	v_addc_co_u32_e32 v6, vcc, 0, v3, vcc
	v_mad_u64_u32 v[4:5], s[4:5], s8, 3, v[0:1]
	v_mov_b32_e32 v3, v4
	v_add_u32_e32 v4, -3, v6
	v_lshl_add_u32 v2, s8, 1, v0
	v_lshrrev_b32_e32 v5, 2, v4
	v_add_u32_e32 v9, 1, v5
	v_cmp_lt_u32_e32 vcc, 11, v4
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b32_e32 v12, 0
	v_mov_b64_e32 v[4:5], v[0:1]
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB4_5
; %bb.2:                                ; %vector.ph.new
	v_mov_b64_e32 v[6:7], v[2:3]
	v_and_b32_e32 v10, 0x7ffffffc, v9
	s_lshl_b32 s9, s8, 3
	v_lshlrev_b32_e32 v11, 1, v0
	s_lshl_b32 s10, s8, 5
	s_lshl_b32 s11, s8, 4
	s_mul_i32 s12, s8, 24
	s_mov_b32 s13, 0
	s_mov_b64 s[6:7], 0
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB4_3:                                ; %vector.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v1, v7
	v_cvt_f32_u32_e32 v2, v6
	v_cvt_f32_u32_e32 v3, v5
	v_cvt_f32_u32_e32 v13, v4
	v_add_u32_e32 v12, 4, v4
	v_add_u32_e32 v14, 4, v5
	v_add_u32_e32 v15, 4, v6
	v_add_u32_e32 v16, 4, v7
	v_add_u32_e32 v17, 8, v4
	v_add_u32_e32 v19, 8, v5
	v_add_u32_e32 v20, 8, v6
	v_add_u32_e32 v21, 8, v7
	v_add_u32_e32 v23, 12, v4
	v_add_u32_e32 v24, 12, v5
	v_add_u32_e32 v25, 12, v6
	v_add_u32_e32 v26, 12, v7
	v_cvt_f32_u32_e32 v16, v16
	v_cvt_f32_u32_e32 v15, v15
	v_cvt_f32_u32_e32 v14, v14
	v_cvt_f32_u32_e32 v28, v12
	v_cvt_f32_u32_e32 v21, v21
	v_cvt_f32_u32_e32 v20, v20
	v_cvt_f32_u32_e32 v19, v19
	v_cvt_f32_u32_e32 v17, v17
	v_cvt_f32_u32_e32 v26, v26
	v_cvt_f32_u32_e32 v25, v25
	v_cvt_f32_u32_e32 v24, v24
	v_cvt_f32_u32_e32 v23, v23
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v29, v3
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v28, v28
	v_cvt_f16_f32_e32 v21, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v26, v26
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v24, v24
	v_cvt_f16_f32_e32 v23, v23
	v_add_u32_e32 v10, -4, v10
	s_add_i32 s13, s13, 16
	v_cmp_eq_u32_e32 vcc, 0, v10
	v_pack_b32_f16 v3, v2, v1
	v_pack_b32_f16 v2, v13, v29
	v_add_u32_e32 v18, s9, v11
	v_add_u32_e32 v22, s11, v11
	v_add_u32_e32 v27, s12, v11
	v_add_u32_e32 v7, 16, v7
	v_add_u32_e32 v6, 16, v6
	v_add_u32_e32 v5, 16, v5
	v_add_u32_e32 v4, 16, v4
	v_mov_b32_e32 v12, s13
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v11, v[2:3]
	v_pack_b32_f16 v3, v15, v16
	v_pack_b32_f16 v2, v28, v14
	v_add_u32_e32 v11, s10, v11
	v_pack_b32_f16 v15, v20, v21
	v_pack_b32_f16 v14, v17, v19
	v_pack_b32_f16 v17, v25, v26
	v_pack_b32_f16 v16, v23, v24
	ds_write_b64 v18, v[2:3]
	ds_write_b64 v22, v[14:15]
	ds_write_b64 v27, v[16:17]
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB4_3
; %bb.4:                                ; %Flow64
	s_or_b64 exec, exec, s[6:7]
.LBB4_5:                                ; %Flow65
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, 3, v9
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB4_8
; %bb.6:                                ; %vector.body.epil.preheader
	v_mul_lo_u32 v2, v12, s8
	v_add_lshl_u32 v2, v0, v2, 1
	s_lshl_b32 s9, s8, 3
	s_mov_b64 s[6:7], 0
.LBB4_7:                                ; %vector.body.epil
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v7
	v_cvt_f32_u32_e32 v9, v6
	v_cvt_f32_u32_e32 v10, v5
	v_cvt_f32_u32_e32 v11, v4
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v12, v11
	v_add_u32_e32 v1, -1, v1
	v_cmp_eq_u32_e32 vcc, 0, v1
	v_pack_b32_f16 v11, v9, v3
	v_pack_b32_f16 v10, v12, v10
	v_add_u32_e32 v7, 4, v7
	v_add_u32_e32 v6, 4, v6
	v_add_u32_e32 v5, 4, v5
	v_add_u32_e32 v4, 4, v4
	s_or_b64 s[6:7], vcc, s[6:7]
	ds_write_b64 v2, v[10:11]
	v_add_u32_e32 v2, s9, v2
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB4_7
.LBB4_8:                                ; %Flow63
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB4_9:                                ; %Flow66
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB4_12
; %bb.10:                               ; %for.body.preheader
	v_lshlrev_b32_e32 v1, 1, v2
	s_lshl_b32 s6, s8, 1
	s_mov_b64 s[4:5], 0
	s_movk_i32 s7, 0x7ff
.LBB4_11:                               ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v2
	v_add_u32_e32 v2, s8, v2
	v_cmp_lt_u32_e32 vcc, s7, v2
	s_or_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v3, v3
	ds_write_b16 v1, v3
	v_add_u32_e32 v1, s6, v1
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB4_11
.LBB4_12:                               ; %Flow67
	s_or_b64 exec, exec, s[2:3]
	v_cmp_gt_u32_e32 vcc, 32, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB4_14
; %bb.13:                               ; %for.body10.preheader
	v_lshlrev_b32_e32 v1, 1, v0
	ds_read_u16 v2, v1
	ds_read_u16 v3, v1 offset:64
	ds_read_u16 v4, v1 offset:128
	ds_read_u16 v5, v1 offset:192
	ds_read_u16 v6, v1 offset:256
	ds_read_u16 v7, v1 offset:320
	ds_read_u16 v8, v1 offset:384
	ds_read_u16 v1, v1 offset:448
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v2, v2
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v3, v3
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v5, v5
	v_add_f32_e32 v2, 0, v2
	v_add_f32_e32 v2, v2, v3
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v3, v6
	v_add_f32_e32 v2, v2, v4
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v4, v7
	v_add_f32_e32 v2, v2, v5
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v5, v8
	s_load_dwordx2 s[0:1], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v1, v1
	v_add_f32_e32 v2, v2, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v1, v2, v1
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v0, v1, s[0:1]
.LBB4_14:                               ; %if.end
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z27test_scaled_intra_conflictsPKDF16_Pf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 14
		.amdhsa_accum_offset 32
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end4:
	.size	_Z27test_scaled_intra_conflictsPKDF16_Pf, .Lfunc_end4-_Z27test_scaled_intra_conflictsPKDF16_Pf
                                        ; -- End function
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.num_vgpr, 30
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.num_agpr, 0
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.numbered_sgpr, 14
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.private_seg_size, 0
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.uses_vcc, 1
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.uses_flat_scratch, 0
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.has_dyn_sized_stack, 0
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.has_recursion, 0
	.set _Z27test_scaled_intra_conflictsPKDF16_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1108
; TotalNumSgprs: 20
; NumVgprs: 30
; NumAgprs: 0
; TotalNumVgprs: 30
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 20
; NumVGPRsForWavesPerEU: 30
; AccumOffset: 32
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 7
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_fad6a0bef469ea87,@object ; @__hip_cuid_fad6a0bef469ea87
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_fad6a0bef469ea87
__hip_cuid_fad6a0bef469ea87:
	.byte	0                               ; 0x0
	.size	__hip_cuid_fad6a0bef469ea87, 1

	.ident	"AMD clang version 21.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25401 965357120e93d691c2c2f6b221deb863caf44a62)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_fad6a0bef469ea87
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .name:           lds_ptr.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           output.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z33test_intra_lane_transpose_patternPKDF16_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z33test_intra_lane_transpose_patternPKDF16_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .name:           lds_ptr.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           output.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z28test_intra_lane_no_conflictsPKDF16_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z28test_intra_lane_no_conflictsPKDF16_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .name:           lds_ptr.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           output.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .name:           lds_ptr.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           output.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z24test_full_phase0_patternPKDF16_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z24test_full_phase0_patternPKDF16_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .name:           lds_ptr.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           output.coerce
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z27test_scaled_intra_conflictsPKDF16_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z27test_scaled_intra_conflictsPKDF16_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx942

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"test_intra_lane_conflicts.cpp"
	.text
	.globl	_Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf # -- Begin function _Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf
	.p2align	4
	.type	_Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf,@function
_Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf: # @_Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z33test_intra_lane_transpose_patternPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end0:
	.size	_Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf, .Lfunc_end0-_Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf
	.cfi_endproc
                                        # -- End function
	.globl	_Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf # -- Begin function _Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf
	.p2align	4
	.type	_Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf,@function
_Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf: # @_Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z28test_intra_lane_no_conflictsPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end1:
	.size	_Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf, .Lfunc_end1-_Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf
	.cfi_endproc
                                        # -- End function
	.globl	_Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf # -- Begin function _Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.p2align	4
	.type	_Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf,@function
_Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf: # @_Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end2:
	.size	_Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf, .Lfunc_end2-_Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.cfi_endproc
                                        # -- End function
	.globl	_Z39__device_stub__test_full_phase0_patternPKDF16_Pf # -- Begin function _Z39__device_stub__test_full_phase0_patternPKDF16_Pf
	.p2align	4
	.type	_Z39__device_stub__test_full_phase0_patternPKDF16_Pf,@function
_Z39__device_stub__test_full_phase0_patternPKDF16_Pf: # @_Z39__device_stub__test_full_phase0_patternPKDF16_Pf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z24test_full_phase0_patternPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end3:
	.size	_Z39__device_stub__test_full_phase0_patternPKDF16_Pf, .Lfunc_end3-_Z39__device_stub__test_full_phase0_patternPKDF16_Pf
	.cfi_endproc
                                        # -- End function
	.globl	_Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf # -- Begin function _Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf
	.p2align	4
	.type	_Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf,@function
_Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf: # @_Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movq	%rsi, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z27test_scaled_intra_conflictsPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end4:
	.size	_Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf, .Lfunc_end4-_Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$88, %rsp
	.cfi_def_cfa_offset 112
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	leaq	80(%rsp), %rdi
	movl	$1024, %esi                     # imm = 0x400
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB5_18
# %bb.1:                                # %if.end
	movabsq	$4294967297, %rbx               # imm = 0x100000001
	movl	$_ZSt4cout, %edi
	movl	$.L.str.1, %esi
	movl	$35, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.2, %esi
	movl	$51, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.3, %esi
	movl	$39, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.4, %esi
	movl	$38, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.5, %esi
	movl	$61, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.6, %esi
	movl	$69, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	leaq	255(%rbx), %r14
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB5_3
# %bb.2:                                # %kcall.configok
	movq	80(%rsp), %rax
	movq	$0, 56(%rsp)
	movq	%rax, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z33test_intra_lane_transpose_patternPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB5_3:                                # %kcall.end
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB5_18
# %bb.4:                                # %if.end21
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$14, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.8, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.9, %esi
	movl	$55, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.10, %esi
	movl	$43, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.11, %esi
	movl	$24, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB5_6
# %bb.5:                                # %kcall.configok31
	movq	80(%rsp), %rax
	movq	$0, 56(%rsp)
	movq	%rax, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z28test_intra_lane_no_conflictsPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB5_6:                                # %kcall.end32
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB5_18
# %bb.7:                                # %if.end41
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$14, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.12, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.13, %esi
	movl	$65, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.14, %esi
	movl	$56, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.15, %esi
	movl	$58, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB5_9
# %bb.8:                                # %kcall.configok51
	movq	80(%rsp), %rax
	movq	$0, 56(%rsp)
	movq	%rax, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB5_9:                                # %kcall.end52
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB5_18
# %bb.10:                               # %if.end61
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$14, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.16, %esi
	movl	$39, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.17, %esi
	movl	$62, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.18, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.19, %esi
	movl	$44, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB5_12
# %bb.11:                               # %kcall.configok71
	movq	80(%rsp), %rax
	movq	$0, 56(%rsp)
	movq	%rax, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z24test_full_phase0_patternPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB5_12:                               # %kcall.end72
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB5_18
# %bb.13:                               # %if.end81
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$14, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.20, %esi
	movl	$56, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.21, %esi
	movl	$52, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.22, %esi
	movl	$39, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB5_15
# %bb.14:                               # %kcall.configok90
	movq	80(%rsp), %rax
	movq	$0, 56(%rsp)
	movq	%rax, 48(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	32(%rsp), %rdi
	leaq	16(%rsp), %rsi
	leaq	8(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	32(%rsp), %rsi
	movl	40(%rsp), %edx
	movq	16(%rsp), %rcx
	movl	24(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z27test_scaled_intra_conflictsPKDF16_Pf, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB5_15:                               # %kcall.end91
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB5_18
# %bb.16:                               # %if.end100
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$14, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	80(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB5_18
# %bb.17:                               # %if.end110
	movl	$_ZSt4cout, %edi
	movl	$.L.str.23, %esi
	movl	$22, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.24, %esi
	movl	$12, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.25, %esi
	movl	$88, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.26, %esi
	movl	$16, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.27, %esi
	movl	$61, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.28, %esi
	movl	$44, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.29, %esi
	movl	$45, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.30, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	xorl	%eax, %eax
	addq	$88, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB5_18:                               # %if.then
	.cfi_def_cfa_offset 112
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	%eax, %ebx
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %r14
	movl	%ebx, %edi
	callq	hipGetErrorString
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	movl	$1, %edi
	callq	exit
.Lfunc_end5:
	.size	main, .Lfunc_end5-main
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4                               # -- Begin function _GLOBAL__sub_I_test_intra_lane_conflicts.cpp
	.type	_GLOBAL__sub_I_test_intra_lane_conflicts.cpp,@function
_GLOBAL__sub_I_test_intra_lane_conflicts.cpp: # @_GLOBAL__sub_I_test_intra_lane_conflicts.cpp
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit                    # TAILCALL
.Lfunc_end6:
	.size	_GLOBAL__sub_I_test_intra_lane_conflicts.cpp, .Lfunc_end6-_GLOBAL__sub_I_test_intra_lane_conflicts.cpp
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_fad6a0bef469ea87(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB7_2
# %bb.1:                                # %if
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_fad6a0bef469ea87(%rip)
.LBB7_2:                                # %exit
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z33test_intra_lane_transpose_patternPKDF16_Pf, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z28test_intra_lane_no_conflictsPKDF16_Pf, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z24test_full_phase0_patternPKDF16_Pf, %esi
	movl	$.L__unnamed_4, %edx
	movl	$.L__unnamed_4, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z27test_scaled_intra_conflictsPKDF16_Pf, %esi
	movl	$.L__unnamed_5, %edx
	movl	$.L__unnamed_5, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end7:
	.size	__hip_module_ctor, .Lfunc_end7-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:                                # %entry
	movq	__hip_gpubin_handle_fad6a0bef469ea87(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB8_2
# %bb.1:                                # %if
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_fad6a0bef469ea87(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB8_2:                                # %exit
	retq
.Lfunc_end8:
	.size	__hip_module_dtor, .Lfunc_end8-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	_Z33test_intra_lane_transpose_patternPKDF16_Pf,@object # @_Z33test_intra_lane_transpose_patternPKDF16_Pf
	.section	.rodata,"a",@progbits
	.globl	_Z33test_intra_lane_transpose_patternPKDF16_Pf
	.p2align	3, 0x0
_Z33test_intra_lane_transpose_patternPKDF16_Pf:
	.quad	_Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf
	.size	_Z33test_intra_lane_transpose_patternPKDF16_Pf, 8

	.type	_Z28test_intra_lane_no_conflictsPKDF16_Pf,@object # @_Z28test_intra_lane_no_conflictsPKDF16_Pf
	.globl	_Z28test_intra_lane_no_conflictsPKDF16_Pf
	.p2align	3, 0x0
_Z28test_intra_lane_no_conflictsPKDF16_Pf:
	.quad	_Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf
	.size	_Z28test_intra_lane_no_conflictsPKDF16_Pf, 8

	.type	_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf,@object # @_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.globl	_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.p2align	3, 0x0
_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf:
	.quad	_Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.size	_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf, 8

	.type	_Z24test_full_phase0_patternPKDF16_Pf,@object # @_Z24test_full_phase0_patternPKDF16_Pf
	.globl	_Z24test_full_phase0_patternPKDF16_Pf
	.p2align	3, 0x0
_Z24test_full_phase0_patternPKDF16_Pf:
	.quad	_Z39__device_stub__test_full_phase0_patternPKDF16_Pf
	.size	_Z24test_full_phase0_patternPKDF16_Pf, 8

	.type	_Z27test_scaled_intra_conflictsPKDF16_Pf,@object # @_Z27test_scaled_intra_conflictsPKDF16_Pf
	.globl	_Z27test_scaled_intra_conflictsPKDF16_Pf
	.p2align	3, 0x0
_Z27test_scaled_intra_conflictsPKDF16_Pf:
	.quad	_Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf
	.size	_Z27test_scaled_intra_conflictsPKDF16_Pf, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"HIP Error: "
	.size	.L.str, 12

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"=== INTRA-LANE CONFLICT TESTS ===\n\n"
	.size	.L.str.1, 36

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"Test 1: ONE thread, transpose pattern (column k=0)\n"
	.size	.L.str.2, 52

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"  Pattern: Thread 0 reads m=[0-7], k=0\n"
	.size	.L.str.3, 40

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"  Banks: {0, 16, 0, 16, 0, 16, 0, 16}\n"
	.size	.L.str.4, 39

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"  Slots: {0, 16, 32, 48, 64, 80, 96, 112} (different slots!)\n"
	.size	.L.str.5, 62

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"  Expected: INTRA-lane conflicts (bank 0: 4 slots, bank 16: 4 slots)\n"
	.size	.L.str.6, 70

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"  Completed.\n\n"
	.size	.L.str.7, 15

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"Test 2: ONE thread, different banks (baseline)\n"
	.size	.L.str.8, 48

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"  Pattern: Thread 0 reads offsets {0,2,4,6,8,10,12,14}\n"
	.size	.L.str.9, 56

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"  Banks: {0,1,2,3,4,5,6,7} (all different)\n"
	.size	.L.str.10, 44

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"  Expected: 0 conflicts\n"
	.size	.L.str.11, 25

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"Test 3: ONE thread, same bank, different slots\n"
	.size	.L.str.12, 48

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"  Pattern: Thread 0 reads offsets {0,64,128,192,256,320,384,448}\n"
	.size	.L.str.13, 66

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"  Banks: All bank 0, slots {0,32,64,96,128,160,192,224}\n"
	.size	.L.str.14, 57

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"  Expected: HIGH intra-lane conflicts (8 different slots)\n"
	.size	.L.str.15, 59

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"Test 4: Full Phase 0 pattern (8 lanes)\n"
	.size	.L.str.16, 40

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"  Pattern: Lanes {0,1,2,3,20,21,22,23} each read their column\n"
	.size	.L.str.17, 63

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	"  Each lane has intra-lane pattern like Test 1\n"
	.size	.L.str.18, 48

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"  Expected: 8 lanes \303\227 intra-lane conflicts\n"
	.size	.L.str.19, 45

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"Test 5: Scaled - 32 threads each with transpose pattern\n"
	.size	.L.str.20, 57

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	"  Pattern: 32 threads (k=0-31) each read 8 M values\n"
	.size	.L.str.21, 53

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"  Expected: 32 \303\227 intra-lane conflicts\n"
	.size	.L.str.22, 40

	.type	.L.str.23,@object               # @.str.23
.L.str.23:
	.asciz	"All tests completed.\n\n"
	.size	.L.str.23, 23

	.type	.L.str.24,@object               # @.str.24
.L.str.24:
	.asciz	"To profile:\n"
	.size	.L.str.24, 13

	.type	.L.str.25,@object               # @.str.25
.L.str.25:
	.asciz	"  rocprofv3 -i lds_conflict.txt -d intra_results -f csv -- ./test_intra_lane_conflicts\n\n"
	.size	.L.str.25, 89

	.type	.L.str.26,@object               # @.str.26
.L.str.26:
	.asciz	"KEY HYPOTHESIS:\n"
	.size	.L.str.26, 17

	.type	.L.str.27,@object               # @.str.27
.L.str.27:
	.asciz	"  Test 1 should show conflicts (intra-lane, different slots)\n"
	.size	.L.str.27, 62

	.type	.L.str.28,@object               # @.str.28
.L.str.28:
	.asciz	"  Test 4 should show 8\303\227 Test 1's conflicts\n"
	.size	.L.str.28, 45

	.type	.L.str.29,@object               # @.str.29
.L.str.29:
	.asciz	"  Test 5 should show 32\303\227 Test 1's conflicts\n"
	.size	.L.str.29, 46

	.type	.L.str.30,@object               # @.str.30
.L.str.30:
	.asciz	"  This is where the 7,168 conflicts come from!\n"
	.size	.L.str.30, 48

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z33test_intra_lane_transpose_patternPKDF16_Pf"
	.size	.L__unnamed_1, 47

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z28test_intra_lane_no_conflictsPKDF16_Pf"
	.size	.L__unnamed_2, 42

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf"
	.size	.L__unnamed_3, 50

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z24test_full_phase0_patternPKDF16_Pf"
	.size	.L__unnamed_4, 38

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_Z27test_scaled_intra_conflictsPKDF16_Pf"
	.size	.L__unnamed_5, 41

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_fad6a0bef469ea87
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_fad6a0bef469ea87,@object # @__hip_gpubin_handle_fad6a0bef469ea87
	.local	__hip_gpubin_handle_fad6a0bef469ea87
	.comm	__hip_gpubin_handle_fad6a0bef469ea87,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_test_intra_lane_conflicts.cpp
	.quad	__hip_module_ctor
	.type	__hip_cuid_fad6a0bef469ea87,@object # @__hip_cuid_fad6a0bef469ea87
	.bss
	.globl	__hip_cuid_fad6a0bef469ea87
__hip_cuid_fad6a0bef469ea87:
	.byte	0                               # 0x0
	.size	__hip_cuid_fad6a0bef469ea87, 1

	.ident	"AMD clang version 21.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25401 965357120e93d691c2c2f6b221deb863caf44a62)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z48__device_stub__test_intra_lane_transpose_patternPKDF16_Pf
	.addrsig_sym _Z43__device_stub__test_intra_lane_no_conflictsPKDF16_Pf
	.addrsig_sym _Z51__device_stub__test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.addrsig_sym _Z39__device_stub__test_full_phase0_patternPKDF16_Pf
	.addrsig_sym _Z42__device_stub__test_scaled_intra_conflictsPKDF16_Pf
	.addrsig_sym _GLOBAL__sub_I_test_intra_lane_conflicts.cpp
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _Z33test_intra_lane_transpose_patternPKDF16_Pf
	.addrsig_sym _Z28test_intra_lane_no_conflictsPKDF16_Pf
	.addrsig_sym _Z36test_intra_lane_same_bank_diff_slotsPKDF16_Pf
	.addrsig_sym _Z24test_full_phase0_patternPKDF16_Pf
	.addrsig_sym _Z27test_scaled_intra_conflictsPKDF16_Pf
	.addrsig_sym _ZSt4cerr
	.addrsig_sym _ZSt4cout
	.addrsig_sym __hip_fatbin_fad6a0bef469ea87
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_fad6a0bef469ea87

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
