
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx942
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z26test_same_slot_all_threadsPf ; -- Begin function _Z26test_same_slot_all_threadsPf
	.globl	_Z26test_same_slot_all_threadsPf
	.p2align	8
	.type	_Z26test_same_slot_all_threadsPf,@function
_Z26test_same_slot_all_threadsPf:       ; @_Z26test_same_slot_all_threadsPf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x14
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
; %bb.4:                                ; %Flow31
	s_or_b64 exec, exec, s[6:7]
.LBB0_5:                                ; %Flow32
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
.LBB0_8:                                ; %Flow30
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB0_9:                                ; %Flow33
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
.LBB0_12:                               ; %Flow34
	s_or_b64 exec, exec, s[2:3]
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v1, v1
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v1, v1
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z26test_same_slot_all_threadsPf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
	.size	_Z26test_same_slot_all_threadsPf, .Lfunc_end0-_Z26test_same_slot_all_threadsPf
                                        ; -- End function
	.set _Z26test_same_slot_all_threadsPf.num_vgpr, 30
	.set _Z26test_same_slot_all_threadsPf.num_agpr, 0
	.set _Z26test_same_slot_all_threadsPf.numbered_sgpr, 14
	.set _Z26test_same_slot_all_threadsPf.private_seg_size, 0
	.set _Z26test_same_slot_all_threadsPf.uses_vcc, 1
	.set _Z26test_same_slot_all_threadsPf.uses_flat_scratch, 0
	.set _Z26test_same_slot_all_threadsPf.has_dyn_sized_stack, 0
	.set _Z26test_same_slot_all_threadsPf.has_recursion, 0
	.set _Z26test_same_slot_all_threadsPf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 952
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
	.protected	_Z37test_same_bank_diff_slots_all_threadsPf ; -- Begin function _Z37test_same_bank_diff_slots_all_threadsPf
	.globl	_Z37test_same_bank_diff_slots_all_threadsPf
	.p2align	8
	.type	_Z37test_same_bank_diff_slots_all_threadsPf,@function
_Z37test_same_bank_diff_slots_all_threadsPf: ; @_Z37test_same_bank_diff_slots_all_threadsPf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x14
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
; %bb.4:                                ; %Flow34
	s_or_b64 exec, exec, s[6:7]
.LBB1_5:                                ; %Flow35
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
.LBB1_8:                                ; %Flow33
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB1_9:                                ; %Flow36
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
.LBB1_12:                               ; %Flow37
	s_or_b64 exec, exec, s[2:3]
	v_lshlrev_b32_e32 v1, 7, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v1, v1
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v1, v1
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z37test_same_bank_diff_slots_all_threadsPf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
	.size	_Z37test_same_bank_diff_slots_all_threadsPf, .Lfunc_end1-_Z37test_same_bank_diff_slots_all_threadsPf
                                        ; -- End function
	.set _Z37test_same_bank_diff_slots_all_threadsPf.num_vgpr, 30
	.set _Z37test_same_bank_diff_slots_all_threadsPf.num_agpr, 0
	.set _Z37test_same_bank_diff_slots_all_threadsPf.numbered_sgpr, 14
	.set _Z37test_same_bank_diff_slots_all_threadsPf.private_seg_size, 0
	.set _Z37test_same_bank_diff_slots_all_threadsPf.uses_vcc, 1
	.set _Z37test_same_bank_diff_slots_all_threadsPf.uses_flat_scratch, 0
	.set _Z37test_same_bank_diff_slots_all_threadsPf.has_dyn_sized_stack, 0
	.set _Z37test_same_bank_diff_slots_all_threadsPf.has_recursion, 0
	.set _Z37test_same_bank_diff_slots_all_threadsPf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 952
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
	.protected	_Z24test_one_phase_same_bankPf ; -- Begin function _Z24test_one_phase_same_bankPf
	.globl	_Z24test_one_phase_same_bankPf
	.p2align	8
	.type	_Z24test_one_phase_same_bankPf,@function
_Z24test_one_phase_same_bankPf:         ; @_Z24test_one_phase_same_bankPf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x14
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
; %bb.4:                                ; %Flow36
	s_or_b64 exec, exec, s[6:7]
.LBB2_5:                                ; %Flow37
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
.LBB2_8:                                ; %Flow35
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB2_9:                                ; %Flow38
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
.LBB2_12:                               ; %Flow39
	s_or_b64 exec, exec, s[2:3]
	v_cmp_gt_u32_e32 vcc, 8, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_14
; %bb.13:                               ; %if.then
	v_lshlrev_b32_e32 v1, 7, v0
	ds_read_u16 v1, v1
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v1, v1
	global_store_dword v0, v1, s[0:1]
.LBB2_14:                               ; %if.end
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z24test_one_phase_same_bankPf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
	.size	_Z24test_one_phase_same_bankPf, .Lfunc_end2-_Z24test_one_phase_same_bankPf
                                        ; -- End function
	.set _Z24test_one_phase_same_bankPf.num_vgpr, 30
	.set _Z24test_one_phase_same_bankPf.num_agpr, 0
	.set _Z24test_one_phase_same_bankPf.numbered_sgpr, 14
	.set _Z24test_one_phase_same_bankPf.private_seg_size, 0
	.set _Z24test_one_phase_same_bankPf.uses_vcc, 1
	.set _Z24test_one_phase_same_bankPf.uses_flat_scratch, 0
	.set _Z24test_one_phase_same_bankPf.has_dyn_sized_stack, 0
	.set _Z24test_one_phase_same_bankPf.has_recursion, 0
	.set _Z24test_one_phase_same_bankPf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 964
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
	.protected	_Z29test_many_intra_lane_patternsPf ; -- Begin function _Z29test_many_intra_lane_patternsPf
	.globl	_Z29test_many_intra_lane_patternsPf
	.p2align	8
	.type	_Z29test_many_intra_lane_patternsPf,@function
_Z29test_many_intra_lane_patternsPf:    ; @_Z29test_many_intra_lane_patternsPf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x14
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
; %bb.4:                                ; %Flow63
	s_or_b64 exec, exec, s[6:7]
.LBB3_5:                                ; %Flow64
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
.LBB3_8:                                ; %Flow62
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB3_9:                                ; %Flow65
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
.LBB3_12:                               ; %Flow66
	s_or_b64 exec, exec, s[2:3]
	v_and_b32_e32 v1, 31, v0
	v_lshlrev_b32_e32 v1, 1, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v2, v1
	ds_read_u16 v3, v1 offset:128
	ds_read_u16 v4, v1 offset:256
	ds_read_u16 v5, v1 offset:384
	ds_read_u16 v6, v1 offset:512
	ds_read_u16 v7, v1 offset:640
	ds_read_u16 v8, v1 offset:768
	ds_read_u16 v1, v1 offset:896
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
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v1, v1
	v_add_f32_e32 v2, v2, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v1, v2, v1
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z29test_many_intra_lane_patternsPf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
	.size	_Z29test_many_intra_lane_patternsPf, .Lfunc_end3-_Z29test_many_intra_lane_patternsPf
                                        ; -- End function
	.set _Z29test_many_intra_lane_patternsPf.num_vgpr, 30
	.set _Z29test_many_intra_lane_patternsPf.num_agpr, 0
	.set _Z29test_many_intra_lane_patternsPf.numbered_sgpr, 14
	.set _Z29test_many_intra_lane_patternsPf.private_seg_size, 0
	.set _Z29test_many_intra_lane_patternsPf.uses_vcc, 1
	.set _Z29test_many_intra_lane_patternsPf.uses_flat_scratch, 0
	.set _Z29test_many_intra_lane_patternsPf.has_dyn_sized_stack, 0
	.set _Z29test_many_intra_lane_patternsPf.has_recursion, 0
	.set _Z29test_many_intra_lane_patternsPf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1100
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
	.protected	_Z25test_exact_kernel_patternPf ; -- Begin function _Z25test_exact_kernel_patternPf
	.globl	_Z25test_exact_kernel_patternPf
	.p2align	8
	.type	_Z25test_exact_kernel_patternPf,@function
_Z25test_exact_kernel_patternPf:        ; @_Z25test_exact_kernel_patternPf
; %bb.0:                                ; %entry
	s_load_dword s2, s[0:1], 0x14
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
; %bb.4:                                ; %Flow79
	s_or_b64 exec, exec, s[6:7]
.LBB4_5:                                ; %Flow80
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
.LBB4_8:                                ; %Flow78
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB4_9:                                ; %Flow81
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
.LBB4_12:                               ; %Flow82
	s_or_b64 exec, exec, s[2:3]
	v_and_b32_e32 v1, 7, v0
	v_lshrrev_b32_e32 v3, 3, v0
	s_movk_i32 s2, 0x78
	v_and_or_b32 v1, v3, s2, v1
	v_and_b32_e32 v2, 56, v0
	v_lshlrev_b32_e32 v1, 1, v1
	v_lshl_or_b32 v1, v2, 6, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
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
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v1, v1
	v_add_f32_e32 v2, v2, v3
	v_add_f32_e32 v2, v2, v4
	v_add_f32_e32 v2, v2, v5
	v_add_f32_e32 v1, v2, v1
	v_lshlrev_b32_e32 v0, 2, v0
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z25test_exact_kernel_patternPf
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
	.size	_Z25test_exact_kernel_patternPf, .Lfunc_end4-_Z25test_exact_kernel_patternPf
                                        ; -- End function
	.set _Z25test_exact_kernel_patternPf.num_vgpr, 30
	.set _Z25test_exact_kernel_patternPf.num_agpr, 0
	.set _Z25test_exact_kernel_patternPf.numbered_sgpr, 14
	.set _Z25test_exact_kernel_patternPf.private_seg_size, 0
	.set _Z25test_exact_kernel_patternPf.uses_vcc, 1
	.set _Z25test_exact_kernel_patternPf.uses_flat_scratch, 0
	.set _Z25test_exact_kernel_patternPf.has_dyn_sized_stack, 0
	.set _Z25test_exact_kernel_patternPf.has_recursion, 0
	.set _Z25test_exact_kernel_patternPf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1128
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
	.protected	_Z27test_repeated_exact_patternPfi ; -- Begin function _Z27test_repeated_exact_patternPfi
	.globl	_Z27test_repeated_exact_patternPfi
	.p2align	8
	.type	_Z27test_repeated_exact_patternPfi,@function
_Z27test_repeated_exact_patternPfi:     ; @_Z27test_repeated_exact_patternPfi
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
	s_cbranch_execz .LBB5_9
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
	s_cbranch_execz .LBB5_5
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
.LBB5_3:                                ; %vector.body
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
	s_cbranch_execnz .LBB5_3
; %bb.4:                                ; %Flow100
	s_or_b64 exec, exec, s[6:7]
.LBB5_5:                                ; %Flow101
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, 3, v9
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB5_8
; %bb.6:                                ; %vector.body.epil.preheader
	v_mul_lo_u32 v2, v12, s8
	v_add_lshl_u32 v2, v0, v2, 1
	s_lshl_b32 s9, s8, 3
	s_mov_b64 s[6:7], 0
.LBB5_7:                                ; %vector.body.epil
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
	s_cbranch_execnz .LBB5_7
.LBB5_8:                                ; %Flow99
	s_or_b64 exec, exec, s[4:5]
	v_and_b32_e32 v1, -4, v8
	v_mad_u64_u32 v[2:3], s[4:5], v1, s8, v[0:1]
	v_cmp_ne_u32_e32 vcc, v8, v1
	s_orn2_b64 s[4:5], vcc, exec
.LBB5_9:                                ; %Flow102
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB5_12
; %bb.10:                               ; %for.body.preheader
	v_lshlrev_b32_e32 v1, 1, v2
	s_lshl_b32 s6, s8, 1
	s_mov_b64 s[4:5], 0
	s_movk_i32 s7, 0x7ff
.LBB5_11:                               ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_cvt_f32_u32_e32 v3, v2
	v_add_u32_e32 v2, s8, v2
	v_cmp_lt_u32_e32 vcc, s7, v2
	s_or_b64 s[4:5], vcc, s[4:5]
	v_cvt_f16_f32_e32 v3, v3
	ds_write_b16 v1, v3
	v_add_u32_e32 v1, s6, v1
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB5_11
.LBB5_12:                               ; %Flow103
	s_or_b64 exec, exec, s[2:3]
	s_load_dword s2, s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB5_15
; %bb.13:                               ; %for.cond12.preheader.preheader
	v_lshrrev_b32_e32 v1, 3, v0
	v_and_b32_e32 v2, 7, v0
	s_movk_i32 s3, 0x78
	v_and_or_b32 v1, v1, s3, v2
	v_and_b32_e32 v2, 56, v0
	v_lshlrev_b32_e32 v1, 1, v1
	v_lshl_or_b32 v2, v2, 6, v1
	v_mov_b32_e32 v1, 0
.LBB5_14:                               ; %for.cond12.preheader
                                        ; =>This Inner Loop Header: Depth=1
	ds_read_u16 v3, v2
	ds_read_u16 v4, v2 offset:64
	ds_read_u16 v5, v2 offset:128
	ds_read_u16 v6, v2 offset:192
	ds_read_u16 v7, v2 offset:256
	ds_read_u16 v8, v2 offset:320
	ds_read_u16 v9, v2 offset:384
	ds_read_u16 v10, v2 offset:448
	s_waitcnt lgkmcnt(7)
	v_cvt_f32_f16_e32 v3, v3
	s_waitcnt lgkmcnt(6)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt lgkmcnt(5)
	v_cvt_f32_f16_e32 v5, v5
	s_waitcnt lgkmcnt(4)
	v_cvt_f32_f16_e32 v6, v6
	s_waitcnt lgkmcnt(3)
	v_cvt_f32_f16_e32 v7, v7
	v_add_f32_e32 v1, v1, v3
	s_waitcnt lgkmcnt(2)
	v_cvt_f32_f16_e32 v8, v8
	v_add_f32_e32 v1, v1, v4
	s_waitcnt lgkmcnt(1)
	v_cvt_f32_f16_e32 v9, v9
	v_add_f32_e32 v1, v1, v5
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v10, v10
	v_add_f32_e32 v1, v1, v6
	v_add_f32_e32 v1, v1, v7
	v_add_f32_e32 v1, v1, v8
	s_add_i32 s2, s2, -1
	v_add_f32_e32 v1, v1, v9
	s_cmp_lg_u32 s2, 0
	v_add_f32_e32 v1, v1, v10
	s_barrier
	s_cbranch_scc1 .LBB5_14
	s_branch .LBB5_16
.LBB5_15:
	v_mov_b32_e32 v1, 0
.LBB5_16:                               ; %for.cond.cleanup9
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_dword v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z27test_repeated_exact_patternPfi
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
.Lfunc_end5:
	.size	_Z27test_repeated_exact_patternPfi, .Lfunc_end5-_Z27test_repeated_exact_patternPfi
                                        ; -- End function
	.set _Z27test_repeated_exact_patternPfi.num_vgpr, 30
	.set _Z27test_repeated_exact_patternPfi.num_agpr, 0
	.set _Z27test_repeated_exact_patternPfi.numbered_sgpr, 14
	.set _Z27test_repeated_exact_patternPfi.private_seg_size, 0
	.set _Z27test_repeated_exact_patternPfi.uses_vcc, 1
	.set _Z27test_repeated_exact_patternPfi.uses_flat_scratch, 0
	.set _Z27test_repeated_exact_patternPfi.has_dyn_sized_stack, 0
	.set _Z27test_repeated_exact_patternPfi.has_recursion, 0
	.set _Z27test_repeated_exact_patternPfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1176
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
	.type	__hip_cuid_3b2d6147d262d257,@object ; @__hip_cuid_3b2d6147d262d257
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_3b2d6147d262d257
__hip_cuid_3b2d6147d262d257:
	.byte	0                               ; 0x0
	.size	__hip_cuid_3b2d6147d262d257, 1

	.ident	"AMD clang version 21.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25401 965357120e93d691c2c2f6b221deb863caf44a62)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_3b2d6147d262d257
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           output.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z26test_same_slot_all_threadsPf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z26test_same_slot_all_threadsPf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           output.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z37test_same_bank_diff_slots_all_threadsPf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z37test_same_bank_diff_slots_all_threadsPf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           output.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z24test_one_phase_same_bankPf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z24test_one_phase_same_bankPf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           output.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z29test_many_intra_lane_patternsPf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z29test_many_intra_lane_patternsPf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           output.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z25test_exact_kernel_patternPf
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z25test_exact_kernel_patternPf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     30
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           output.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           iterations
        .offset:         8
        .size:           4
        .value_kind:     by_value
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
    .name:           _Z27test_repeated_exact_patternPfi
    .private_segment_fixed_size: 0
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         _Z27test_repeated_exact_patternPfi.kd
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
	.file	"test_multi_thread_conflict.cpp"
	.text
	.globl	_Z41__device_stub__test_same_slot_all_threadsPf # -- Begin function _Z41__device_stub__test_same_slot_all_threadsPf
	.p2align	4
	.type	_Z41__device_stub__test_same_slot_all_threadsPf,@function
_Z41__device_stub__test_same_slot_all_threadsPf: # @_Z41__device_stub__test_same_slot_all_threadsPf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$72, %rsp
	.cfi_def_cfa_offset 80
	movq	%rdi, 64(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z26test_same_slot_all_threadsPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$88, %rsp
	.cfi_adjust_cfa_offset -88
	retq
.Lfunc_end0:
	.size	_Z41__device_stub__test_same_slot_all_threadsPf, .Lfunc_end0-_Z41__device_stub__test_same_slot_all_threadsPf
	.cfi_endproc
                                        # -- End function
	.globl	_Z52__device_stub__test_same_bank_diff_slots_all_threadsPf # -- Begin function _Z52__device_stub__test_same_bank_diff_slots_all_threadsPf
	.p2align	4
	.type	_Z52__device_stub__test_same_bank_diff_slots_all_threadsPf,@function
_Z52__device_stub__test_same_bank_diff_slots_all_threadsPf: # @_Z52__device_stub__test_same_bank_diff_slots_all_threadsPf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$72, %rsp
	.cfi_def_cfa_offset 80
	movq	%rdi, 64(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z37test_same_bank_diff_slots_all_threadsPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$88, %rsp
	.cfi_adjust_cfa_offset -88
	retq
.Lfunc_end1:
	.size	_Z52__device_stub__test_same_bank_diff_slots_all_threadsPf, .Lfunc_end1-_Z52__device_stub__test_same_bank_diff_slots_all_threadsPf
	.cfi_endproc
                                        # -- End function
	.globl	_Z39__device_stub__test_one_phase_same_bankPf # -- Begin function _Z39__device_stub__test_one_phase_same_bankPf
	.p2align	4
	.type	_Z39__device_stub__test_one_phase_same_bankPf,@function
_Z39__device_stub__test_one_phase_same_bankPf: # @_Z39__device_stub__test_one_phase_same_bankPf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$72, %rsp
	.cfi_def_cfa_offset 80
	movq	%rdi, 64(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z24test_one_phase_same_bankPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$88, %rsp
	.cfi_adjust_cfa_offset -88
	retq
.Lfunc_end2:
	.size	_Z39__device_stub__test_one_phase_same_bankPf, .Lfunc_end2-_Z39__device_stub__test_one_phase_same_bankPf
	.cfi_endproc
                                        # -- End function
	.globl	_Z44__device_stub__test_many_intra_lane_patternsPf # -- Begin function _Z44__device_stub__test_many_intra_lane_patternsPf
	.p2align	4
	.type	_Z44__device_stub__test_many_intra_lane_patternsPf,@function
_Z44__device_stub__test_many_intra_lane_patternsPf: # @_Z44__device_stub__test_many_intra_lane_patternsPf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$72, %rsp
	.cfi_def_cfa_offset 80
	movq	%rdi, 64(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z29test_many_intra_lane_patternsPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$88, %rsp
	.cfi_adjust_cfa_offset -88
	retq
.Lfunc_end3:
	.size	_Z44__device_stub__test_many_intra_lane_patternsPf, .Lfunc_end3-_Z44__device_stub__test_many_intra_lane_patternsPf
	.cfi_endproc
                                        # -- End function
	.globl	_Z40__device_stub__test_exact_kernel_patternPf # -- Begin function _Z40__device_stub__test_exact_kernel_patternPf
	.p2align	4
	.type	_Z40__device_stub__test_exact_kernel_patternPf,@function
_Z40__device_stub__test_exact_kernel_patternPf: # @_Z40__device_stub__test_exact_kernel_patternPf
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$72, %rsp
	.cfi_def_cfa_offset 80
	movq	%rdi, 64(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z25test_exact_kernel_patternPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$88, %rsp
	.cfi_adjust_cfa_offset -88
	retq
.Lfunc_end4:
	.size	_Z40__device_stub__test_exact_kernel_patternPf, .Lfunc_end4-_Z40__device_stub__test_exact_kernel_patternPf
	.cfi_endproc
                                        # -- End function
	.globl	_Z42__device_stub__test_repeated_exact_patternPfi # -- Begin function _Z42__device_stub__test_repeated_exact_patternPfi
	.p2align	4
	.type	_Z42__device_stub__test_repeated_exact_patternPfi,@function
_Z42__device_stub__test_repeated_exact_patternPfi: # @_Z42__device_stub__test_repeated_exact_patternPfi
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$88, %rsp
	.cfi_def_cfa_offset 96
	movq	%rdi, 56(%rsp)
	movl	%esi, 4(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 64(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	64(%rsp), %r9
	movl	$_Z27test_repeated_exact_patternPfi, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$104, %rsp
	.cfi_adjust_cfa_offset -104
	retq
.Lfunc_end5:
	.size	_Z42__device_stub__test_repeated_exact_patternPfi, .Lfunc_end5-_Z42__device_stub__test_repeated_exact_patternPfi
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
	subq	$104, %rsp
	.cfi_def_cfa_offset 128
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	leaq	88(%rsp), %rdi
	movl	$1024, %esi                     # imm = 0x400
	callq	hipMalloc
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.1:                                # %if.end
	movabsq	$4294967297, %rbx               # imm = 0x100000001
	movl	$_ZSt4cout, %edi
	movl	$.L.str.1, %esi
	movl	$37, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.2, %esi
	movl	$55, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.3, %esi
	movl	$54, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	leaq	63(%rbx), %r14
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB6_3
# %bb.2:                                # %kcall.configok
	movq	88(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z26test_same_slot_all_threadsPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_3:                                # %kcall.end
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.4:                                # %if.end18
	movl	$_ZSt4cout, %edi
	movl	$.L.str.4, %esi
	movl	$55, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.5, %esi
	movl	$54, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.6, %esi
	movl	$50, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB6_6
# %bb.5:                                # %kcall.configok26
	movq	88(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z37test_same_bank_diff_slots_all_threadsPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_6:                                # %kcall.end27
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.7:                                # %if.end36
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$57, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.8, %esi
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
	jne	.LBB6_9
# %bb.8:                                # %kcall.configok43
	movq	88(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z24test_one_phase_same_bankPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_9:                                # %kcall.end44
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.10:                               # %if.end53
	movl	$_ZSt4cout, %edi
	movl	$.L.str.9, %esi
	movl	$56, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.10, %esi
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
	jne	.LBB6_12
# %bb.11:                               # %kcall.configok60
	movq	88(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z29test_many_intra_lane_patternsPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_12:                               # %kcall.end61
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.13:                               # %if.end70
	movl	$_ZSt4cout, %edi
	movl	$.L.str.11, %esi
	movl	$51, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.12, %esi
	movl	$47, %edx
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
	jne	.LBB6_15
# %bb.14:                               # %kcall.configok77
	movq	88(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z25test_exact_kernel_patternPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_15:                               # %kcall.end78
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.16:                               # %if.end87
	movl	$_ZSt4cout, %edi
	movl	$.L.str.13, %esi
	movl	$59, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.14, %esi
	movl	$33, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB6_18
# %bb.17:                               # %kcall.configok94
	movq	88(%rsp), %rax
	movq	%rax, 40(%rsp)
	movl	$4, 100(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 48(%rsp)
	leaq	100(%rsp), %rax
	movq	%rax, 56(%rsp)
	leaq	24(%rsp), %rdi
	leaq	72(%rsp), %rsi
	leaq	16(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	24(%rsp), %rsi
	movl	32(%rsp), %edx
	movq	72(%rsp), %rcx
	movl	80(%rsp), %r8d
	leaq	48(%rsp), %r9
	movl	$_Z27test_repeated_exact_patternPfi, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_18:                               # %kcall.end95
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.19:                               # %if.end104
	movl	$_ZSt4cout, %edi
	movl	$.L.str.15, %esi
	movl	$55, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.14, %esi
	movl	$33, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	addq	$3, %rbx
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB6_21
# %bb.20:                               # %kcall.configok111
	movq	88(%rsp), %rax
	movq	%rax, 72(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, (%rsp)
	leaq	48(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	40(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	movq	%rsp, %r9
	movl	$_Z25test_exact_kernel_patternPf, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_21:                               # %kcall.end112
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.22:                               # %if.end121
	movl	$_ZSt4cout, %edi
	movl	$.L.str.16, %esi
	movl	$50, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.17, %esi
	movl	$55, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1, %esi
	movq	%r14, %rdx
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	jne	.LBB6_24
# %bb.23:                               # %kcall.configok128
	movq	88(%rsp), %rax
	movq	%rax, 40(%rsp)
	movl	$4, 100(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 48(%rsp)
	leaq	100(%rsp), %rax
	movq	%rax, 56(%rsp)
	leaq	24(%rsp), %rdi
	leaq	72(%rsp), %rsi
	leaq	16(%rsp), %rdx
	movq	%rsp, %rcx
	callq	__hipPopCallConfiguration
	movq	24(%rsp), %rsi
	movl	32(%rsp), %edx
	movq	72(%rsp), %rcx
	movl	80(%rsp), %r8d
	leaq	48(%rsp), %r9
	movl	$_Z27test_repeated_exact_patternPfi, %edi
	pushq	(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.LBB6_24:                               # %kcall.end129
	callq	hipDeviceSynchronize
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.25:                               # %if.end138
	movq	88(%rsp), %rdi
	callq	hipFree
	testl	%eax, %eax
	jne	.LBB6_27
# %bb.26:                               # %if.end147
	movl	$_ZSt4cout, %edi
	movl	$.L.str.18, %esi
	movl	$22, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.19, %esi
	movl	$83, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	xorl	%eax, %eax
	addq	$104, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB6_27:                               # %if.then
	.cfi_def_cfa_offset 128
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
.Lfunc_end6:
	.size	main, .Lfunc_end6-main
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4                               # -- Begin function _GLOBAL__sub_I_test_multi_thread_conflict.cpp
	.type	_GLOBAL__sub_I_test_multi_thread_conflict.cpp,@function
_GLOBAL__sub_I_test_multi_thread_conflict.cpp: # @_GLOBAL__sub_I_test_multi_thread_conflict.cpp
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
.Lfunc_end7:
	.size	_GLOBAL__sub_I_test_multi_thread_conflict.cpp, .Lfunc_end7-_GLOBAL__sub_I_test_multi_thread_conflict.cpp
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
	movq	__hip_gpubin_handle_3b2d6147d262d257(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB8_2
# %bb.1:                                # %if
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_3b2d6147d262d257(%rip)
.LBB8_2:                                # %exit
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z26test_same_slot_all_threadsPf, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z37test_same_bank_diff_slots_all_threadsPf, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z24test_one_phase_same_bankPf, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z29test_many_intra_lane_patternsPf, %esi
	movl	$.L__unnamed_4, %edx
	movl	$.L__unnamed_4, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z25test_exact_kernel_patternPf, %esi
	movl	$.L__unnamed_5, %edx
	movl	$.L__unnamed_5, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z27test_repeated_exact_patternPfi, %esi
	movl	$.L__unnamed_6, %edx
	movl	$.L__unnamed_6, %ecx
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
.Lfunc_end8:
	.size	__hip_module_ctor, .Lfunc_end8-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:                                # %entry
	movq	__hip_gpubin_handle_3b2d6147d262d257(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB9_2
# %bb.1:                                # %if
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_3b2d6147d262d257(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB9_2:                                # %exit
	retq
.Lfunc_end9:
	.size	__hip_module_dtor, .Lfunc_end9-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	_Z26test_same_slot_all_threadsPf,@object # @_Z26test_same_slot_all_threadsPf
	.section	.rodata,"a",@progbits
	.globl	_Z26test_same_slot_all_threadsPf
	.p2align	3, 0x0
_Z26test_same_slot_all_threadsPf:
	.quad	_Z41__device_stub__test_same_slot_all_threadsPf
	.size	_Z26test_same_slot_all_threadsPf, 8

	.type	_Z37test_same_bank_diff_slots_all_threadsPf,@object # @_Z37test_same_bank_diff_slots_all_threadsPf
	.globl	_Z37test_same_bank_diff_slots_all_threadsPf
	.p2align	3, 0x0
_Z37test_same_bank_diff_slots_all_threadsPf:
	.quad	_Z52__device_stub__test_same_bank_diff_slots_all_threadsPf
	.size	_Z37test_same_bank_diff_slots_all_threadsPf, 8

	.type	_Z24test_one_phase_same_bankPf,@object # @_Z24test_one_phase_same_bankPf
	.globl	_Z24test_one_phase_same_bankPf
	.p2align	3, 0x0
_Z24test_one_phase_same_bankPf:
	.quad	_Z39__device_stub__test_one_phase_same_bankPf
	.size	_Z24test_one_phase_same_bankPf, 8

	.type	_Z29test_many_intra_lane_patternsPf,@object # @_Z29test_many_intra_lane_patternsPf
	.globl	_Z29test_many_intra_lane_patternsPf
	.p2align	3, 0x0
_Z29test_many_intra_lane_patternsPf:
	.quad	_Z44__device_stub__test_many_intra_lane_patternsPf
	.size	_Z29test_many_intra_lane_patternsPf, 8

	.type	_Z25test_exact_kernel_patternPf,@object # @_Z25test_exact_kernel_patternPf
	.globl	_Z25test_exact_kernel_patternPf
	.p2align	3, 0x0
_Z25test_exact_kernel_patternPf:
	.quad	_Z40__device_stub__test_exact_kernel_patternPf
	.size	_Z25test_exact_kernel_patternPf, 8

	.type	_Z27test_repeated_exact_patternPfi,@object # @_Z27test_repeated_exact_patternPfi
	.globl	_Z27test_repeated_exact_patternPfi
	.p2align	3, 0x0
_Z27test_repeated_exact_patternPfi:
	.quad	_Z42__device_stub__test_repeated_exact_patternPfi
	.size	_Z27test_repeated_exact_patternPfi, 8

	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"HIP Error: "
	.size	.L.str, 12

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"=== MULTI-THREAD CONFLICT TESTS ===\n\n"
	.size	.L.str.1, 38

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"Test 1: All 64 threads read SAME slot (bank 0, slot 0)\n"
	.size	.L.str.2, 56

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"  Expected: 0 conflicts (FP16 same-slot optimization)\n"
	.size	.L.str.3, 55

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"Test 2: All 64 threads read SAME bank, DIFFERENT slots\n"
	.size	.L.str.4, 56

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"  Pattern: tid*64 -> slots 0,32,64,96... (all bank 0)\n"
	.size	.L.str.5, 55

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"  Expected: HIGH conflicts (64 slots in one bank)\n"
	.size	.L.str.6, 51

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"Test 3: 8 threads (one phase) read same bank, diff slots\n"
	.size	.L.str.7, 58

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"  Expected: conflicts (8 slots in one bank)\n"
	.size	.L.str.8, 45

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"Test 4: 64 threads, each with 8-read intra-lane pattern\n"
	.size	.L.str.9, 57

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"  Expected: Intra-lane conflicts \303\227 64\n"
	.size	.L.str.10, 40

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"Test 5: Exact kernel pattern (256 threads = 4 WFs)\n"
	.size	.L.str.11, 52

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"  Expected: Should match kernel conflict count\n"
	.size	.L.str.12, 48

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"Test 6: Repeated pattern (4 K iterations like real kernel)\n"
	.size	.L.str.13, 60

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"  Expected: 4\303\227 Test 5 conflicts\n"
	.size	.L.str.14, 34

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	"Test 7: 4 blocks \303\227 256 threads (matches M=256 kernel)\n"
	.size	.L.str.15, 56

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"Test 8: 4 blocks \303\227 repeated pattern (full match)\n"
	.size	.L.str.16, 51

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"  Expected: Should match real kernel's 7,168 conflicts\n"
	.size	.L.str.17, 56

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	"\nAll tests completed.\n"
	.size	.L.str.18, 23

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"Profile with: rocprofv3 --pmc SQ_LDS_BANK_CONFLICT -- ./test_multi_thread_conflict\n"
	.size	.L.str.19, 84

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z26test_same_slot_all_threadsPf"
	.size	.L__unnamed_1, 33

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z37test_same_bank_diff_slots_all_threadsPf"
	.size	.L__unnamed_2, 44

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z24test_one_phase_same_bankPf"
	.size	.L__unnamed_3, 31

	.type	.L__unnamed_4,@object           # @3
.L__unnamed_4:
	.asciz	"_Z29test_many_intra_lane_patternsPf"
	.size	.L__unnamed_4, 36

	.type	.L__unnamed_5,@object           # @4
.L__unnamed_5:
	.asciz	"_Z25test_exact_kernel_patternPf"
	.size	.L__unnamed_5, 32

	.type	.L__unnamed_6,@object           # @5
.L__unnamed_6:
	.asciz	"_Z27test_repeated_exact_patternPfi"
	.size	.L__unnamed_6, 35

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_3b2d6147d262d257
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_3b2d6147d262d257,@object # @__hip_gpubin_handle_3b2d6147d262d257
	.local	__hip_gpubin_handle_3b2d6147d262d257
	.comm	__hip_gpubin_handle_3b2d6147d262d257,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_test_multi_thread_conflict.cpp
	.quad	__hip_module_ctor
	.type	__hip_cuid_3b2d6147d262d257,@object # @__hip_cuid_3b2d6147d262d257
	.bss
	.globl	__hip_cuid_3b2d6147d262d257
__hip_cuid_3b2d6147d262d257:
	.byte	0                               # 0x0
	.size	__hip_cuid_3b2d6147d262d257, 1

	.ident	"AMD clang version 21.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25401 965357120e93d691c2c2f6b221deb863caf44a62)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z41__device_stub__test_same_slot_all_threadsPf
	.addrsig_sym _Z52__device_stub__test_same_bank_diff_slots_all_threadsPf
	.addrsig_sym _Z39__device_stub__test_one_phase_same_bankPf
	.addrsig_sym _Z44__device_stub__test_many_intra_lane_patternsPf
	.addrsig_sym _Z40__device_stub__test_exact_kernel_patternPf
	.addrsig_sym _Z42__device_stub__test_repeated_exact_patternPfi
	.addrsig_sym _GLOBAL__sub_I_test_multi_thread_conflict.cpp
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _Z26test_same_slot_all_threadsPf
	.addrsig_sym _Z37test_same_bank_diff_slots_all_threadsPf
	.addrsig_sym _Z24test_one_phase_same_bankPf
	.addrsig_sym _Z29test_many_intra_lane_patternsPf
	.addrsig_sym _Z25test_exact_kernel_patternPf
	.addrsig_sym _Z27test_repeated_exact_patternPfi
	.addrsig_sym _ZSt4cerr
	.addrsig_sym _ZSt4cout
	.addrsig_sym __hip_fatbin_3b2d6147d262d257
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_3b2d6147d262d257

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
