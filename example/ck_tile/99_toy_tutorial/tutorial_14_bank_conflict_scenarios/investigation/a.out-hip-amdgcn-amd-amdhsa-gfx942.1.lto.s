	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.section	.text._ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,"axG",@progbits,_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,comdat
	.protected	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_ ; -- Begin function _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.weak	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.p2align	8
	.type	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,@function
_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_: ; @_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
; %bb.0:                                ; %entry
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_lshl_b32 s4, s2, 6
	s_waitcnt lgkmcnt(0)
	s_cmp_ge_i32 s4, s8
	s_cbranch_scc1 .LBB0_4
; %bb.1:                                ; %_ZN7ck_tile5arrayIiLi2EEC2ESt16initializer_listIiE.exit.i
	s_cmp_lt_i32 s9, 1
	v_readfirstlane_b32 s2, v0
	s_cbranch_scc1 .LBB0_4
; %bb.2:                                ; %for.body.lr.ph.i
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_mbcnt_hi_u32_b32 v5, -1, v1
	s_load_dwordx4 s[12:15], s[0:1], 0x0
	s_lshr_b32 s0, s2, 2
	v_lshrrev_b32_e32 v1, 2, v5
	s_and_b32 s0, s0, 0x3ffffff0
	v_add_u32_e32 v3, s0, v1
	s_mul_i32 s0, s9, s4
	s_ashr_i32 s1, s0, 31
	v_lshlrev_b32_e32 v4, 3, v5
	v_lshrrev_b32_e32 v3, 1, v3
	s_lshl_b64 s[0:1], s[0:1], 1
	v_and_b32_e32 v2, 24, v4
	v_xor_b32_e32 v6, v5, v3
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s12, s0
	v_and_b32_e32 v4, 56, v4
	v_and_b32_e32 v6, 7, v6
	s_addc_u32 s1, s13, s1
	s_mul_i32 s2, s9, 0x7e
	s_ashr_i32 s5, s4, 31
	v_lshrrev_b32_e32 v7, 1, v4
	s_add_i32 s2, s2, 64
	v_lshlrev_b32_e32 v13, 4, v6
	s_lshl_b64 s[4:5], s[4:5], 1
	v_bfe_u32 v6, v5, 3, 3
	v_or_b32_e32 v8, 1, v7
	v_lshlrev_b32_e32 v12, 7, v3
	v_lshrrev_b32_e32 v3, 3, v5
	s_add_u32 s12, s14, s4
	s_mul_i32 s3, s8, 62
	v_and_b32_e32 v5, 4, v7
	v_lshl_or_b32 v6, v4, 5, v6
	v_sub_u32_e32 v9, v8, v7
	s_mov_b32 s10, 0
	s_addc_u32 s13, s15, s5
	s_movk_i32 s14, 0x80
	s_add_i32 s6, s3, 0x80
	v_and_b32_e32 v7, 5, v8
	v_lshlrev_b32_e32 v8, 6, v9
	v_or_b32_e32 v9, 2, v5
	v_or_b32_e32 v10, 3, v5
	v_lshlrev_b32_e32 v11, 1, v6
	s_lshl_b32 s15, s8, 5
	s_mov_b32 s3, 0x20000
	s_mov_b32 s16, 0x5040100
	v_add_u32_e32 v12, v12, v13
	s_mov_b32 s17, 0
.LBB0_3:                                ; %for.body.i
                                        ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	s_lshr_b32 s5, s4, 2
	s_and_b32 s5, s5, 0x3ffffff0
	v_add_u32_e32 v13, s5, v1
	v_mul_lo_u32 v13, v13, s9
	v_add_lshl_u32 v13, v13, v2, 1
	buffer_load_ushort v14, v13, s[0:3], 0 offen offset:4
	buffer_load_ushort v15, v13, s[0:3], 0 offen offset:6
	buffer_load_ushort v16, v13, s[0:3], 0 offen offset:8
	buffer_load_ushort v17, v13, s[0:3], 0 offen offset:12
	buffer_load_ushort v18, v13, s[0:3], 0 offen offset:14
	buffer_load_ushort v19, v13, s[0:3], 0 offen offset:10
	buffer_load_ushort v20, v13, s[0:3], 0 offen
	buffer_load_ushort v21, v13, s[0:3], 0 offen offset:2
	s_ashr_i32 s11, s10, 31
	s_lshr_b32 s18, s4, 3
	s_lshl_b64 s[4:5], s[10:11], 1
	s_and_b32 s11, s18, 0x1ffffff8
	v_add_u32_e32 v13, s11, v3
	v_lshrrev_b32_e32 v22, 3, v13
	v_add_u32_e32 v24, 4, v22
	v_xor_b32_e32 v23, v22, v5
	v_xor_b32_e32 v29, v24, v5
	v_lshlrev_b32_e32 v27, 3, v23
	v_lshl_add_u32 v28, v23, 4, v11
	v_sub_u32_e32 v23, v29, v23
	v_xor_b32_e32 v25, v22, v7
	v_xor_b32_e32 v30, v24, v7
	v_lshlrev_b32_e32 v23, 3, v23
	v_xor_b32_e32 v26, v22, v9
	v_xor_b32_e32 v22, v22, v10
	v_xor_b32_e32 v31, v24, v9
	v_xor_b32_e32 v24, v24, v10
	v_sub_u32_e32 v29, v25, v29
	v_sub_u32_e32 v25, v30, v25
	v_add3_u32 v23, v27, v6, v23
	v_sub_u32_e32 v30, v26, v30
	v_sub_u32_e32 v26, v31, v26
	v_sub_u32_e32 v31, v22, v31
	v_sub_u32_e32 v22, v24, v22
	v_lshl_add_u32 v24, v29, 3, v8
	v_lshlrev_b32_e32 v29, 3, v25
	v_lshlrev_b32_e32 v27, 1, v23
	v_add3_u32 v23, v24, v23, v29
	v_lshl_add_u32 v24, v24, 1, v27
	v_lshlrev_b32_e32 v32, 3, v26
	v_lshlrev_b32_e32 v33, 3, v31
	v_lshl_add_u32 v23, v30, 3, v23
	v_lshl_add_u32 v25, v25, 4, v24
	v_lshlrev_b32_e32 v26, 4, v26
	v_lshlrev_b32_e32 v22, 4, v22
	v_add3_u32 v23, v23, v32, v33
	v_lshl_add_u32 v29, v30, 4, v25
	v_lshl_add_u32 v22, v23, 1, v22
	v_add3_u32 v23, v29, s14, v26
	s_add_u32 s4, s12, s4
	v_lshl_add_u32 v26, v31, 4, v23
	s_addc_u32 s5, s13, s5
	s_add_i32 s17, s17, 32
	s_add_i32 s10, s10, s15
	s_add_u32 s0, s0, 64
	v_mul_lo_u32 v13, v13, s8
	s_addc_u32 s1, s1, 0
	s_mov_b32 s7, s3
	v_add_lshl_u32 v13, v13, v4, 1
	s_cmp_lt_i32 s17, s9
	s_waitcnt vmcnt(6)
	v_perm_b32 v15, v15, v14, s16
	s_waitcnt vmcnt(3)
	v_perm_b32 v17, v18, v17, s16
	s_waitcnt vmcnt(2)
	v_perm_b32 v16, v19, v16, s16
	s_waitcnt vmcnt(0)
	v_perm_b32 v14, v21, v20, s16
	ds_write_b128 v12, v[14:17]
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v14, v28
	ds_read_u16 v15, v27
	ds_read_u16 v16, v24
	ds_read_u16 v17, v25
	ds_read_u16 v18, v29 offset:128
	ds_read_u16 v19, v23
	ds_read_u16 v20, v26 offset:128
	ds_read_u16 v21, v22 offset:256
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_store_short v14, v13, s[4:7], 0 offen
	buffer_store_short v15, v13, s[4:7], 0 offen offset:2
	buffer_store_short v16, v13, s[4:7], 0 offen offset:4
	buffer_store_short v17, v13, s[4:7], 0 offen offset:6
	buffer_store_short v18, v13, s[4:7], 0 offen offset:8
	buffer_store_short v19, v13, s[4:7], 0 offen offset:10
	buffer_store_short v20, v13, s[4:7], 0 offen offset:12
	buffer_store_short v21, v13, s[4:7], 0 offen offset:14
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_3
.LBB0_4:                                ; %_ZNK25ProductionTransposeKernelIDF16_Lb1EEclEPKDF16_PDF16_ii.exit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
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
		.amdhsa_next_free_vgpr 34
		.amdhsa_next_free_sgpr 19
		.amdhsa_accum_offset 36
		.amdhsa_reserve_vcc 0
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
	.section	.text._ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,"axG",@progbits,_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,comdat
.Lfunc_end0:
	.size	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_, .Lfunc_end0-_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
                                        ; -- End function
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.num_vgpr, 34
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.num_agpr, 0
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.numbered_sgpr, 19
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.private_seg_size, 0
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.uses_vcc, 0
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.uses_flat_scratch, 0
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.has_dyn_sized_stack, 0
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.has_recursion, 0
	.set _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 852
; TotalNumSgprs: 25
; NumVgprs: 34
; NumAgprs: 0
; TotalNumVgprs: 34
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 4096 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 25
; NumVGPRsForWavesPerEU: 34
; AccumOffset: 36
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 8
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.ident	"AMD clang version 21.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25401 965357120e93d691c2c2f6b221deb863caf44a62)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           args.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           args.coerce2
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .name:           args6
        .offset:         16
        .size:           4
        .value_kind:     by_value
      - .name:           args8
        .offset:         20
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
    .private_segment_fixed_size: 0
    .sgpr_count:     25
    .sgpr_spill_count: 0
    .symbol:         _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     34
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
