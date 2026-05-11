	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.section	.text._ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7,"axG",@progbits,_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7,comdat
	.weak	_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7 ; -- Begin function _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7
	.p2align	8
	.type	_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7,@function
_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7: ; @_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7
; %bb.0:                                ; %entry
	;;#ASMSTART
	s_icache_inv 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	s_nop 0 
	
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 0
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
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 0
		.amdhsa_accum_offset 4
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
	.section	.text._ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7,"axG",@progbits,_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7,comdat
.Lfunc_end0:
	.size	_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7, .Lfunc_end0-_ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7
                                        ; -- End function
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.num_vgpr, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.num_agpr, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.numbered_sgpr, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.private_seg_size, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.uses_vcc, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.uses_flat_scratch, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.has_dyn_sized_stack, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.has_recursion, 0
	.set _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 140
; TotalNumSgprs: 6
; NumVgprs: 0
; NumAgprs: 0
; TotalNumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 6
; NumVGPRsForWavesPerEU: 1
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 0
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_53b1c7993ec8f2c7,@object ; @__hip_cuid_53b1c7993ec8f2c7
	.section	.bss.__hip_cuid_53b1c7993ec8f2c7,"aw",@nobits
	.globl	__hip_cuid_53b1c7993ec8f2c7
__hip_cuid_53b1c7993ec8f2c7:
	.byte	0                               ; 0x0
	.size	__hip_cuid_53b1c7993ec8f2c7, 1

	.protected	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.ident	"AMD clang version 21.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25401 965357120e93d691c2c2f6b221deb863caf44a62)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_53b1c7993ec8f2c7
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 0
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .sgpr_spill_count: 0
    .symbol:         _ZN7ck_tileL11flush_cacheEv.intern.53b1c7993ec8f2c7.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
