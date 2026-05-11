	.file	"04_row_major_xor.cpp"
	.text
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
# %bb.0:                                # %entry
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$40, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$161, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.1, %esi
	movl	$57, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.2, %esi
	movl	$57, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.3, %esi
	movl	$160, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	leaq	24(%rsp), %r14
	movq	%r14, 8(%rsp)
	movabsq	$2322223672836973908, %rax      # imm = 0x203A322074736554
	movq	%rax, 24(%rsp)
	movabsq	$6000004305267939360, %rax      # imm = 0x53444C20524F5820
	movq	%rax, 31(%rsp)
	movq	$15, 16(%rsp)
	movb	$0, 39(%rsp)
.Ltmp0:
	leaq	8(%rsp), %rdi
	callq	_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp1:
# %bb.1:                                # %invoke.cont6
	movl	%eax, %ebx
	movq	8(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.LBB0_3
# %bb.2:                                # %if.then.i.i12
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
.LBB0_3:                                # %_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev.exit
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$161, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.5, %esi
	movl	$58, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.6, %esi
	movl	$161, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$11, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$.L.str.8, %eax
	movl	$.L.str.9, %esi
	testb	%bl, %bl
	cmovneq	%rax, %rsi
	movl	$_ZSt4cout, %edi
	movl	$10, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.10, %esi
	movl	$2, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	testb	%bl, %bl
	je	.LBB0_5
# %bb.4:                                # %if.then
	movl	$_ZSt4cout, %edi
	movl	$.L.str.11, %esi
	movl	$53, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.12, %esi
	movl	$43, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.13, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.14, %esi
	movl	$45, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.LBB0_5:                                # %if.end
	xorb	$1, %bl
	movzbl	%bl, %eax
	addq	$40, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.LBB0_6:                                # %lpad5
	.cfi_def_cfa_offset 64
.Ltmp2:
	movq	%rax, %rbx
	movq	8(%rsp), %rdi
	cmpq	%r14, %rdi
	je	.LBB0_8
# %bb.7:                                # %if.then.i.i34
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
.LBB0_8:                                # %ehcleanup
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Lfunc_end0-.Ltmp1             #   Call between .Ltmp1 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function _Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.LCPI1_0:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
.LCPI1_1:
	.long	4                               # 0x4
	.long	5                               # 0x5
	.long	6                               # 0x6
	.long	7                               # 0x7
.LCPI1_2:
	.long	8                               # 0x8
	.long	9                               # 0x9
	.long	10                              # 0xa
	.long	11                              # 0xb
.LCPI1_3:
	.long	12                              # 0xc
	.long	13                              # 0xd
	.long	14                              # 0xe
	.long	15                              # 0xf
.LCPI1_4:
	.long	16                              # 0x10
	.long	17                              # 0x11
	.long	18                              # 0x12
	.long	19                              # 0x13
.LCPI1_5:
	.long	20                              # 0x14
	.long	21                              # 0x15
	.long	22                              # 0x16
	.long	23                              # 0x17
.LCPI1_6:
	.long	24                              # 0x18
	.long	25                              # 0x19
	.long	26                              # 0x1a
	.long	27                              # 0x1b
.LCPI1_7:
	.long	28                              # 0x1c
	.long	29                              # 0x1d
	.long	30                              # 0x1e
	.long	31                              # 0x1f
.LCPI1_8:
	.long	32                              # 0x20
	.long	33                              # 0x21
	.long	34                              # 0x22
	.long	35                              # 0x23
.LCPI1_9:
	.long	36                              # 0x24
	.long	37                              # 0x25
	.long	38                              # 0x26
	.long	39                              # 0x27
.LCPI1_10:
	.long	40                              # 0x28
	.long	41                              # 0x29
	.long	42                              # 0x2a
	.long	43                              # 0x2b
.LCPI1_11:
	.long	44                              # 0x2c
	.long	45                              # 0x2d
	.long	46                              # 0x2e
	.long	47                              # 0x2f
.LCPI1_12:
	.long	48                              # 0x30
	.long	49                              # 0x31
	.long	50                              # 0x32
	.long	51                              # 0x33
.LCPI1_13:
	.long	52                              # 0x34
	.long	53                              # 0x35
	.long	54                              # 0x36
	.long	55                              # 0x37
.LCPI1_14:
	.long	56                              # 0x38
	.long	57                              # 0x39
	.long	58                              # 0x3a
	.long	59                              # 0x3b
.LCPI1_15:
	.long	60                              # 0x3c
	.long	61                              # 0x3d
	.long	62                              # 0x3e
	.long	63                              # 0x3f
.LCPI1_16:
	.long	64                              # 0x40
	.long	65                              # 0x41
	.long	66                              # 0x42
	.long	67                              # 0x43
.LCPI1_17:
	.long	68                              # 0x44
	.long	69                              # 0x45
	.long	70                              # 0x46
	.long	71                              # 0x47
.LCPI1_18:
	.long	72                              # 0x48
	.long	73                              # 0x49
	.long	74                              # 0x4a
	.long	75                              # 0x4b
.LCPI1_19:
	.long	76                              # 0x4c
	.long	77                              # 0x4d
	.long	78                              # 0x4e
	.long	79                              # 0x4f
.LCPI1_20:
	.long	80                              # 0x50
	.long	81                              # 0x51
	.long	82                              # 0x52
	.long	83                              # 0x53
.LCPI1_21:
	.long	84                              # 0x54
	.long	85                              # 0x55
	.long	86                              # 0x56
	.long	87                              # 0x57
.LCPI1_22:
	.long	88                              # 0x58
	.long	89                              # 0x59
	.long	90                              # 0x5a
	.long	91                              # 0x5b
.LCPI1_23:
	.long	92                              # 0x5c
	.long	93                              # 0x5d
	.long	94                              # 0x5e
	.long	95                              # 0x5f
.LCPI1_24:
	.long	96                              # 0x60
	.long	97                              # 0x61
	.long	98                              # 0x62
	.long	99                              # 0x63
.LCPI1_25:
	.long	100                             # 0x64
	.long	101                             # 0x65
	.long	102                             # 0x66
	.long	103                             # 0x67
.LCPI1_26:
	.long	104                             # 0x68
	.long	105                             # 0x69
	.long	106                             # 0x6a
	.long	107                             # 0x6b
.LCPI1_27:
	.long	108                             # 0x6c
	.long	109                             # 0x6d
	.long	110                             # 0x6e
	.long	111                             # 0x6f
.LCPI1_28:
	.long	112                             # 0x70
	.long	113                             # 0x71
	.long	114                             # 0x72
	.long	115                             # 0x73
.LCPI1_29:
	.long	116                             # 0x74
	.long	117                             # 0x75
	.long	118                             # 0x76
	.long	119                             # 0x77
.LCPI1_30:
	.long	120                             # 0x78
	.long	121                             # 0x79
	.long	122                             # 0x7a
	.long	123                             # 0x7b
.LCPI1_31:
	.long	124                             # 0x7c
	.long	125                             # 0x7d
	.long	126                             # 0x7e
	.long	127                             # 0x7f
	.section	.text._Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,"axG",@progbits,_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,comdat
	.weak	_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.p2align	4
	.type	_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,@function
_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE: # @_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception1
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$232, %rsp
	.cfi_def_cfa_offset 288
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	movl	$_ZSt4cout, %edi
	movl	$.L.str.16, %esi
	movl	$42, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	(%rbx), %rsi
	movq	8(%rbx), %rdx
	movl	$_ZSt4cout, %edi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$.L.str.17, %esi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	movl	$.L.str.18, %esi
	movl	$42, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$65536, %edi                    # imm = 0x10000
	callq	_Znwm
	movq	%rax, %r12
	movl	$65536, %edx                    # imm = 0x10000
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
.Ltmp3:
	movl	$65536, %edi                    # imm = 0x10000
	movq	%r12, 96(%rsp)                  # 8-byte Spill
	callq	_Znwm
.Ltmp4:
# %bb.1:                                # %invoke.cont6
	xorl	%r15d, %r15d
	movl	$65536, %edx                    # imm = 0x10000
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	addq	$240, %r12
	.p2align	4
.LBB1_2:                                # %for.cond7.preheader
                                        # =>This Inner Loop Header: Depth=1
	movd	%r15d, %xmm0
	pshufd	$0, %xmm0, %xmm0                # xmm0 = xmm0[0,0,0,0]
	movdqa	%xmm0, 64(%rsp)                 # 16-byte Spill
	movdqa	%xmm0, %xmm1
	por	.LCPI1_0(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	por	.LCPI1_1(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -240(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_2(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_3(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -224(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_4(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_5(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -208(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_6(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_7(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -192(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_8(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_9(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -176(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_10(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_11(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -160(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_12(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_13(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -144(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_14(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_15(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -128(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_16(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_17(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -112(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_18(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_19(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -96(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_20(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_21(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -80(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_22(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_23(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -64(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_24(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_25(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -48(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_26(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_27(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -32(%r12)
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movdqa	%xmm0, %xmm1
	paddd	.LCPI1_28(%rip), %xmm1
	movdqa	%xmm1, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_29(%rip), %xmm0
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, (%rsp), %xmm0             # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 48(%rsp)                 # 16-byte Spill
	movdqa	(%rsp), %xmm0                   # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, (%rsp), %xmm0              # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	48(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 48(%rsp)                 # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	48(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, -16(%r12)
	movdqa	64(%rsp), %xmm1                 # 16-byte Reload
	movdqa	%xmm1, %xmm0
	paddd	.LCPI1_30(%rip), %xmm0
	movdqa	%xmm0, 32(%rsp)                 # 16-byte Spill
	paddd	.LCPI1_31(%rip), %xmm1
	movdqa	%xmm1, 64(%rsp)                 # 16-byte Spill
	pshufd	$255, %xmm1, %xmm0              # xmm0 = xmm1[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$238, 64(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	16(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, (%rsp)                   # 16-byte Spill
	movdqa	64(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 16(%rsp)                 # 16-byte Spill
	pshufd	$85, 64(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	16(%rsp), %xmm1                 # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	(%rsp), %xmm1           # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	movdqa	%xmm1, 16(%rsp)                 # 16-byte Spill
	pshufd	$255, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[3,3,3,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, 64(%rsp)                 # 16-byte Spill
	pshufd	$238, 32(%rsp), %xmm0           # 16-byte Folded Reload
                                        # xmm0 = mem[2,3,2,3]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	punpcklwd	64(%rsp), %xmm0         # 16-byte Folded Reload
                                        # xmm0 = xmm0[0],mem[0],xmm0[1],mem[1],xmm0[2],mem[2],xmm0[3],mem[3]
	movdqa	%xmm0, 64(%rsp)                 # 16-byte Spill
	movdqa	32(%rsp), %xmm0                 # 16-byte Reload
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movaps	%xmm0, (%rsp)                   # 16-byte Spill
	pshufd	$85, 32(%rsp), %xmm0            # 16-byte Folded Reload
                                        # xmm0 = mem[1,1,1,1]
	movd	%xmm0, %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%rax, %xmm0
	callq	__truncsfhf2@PLT
	movdqa	(%rsp), %xmm1                   # 16-byte Reload
	punpcklwd	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1],xmm1[2],xmm0[2],xmm1[3],xmm0[3]
	punpckldq	64(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0],xmm1[1],mem[1]
	punpcklqdq	16(%rsp), %xmm1         # 16-byte Folded Reload
                                        # xmm1 = xmm1[0],mem[0]
	movdqu	%xmm1, (%r12)
	addq	$256, %r12                      # imm = 0x100
	addq	$1000, %r15                     # imm = 0x3E8
	cmpq	$256000, %r15                   # imm = 0x3E800
	jne	.LBB1_2
# %bb.3:                                # %for.cond.cleanup
.Ltmp6:
	leaq	120(%rsp), %rdi
	movl	$65536, %esi                    # imm = 0x10000
	callq	_ZN7ck_tile9DeviceMemC2Em
.Ltmp7:
# %bb.4:                                # %invoke.cont19
.Ltmp9:
	leaq	104(%rsp), %rdi
	movl	$65536, %esi                    # imm = 0x10000
	callq	_ZN7ck_tile9DeviceMemC2Em
.Ltmp10:
# %bb.5:                                # %invoke.cont21
.Ltmp12:
	leaq	120(%rsp), %rdi
	movl	$65536, %edx                    # imm = 0x10000
	movq	96(%rsp), %rsi                  # 8-byte Reload
	callq	_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm
.Ltmp13:
# %bb.6:                                # %invoke.cont24
.Ltmp15:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.19, %esi
	movl	$15, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp16:
# %bb.7:                                # %invoke.cont26
.Ltmp17:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.20, %esi
	movl	$11, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp18:
# %bb.8:                                # %invoke.cont28
.Ltmp19:
	movl	$_ZSt4cout, %edi
	movl	$256, %esi                      # imm = 0x100
	callq	_ZNSolsEi
.Ltmp20:
# %bb.9:                                # %invoke.cont30
.Ltmp21:
	movq	%rax, %r15
	movl	$.L.str.21, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp22:
# %bb.10:                               # %invoke.cont32
.Ltmp23:
	movq	%r15, %rdi
	movl	$128, %esi
	callq	_ZNSolsEi
.Ltmp24:
# %bb.11:                               # %invoke.cont34
.Ltmp25:
	movl	$.L.str.22, %esi
	movl	$14, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp26:
# %bb.12:                               # %invoke.cont36
.Ltmp27:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.23, %esi
	movl	$11, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp28:
# %bb.13:                               # %invoke.cont38
.Ltmp29:
	movl	$_ZSt4cout, %edi
	movl	$128, %esi
	callq	_ZNSolsEi
.Ltmp30:
# %bb.14:                               # %invoke.cont40
.Ltmp31:
	movq	%rax, %r15
	movl	$.L.str.21, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp32:
# %bb.15:                               # %invoke.cont42
.Ltmp33:
	movq	%r15, %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	_ZNSolsEi
.Ltmp34:
# %bb.16:                               # %invoke.cont44
.Ltmp35:
	movl	$.L.str.24, %esi
	movl	$15, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp36:
# %bb.17:                               # %invoke.cont46
.Ltmp37:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.25, %esi
	movl	$7, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp38:
# %bb.18:                               # %invoke.cont48
.Ltmp39:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.26, %esi
	movl	$7, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp40:
# %bb.19:                               # %invoke.cont50
.Ltmp41:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.17, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp42:
# %bb.20:                               # %invoke.cont52
.Ltmp43:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.27, %esi
	movl	$42, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp44:
# %bb.21:                               # %invoke.cont66
	movq	$0, 136(%rsp)
	movb	$0, 144(%rsp)
	movabsq	$12884901888, %rax              # imm = 0x300000000
	movq	%rax, 148(%rsp)
	movl	$10, 156(%rsp)
	movw	$1, 160(%rsp)
	movl	$1, 164(%rsp)
	movq	120(%rsp), %rax
	movq	104(%rsp), %rcx
	movq	$_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_, 168(%rsp)
	movabsq	$4294967300, %rdx               # imm = 0x100000004
	movq	%rdx, 176(%rsp)
	movl	$1, 184(%rsp)
	addq	$252, %rdx
	movq	%rdx, 188(%rsp)
	movl	$1, 196(%rsp)
	movq	$4096, 200(%rsp)                # imm = 0x1000
	movq	%rax, 208(%rsp)
	movq	%rcx, 216(%rsp)
	movabsq	$549755814144, %rax             # imm = 0x8000000100
	movq	%rax, 224(%rsp)
.Ltmp46:
	leaq	136(%rsp), %rdi
	leaq	168(%rsp), %rsi
	callq	_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_
.Ltmp47:
# %bb.22:                               # %invoke.cont67
.Ltmp49:
	callq	hipDeviceSynchronize
.Ltmp50:
# %bb.23:                               # %invoke.cont70
.Ltmp51:
	movl	%eax, %edi
	callq	_ZN7ck_tile15hip_check_errorE10hipError_t
.Ltmp52:
# %bb.24:                               # %invoke.cont72
.Ltmp53:
	leaq	104(%rsp), %rdi
	movl	$65536, %edx                    # imm = 0x10000
	movq	88(%rsp), %rsi                  # 8-byte Reload
	callq	_ZNK7ck_tile9DeviceMem10FromDeviceEPvm
.Ltmp54:
# %bb.25:                               # %for.cond82.preheader.preheader
	movb	$1, %r13b
	xorl	%r15d, %r15d
	movq	96(%rsp), %rbp                  # 8-byte Reload
	movq	88(%rsp), %r14                  # 8-byte Reload
	xorl	%ebx, %ebx
	.p2align	4
.LBB1_26:                               # %for.cond82.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_27 Depth 2
	movq	%rbp, 16(%rsp)                  # 8-byte Spill
	xorl	%r12d, %r12d
	.p2align	4
.LBB1_27:                               # %for.body88
                                        #   Parent Loop BB1_26 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	pinsrw	$0, (%rbp), %xmm1
	pinsrw	$0, (%r14,%r12,2), %xmm0
	movdqa	%xmm1, 64(%rsp)                 # 16-byte Spill
	pextrw	$0, %xmm1, %eax
	movdqa	%xmm0, 32(%rsp)                 # 16-byte Spill
	pextrw	$0, %xmm0, %ecx
	cmpw	%cx, %ax
	je	.LBB1_39
# %bb.28:                               # %if.then
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp56:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.28, %esi
	movl	$10, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp57:
# %bb.29:                               # %invoke.cont105
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp58:
	movl	$_ZSt4cout, %edi
	movl	%r15d, %esi
	callq	_ZNSolsEi
.Ltmp59:
# %bb.30:                               # %invoke.cont107
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp60:
	movq	%rax, %r13
	movl	$.L.str.29, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp61:
# %bb.31:                               # %invoke.cont109
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp62:
	movq	%r13, %rdi
	movl	%r12d, %esi
	callq	_ZNSolsEi
.Ltmp63:
# %bb.32:                               # %invoke.cont111
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp64:
	movq	%rax, %r13
	movl	$.L.str.30, %esi
	movl	$3, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp65:
# %bb.33:                               # %invoke.cont113
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp66:
	movl	$.L.str.31, %esi
	movl	$9, %edx
	movq	%r13, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp67:
# %bb.34:                               # %invoke.cont115
                                        #   in Loop: Header=BB1_27 Depth=2
	movaps	64(%rsp), %xmm0                 # 16-byte Reload
	callq	__extendhfsf2@PLT
	cvtss2sd	%xmm0, %xmm0
.Ltmp68:
	movq	%r13, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_
.Ltmp69:
# %bb.35:                               # %invoke.cont118
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp70:
	movq	%rax, %r13
	movl	$.L.str.32, %esi
	movl	$6, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp71:
# %bb.36:                               # %invoke.cont120
                                        #   in Loop: Header=BB1_27 Depth=2
	movaps	32(%rsp), %xmm0                 # 16-byte Reload
	callq	__extendhfsf2@PLT
	cvtss2sd	%xmm0, %xmm0
.Ltmp72:
	movq	%r13, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_
.Ltmp73:
# %bb.37:                               # %invoke.cont123
                                        #   in Loop: Header=BB1_27 Depth=2
.Ltmp74:
	movl	$.L.str.17, %esi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp75:
# %bb.38:                               # %invoke.cont125
                                        #   in Loop: Header=BB1_27 Depth=2
	incl	%ebx
	xorl	%r13d, %r13d
.LBB1_39:                               # %if.end
                                        #   in Loop: Header=BB1_27 Depth=2
	cmpq	$254, %r12
	ja	.LBB1_41
# %bb.40:                               # %if.end
                                        #   in Loop: Header=BB1_27 Depth=2
	incq	%r12
	addq	$256, %rbp                      # imm = 0x100
	cmpl	$10, %ebx
	jl	.LBB1_27
.LBB1_41:                               # %for.cond.cleanup87
                                        #   in Loop: Header=BB1_26 Depth=1
	cmpq	$126, %r15
	movq	16(%rsp), %rbp                  # 8-byte Reload
	ja	.LBB1_43
# %bb.42:                               # %for.cond.cleanup87
                                        #   in Loop: Header=BB1_26 Depth=1
	incq	%r15
	addq	$512, %r14                      # imm = 0x200
	addq	$2, %rbp
	cmpl	$10, %ebx
	jl	.LBB1_26
.LBB1_43:                               # %for.cond.cleanup79
.Ltmp77:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.33, %esi
	movl	$8, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp78:
# %bb.44:                               # %invoke.cont135
	movl	$.L.str.8, %eax
	movl	$.L.str.9, %esi
	testb	$1, %r13b
	cmovneq	%rax, %rsi
.Ltmp79:
	movl	$_ZSt4cout, %edi
	movl	$10, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp80:
# %bb.45:                               # %invoke.cont137
.Ltmp81:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.34, %esi
	movl	$17, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp82:
# %bb.46:                               # %invoke.cont139
.Ltmp83:
	movl	$_ZSt4cout, %edi
	movl	$128, %esi
	callq	_ZNSolsEi
.Ltmp84:
# %bb.47:                               # %invoke.cont141
.Ltmp85:
	movq	%rax, %r15
	movl	$.L.str.21, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp86:
# %bb.48:                               # %invoke.cont143
.Ltmp87:
	movq	%r15, %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	_ZNSolsEi
.Ltmp88:
# %bb.49:                               # %invoke.cont145
.Ltmp89:
	movl	$.L.str.35, %esi
	movl	$3, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp90:
# %bb.50:                               # %_ZNSt6vectorIDF16_SaIDF16_EED2Ev.exit140
	leaq	104(%rsp), %rdi
	callq	_ZN7ck_tile9DeviceMemD2Ev
	leaq	120(%rsp), %rdi
	callq	_ZN7ck_tile9DeviceMemD2Ev
	movl	$65536, %esi                    # imm = 0x10000
	movq	88(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm
	movl	$65536, %esi                    # imm = 0x10000
	movq	96(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm
	andb	$1, %r13b
	movl	%r13d, %eax
	addq	$232, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB1_55:                               # %lpad58
	.cfi_def_cfa_offset 288
.Ltmp48:
	jmp	.LBB1_59
.LBB1_53:                               # %lpad23
.Ltmp14:
	jmp	.LBB1_59
.LBB1_52:                               # %lpad20
.Ltmp11:
	movq	%rax, %r15
	jmp	.LBB1_60
.LBB1_51:                               # %lpad18
.Ltmp8:
	movq	%rax, %r15
	jmp	.LBB1_61
.LBB1_63:                               # %lpad5
.Ltmp5:
	movq	%rax, %r15
	jmp	.LBB1_62
.LBB1_56:                               # %lpad69
.Ltmp55:
	jmp	.LBB1_59
.LBB1_58:                               # %lpad134
.Ltmp91:
	jmp	.LBB1_59
.LBB1_54:                               # %lpad25
.Ltmp45:
	jmp	.LBB1_59
.LBB1_57:                               # %lpad97
.Ltmp76:
.LBB1_59:                               # %ehcleanup151
	movq	%rax, %r15
	leaq	104(%rsp), %rdi
	callq	_ZN7ck_tile9DeviceMemD2Ev
.LBB1_60:                               # %ehcleanup157
	leaq	120(%rsp), %rdi
	callq	_ZN7ck_tile9DeviceMemD2Ev
.LBB1_61:                               # %_ZNSt6vectorIDF16_SaIDF16_EED2Ev.exit147
	movl	$65536, %esi                    # imm = 0x10000
	movq	88(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm
.LBB1_62:                               # %_ZNSt6vectorIDF16_SaIDF16_EED2Ev.exit154
	movl	$65536, %esi                    # imm = 0x10000
	movq	96(%rsp), %rdi                  # 8-byte Reload
	callq	_ZdlPvm
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end1:
	.size	_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE, .Lfunc_end1-_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
	.cfi_endproc
	.section	.gcc_except_table._Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,"aG",@progbits,_Z8run_testILb1EEbRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE,comdat
	.p2align	2, 0x0
GCC_except_table1:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp3-.Lfunc_begin1           #   Call between .Lfunc_begin1 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin1           # >> Call Site 2 <<
	.uleb128 .Ltmp4-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp4
	.uleb128 .Ltmp5-.Lfunc_begin1           #     jumps to .Ltmp5
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp4-.Lfunc_begin1           # >> Call Site 3 <<
	.uleb128 .Ltmp6-.Ltmp4                  #   Call between .Ltmp4 and .Ltmp6
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp6-.Lfunc_begin1           # >> Call Site 4 <<
	.uleb128 .Ltmp7-.Ltmp6                  #   Call between .Ltmp6 and .Ltmp7
	.uleb128 .Ltmp8-.Lfunc_begin1           #     jumps to .Ltmp8
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp9-.Lfunc_begin1           # >> Call Site 5 <<
	.uleb128 .Ltmp10-.Ltmp9                 #   Call between .Ltmp9 and .Ltmp10
	.uleb128 .Ltmp11-.Lfunc_begin1          #     jumps to .Ltmp11
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin1          # >> Call Site 6 <<
	.uleb128 .Ltmp13-.Ltmp12                #   Call between .Ltmp12 and .Ltmp13
	.uleb128 .Ltmp14-.Lfunc_begin1          #     jumps to .Ltmp14
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp15-.Lfunc_begin1          # >> Call Site 7 <<
	.uleb128 .Ltmp44-.Ltmp15                #   Call between .Ltmp15 and .Ltmp44
	.uleb128 .Ltmp45-.Lfunc_begin1          #     jumps to .Ltmp45
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp46-.Lfunc_begin1          # >> Call Site 8 <<
	.uleb128 .Ltmp47-.Ltmp46                #   Call between .Ltmp46 and .Ltmp47
	.uleb128 .Ltmp48-.Lfunc_begin1          #     jumps to .Ltmp48
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp49-.Lfunc_begin1          # >> Call Site 9 <<
	.uleb128 .Ltmp54-.Ltmp49                #   Call between .Ltmp49 and .Ltmp54
	.uleb128 .Ltmp55-.Lfunc_begin1          #     jumps to .Ltmp55
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp56-.Lfunc_begin1          # >> Call Site 10 <<
	.uleb128 .Ltmp67-.Ltmp56                #   Call between .Ltmp56 and .Ltmp67
	.uleb128 .Ltmp76-.Lfunc_begin1          #     jumps to .Ltmp76
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp67-.Lfunc_begin1          # >> Call Site 11 <<
	.uleb128 .Ltmp68-.Ltmp67                #   Call between .Ltmp67 and .Ltmp68
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp68-.Lfunc_begin1          # >> Call Site 12 <<
	.uleb128 .Ltmp71-.Ltmp68                #   Call between .Ltmp68 and .Ltmp71
	.uleb128 .Ltmp76-.Lfunc_begin1          #     jumps to .Ltmp76
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp71-.Lfunc_begin1          # >> Call Site 13 <<
	.uleb128 .Ltmp72-.Ltmp71                #   Call between .Ltmp71 and .Ltmp72
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp72-.Lfunc_begin1          # >> Call Site 14 <<
	.uleb128 .Ltmp75-.Ltmp72                #   Call between .Ltmp72 and .Ltmp75
	.uleb128 .Ltmp76-.Lfunc_begin1          #     jumps to .Ltmp76
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp77-.Lfunc_begin1          # >> Call Site 15 <<
	.uleb128 .Ltmp90-.Ltmp77                #   Call between .Ltmp77 and .Ltmp90
	.uleb128 .Ltmp91-.Lfunc_begin1          #     jumps to .Ltmp91
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp90-.Lfunc_begin1          # >> Call Site 16 <<
	.uleb128 .Lfunc_end1-.Ltmp90            #   Call between .Ltmp90 and .Lfunc_end1
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text.__clang_call_terminate,"axG",@progbits,__clang_call_terminate,comdat
	.hidden	__clang_call_terminate          # -- Begin function __clang_call_terminate
	.weak	__clang_call_terminate
	.p2align	4
	.type	__clang_call_terminate,@function
__clang_call_terminate:                 # @__clang_call_terminate
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__cxa_begin_catch
	callq	_ZSt9terminatev
.Lfunc_end2:
	.size	__clang_call_terminate, .Lfunc_end2-__clang_call_terminate
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN7ck_tile9DeviceMemC2Em,"axG",@progbits,_ZN7ck_tile9DeviceMemC2Em,comdat
	.weak	_ZN7ck_tile9DeviceMemC2Em       # -- Begin function _ZN7ck_tile9DeviceMemC2Em
	.p2align	4
	.type	_ZN7ck_tile9DeviceMemC2Em,@function
_ZN7ck_tile9DeviceMemC2Em:              # @_ZN7ck_tile9DeviceMemC2Em
.Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception2
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rsi, 8(%rdi)
	testq	%rsi, %rsi
	je	.LBB3_19
# %bb.1:                                # %do.body
	callq	hipMalloc
	testl	%eax, %eax
	je	.LBB3_20
# %bb.2:                                # %if.then5
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp92:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp93:
# %bb.3:                                # %invoke.cont
.Ltmp94:
	leaq	40(%rsp), %rdi
	movl	$.L.str.38, %esi
	movl	$72, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp95:
# %bb.4:                                # %invoke.cont7
.Ltmp96:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp97:
# %bb.5:                                # %invoke.cont9
.Ltmp98:
	leaq	40(%rsp), %rdi
	movl	$58, %esi
	callq	_ZNSolsEi
.Ltmp99:
# %bb.6:                                # %invoke.cont11
.Ltmp100:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp101:
# %bb.7:                                # %invoke.cont13
.Ltmp102:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp103:
# %bb.8:                                # %invoke.cont15
.Ltmp104:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp105:
# %bb.9:                                # %invoke.cont17
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp107:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp108:
# %bb.10:                               # %invoke.cont20
	movb	$1, %bpl
.Ltmp110:
	leaq	8(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp111:
# %bb.11:                               # %invoke.cont22
	xorl	%ebp, %ebp
.Ltmp112:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp113:
# %bb.21:                               # %unreachable
.LBB3_19:                               # %if.else
	movq	$0, (%rdi)
.LBB3_20:                               # %if.end28
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB3_14:                               # %lpad21
	.cfi_def_cfa_offset 448
.Ltmp114:
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB3_15
# %bb.16:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB3_17
.LBB3_18:                               # %ehcleanup24
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB3_15:                               # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB3_18
	jmp	.LBB3_17
.LBB3_13:                               # %ehcleanup.thread
.Ltmp109:
	movq	%rax, %r14
.LBB3_17:                               # %cleanup.action
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB3_12:                               # %lpad
.Ltmp106:
	movq	%rax, %r14
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end3:
	.size	_ZN7ck_tile9DeviceMemC2Em, .Lfunc_end3-_ZN7ck_tile9DeviceMemC2Em
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9DeviceMemC2Em,"aG",@progbits,_ZN7ck_tile9DeviceMemC2Em,comdat
	.p2align	2, 0x0
GCC_except_table3:
.Lexception2:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end2-.Lcst_begin2
.Lcst_begin2:
	.uleb128 .Lfunc_begin2-.Lfunc_begin2    # >> Call Site 1 <<
	.uleb128 .Ltmp92-.Lfunc_begin2          #   Call between .Lfunc_begin2 and .Ltmp92
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp92-.Lfunc_begin2          # >> Call Site 2 <<
	.uleb128 .Ltmp105-.Ltmp92               #   Call between .Ltmp92 and .Ltmp105
	.uleb128 .Ltmp106-.Lfunc_begin2         #     jumps to .Ltmp106
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp105-.Lfunc_begin2         # >> Call Site 3 <<
	.uleb128 .Ltmp107-.Ltmp105              #   Call between .Ltmp105 and .Ltmp107
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp107-.Lfunc_begin2         # >> Call Site 4 <<
	.uleb128 .Ltmp108-.Ltmp107              #   Call between .Ltmp107 and .Ltmp108
	.uleb128 .Ltmp109-.Lfunc_begin2         #     jumps to .Ltmp109
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp110-.Lfunc_begin2         # >> Call Site 5 <<
	.uleb128 .Ltmp113-.Ltmp110              #   Call between .Ltmp110 and .Ltmp113
	.uleb128 .Ltmp114-.Lfunc_begin2         #     jumps to .Ltmp114
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp113-.Lfunc_begin2         # >> Call Site 6 <<
	.uleb128 .Lfunc_end3-.Ltmp113           #   Call between .Ltmp113 and .Lfunc_end3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end2:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNK7ck_tile9DeviceMem8ToDeviceEPKvm,"axG",@progbits,_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm,comdat
	.weak	_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm # -- Begin function _ZNK7ck_tile9DeviceMem8ToDeviceEPKvm
	.p2align	4
	.type	_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm,@function
_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm:   # @_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm
.Lfunc_begin3:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception3
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB4_19
# %bb.1:                                # %do.body
	movl	$1, %ecx
	callq	hipMemcpy
	testl	%eax, %eax
	jne	.LBB4_2
.LBB4_19:                               # %if.end25
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB4_2:                                # %if.then3
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp115:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp116:
# %bb.3:                                # %invoke.cont
.Ltmp117:
	leaq	40(%rsp), %rdi
	movl	$.L.str.38, %esi
	movl	$72, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp118:
# %bb.4:                                # %invoke.cont5
.Ltmp119:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp120:
# %bb.5:                                # %invoke.cont7
.Ltmp121:
	leaq	40(%rsp), %rdi
	movl	$113, %esi
	callq	_ZNSolsEi
.Ltmp122:
# %bb.6:                                # %invoke.cont9
.Ltmp123:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp124:
# %bb.7:                                # %invoke.cont11
.Ltmp125:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp126:
# %bb.8:                                # %invoke.cont13
.Ltmp127:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp128:
# %bb.9:                                # %invoke.cont15
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp130:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp131:
# %bb.10:                               # %invoke.cont18
	movb	$1, %bpl
.Ltmp133:
	leaq	8(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp134:
# %bb.11:                               # %invoke.cont20
	xorl	%ebp, %ebp
.Ltmp135:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp136:
# %bb.20:                               # %unreachable
.LBB4_14:                               # %lpad19
.Ltmp137:
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB4_15
# %bb.16:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB4_17
.LBB4_18:                               # %ehcleanup22
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB4_15:                               # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB4_18
	jmp	.LBB4_17
.LBB4_13:                               # %ehcleanup.thread
.Ltmp132:
	movq	%rax, %r14
.LBB4_17:                               # %cleanup.action
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB4_12:                               # %lpad
.Ltmp129:
	movq	%rax, %r14
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end4:
	.size	_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm, .Lfunc_end4-_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm
	.cfi_endproc
	.section	.gcc_except_table._ZNK7ck_tile9DeviceMem8ToDeviceEPKvm,"aG",@progbits,_ZNK7ck_tile9DeviceMem8ToDeviceEPKvm,comdat
	.p2align	2, 0x0
GCC_except_table4:
.Lexception3:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end3-.Lcst_begin3
.Lcst_begin3:
	.uleb128 .Lfunc_begin3-.Lfunc_begin3    # >> Call Site 1 <<
	.uleb128 .Ltmp115-.Lfunc_begin3         #   Call between .Lfunc_begin3 and .Ltmp115
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp115-.Lfunc_begin3         # >> Call Site 2 <<
	.uleb128 .Ltmp128-.Ltmp115              #   Call between .Ltmp115 and .Ltmp128
	.uleb128 .Ltmp129-.Lfunc_begin3         #     jumps to .Ltmp129
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp128-.Lfunc_begin3         # >> Call Site 3 <<
	.uleb128 .Ltmp130-.Ltmp128              #   Call between .Ltmp128 and .Ltmp130
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp130-.Lfunc_begin3         # >> Call Site 4 <<
	.uleb128 .Ltmp131-.Ltmp130              #   Call between .Ltmp130 and .Ltmp131
	.uleb128 .Ltmp132-.Lfunc_begin3         #     jumps to .Ltmp132
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp133-.Lfunc_begin3         # >> Call Site 5 <<
	.uleb128 .Ltmp136-.Ltmp133              #   Call between .Ltmp133 and .Ltmp136
	.uleb128 .Ltmp137-.Lfunc_begin3         #     jumps to .Ltmp137
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp136-.Lfunc_begin3         # >> Call Site 6 <<
	.uleb128 .Lfunc_end4-.Ltmp136           #   Call between .Ltmp136 and .Lfunc_end4
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end3:
	.p2align	2, 0x0
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function _ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_
.LCPI5_0:
	.quad	0x41cdcd6500000000              # double 1.0E+9
.LCPI5_1:
	.quad	0x408f400000000000              # double 1000
	.section	.text._ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_,"axG",@progbits,_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_,comdat
	.weak	_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_
	.p2align	4
	.type	_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_,@function
_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_: # @_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_
.Lfunc_begin4:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception4
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %r14
	movq	%rdi, %rbx
	cmpb	$0, 8(%rdi)
	je	.LBB5_1
# %bb.2:                                # %if.end
	cmpb	$1, 24(%rbx)
	jne	.LBB5_22
# %bb.3:                                # %if.then2
	movq	%rsp, %rdi
	callq	_ZN7ck_tile9gpu_timerC2Ev
	cmpl	$0, 16(%rbx)
	jle	.LBB5_7
# %bb.4:                                # %for.body.i.preheader
	xorl	%ebp, %ebp
	.p2align	4
.LBB5_5:                                # %for.body.i
                                        # =>This Inner Loop Header: Depth=1
.Ltmp138:
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
.Ltmp139:
# %bb.6:                                # %.noexc8
                                        #   in Loop: Header=BB5_5 Depth=1
	incl	%ebp
	cmpl	16(%rbx), %ebp
	jl	.LBB5_5
.LBB5_7:                                # %for.cond.cleanup.i
.Ltmp141:
	movq	%rsp, %rdi
	movq	%rbx, %rsi
	callq	_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t
.Ltmp142:
# %bb.8:                                # %.noexc
	cmpl	$0, 20(%rbx)
	jle	.LBB5_15
# %bb.9:                                # %while.body.i.preheader
	xorl	%ebp, %ebp
	.p2align	4
.LBB5_10:                               # %while.body.i
                                        # =>This Inner Loop Header: Depth=1
.Ltmp143:
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
.Ltmp144:
# %bb.11:                               # %.noexc9
                                        #   in Loop: Header=BB5_10 Depth=1
	incl	%ebp
	cmpl	20(%rbx), %ebp
	jl	.LBB5_10
# %bb.12:                               # %if.end.i
.Ltmp146:
	movq	%rsp, %rdi
	movq	%rbx, %rsi
	callq	_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t
.Ltmp147:
# %bb.13:                               # %.noexc10
.Ltmp148:
	movq	%rsp, %rdi
	callq	_ZNK7ck_tile9gpu_timer8durationEv
.Ltmp149:
# %bb.14:                               # %call.i.noexc
	cvtsi2ssl	20(%rbx), %xmm1
	divss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movsd	%xmm0, 40(%rsp)                 # 8-byte Spill
	jmp	.LBB5_16
.LBB5_1:                                # %if.then
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
	xorps	%xmm0, %xmm0
	jmp	.LBB5_31
.LBB5_22:                               # %if.else
	xorps	%xmm0, %xmm0
	movaps	%xmm0, 16(%rsp)
	cmpl	$0, 16(%rbx)
	jle	.LBB5_25
# %bb.23:                               # %for.body.i29.preheader
	xorl	%ebp, %ebp
	.p2align	4
.LBB5_24:                               # %for.body.i29
                                        # =>This Inner Loop Header: Depth=1
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
	incl	%ebp
	cmpl	16(%rbx), %ebp
	jl	.LBB5_24
.LBB5_25:                               # %for.cond.cleanup.i14
	leaq	16(%rsp), %rdi
	movq	%rbx, %rsi
	callq	_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t
	cmpl	$0, 20(%rbx)
	jle	.LBB5_29
# %bb.26:                               # %while.body.i18.preheader
	xorl	%ebp, %ebp
	.p2align	4
.LBB5_27:                               # %while.body.i18
                                        # =>This Inner Loop Header: Depth=1
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
	incl	%ebp
	cmpl	20(%rbx), %ebp
	jl	.LBB5_27
# %bb.28:                               # %if.end.i22
	leaq	16(%rsp), %rdi
	movq	%rbx, %rsi
	callq	_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t
	movq	24(%rsp), %rax
	subq	16(%rsp), %rax
	xorps	%xmm0, %xmm0
	cvtsi2sd	%rax, %xmm0
	divsd	.LCPI5_0(%rip), %xmm0
	mulsd	.LCPI5_1(%rip), %xmm0
	cvtsd2ss	%xmm0, %xmm0
	cvtsi2ssl	20(%rbx), %xmm1
	divss	%xmm1, %xmm0
	cvtss2sd	%xmm0, %xmm0
	jmp	.LBB5_30
.LBB5_29:                               # %while.end.thread.i27
	leaq	16(%rsp), %rdi
	movq	%rbx, %rsi
	callq	_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t
	xorps	%xmm0, %xmm0
	jmp	.LBB5_30
.LBB5_15:                               # %while.end.thread.i
	xorps	%xmm0, %xmm0
	movsd	%xmm0, 40(%rsp)                 # 8-byte Spill
.Ltmp150:
	movq	%rsp, %rdi
	movq	%rbx, %rsi
	callq	_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t
.Ltmp151:
.LBB5_16:                               # %invoke.cont
	movq	%rsp, %rdi
	callq	_ZN7ck_tile9gpu_timerD2Ev
	movsd	40(%rsp), %xmm0                 # 8-byte Reload
                                        # xmm0 = mem[0],zero
.LBB5_30:                               # %cleanup
	cvtsd2ss	%xmm0, %xmm0
.LBB5_31:                               # %return
	addq	$48, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB5_19:                               # %lpad.loopexit.split-lp.loopexit.split-lp
	.cfi_def_cfa_offset 80
.Ltmp152:
	jmp	.LBB5_20
.LBB5_17:                               # %lpad.loopexit
.Ltmp145:
	jmp	.LBB5_20
.LBB5_18:                               # %lpad.loopexit.split-lp.loopexit
.Ltmp140:
.LBB5_20:                               # %lpad
	movq	%rax, %rbx
.Ltmp153:
	movq	%rsp, %rdi
	callq	_ZN7ck_tile9gpu_timerD2Ev
.Ltmp154:
# %bb.21:                               # %invoke.cont3
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB5_32:                               # %terminate.lpad
.Ltmp155:
	movq	%rax, %rdi
	callq	__clang_call_terminate
.Lfunc_end5:
	.size	_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_, .Lfunc_end5-_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_,"aG",@progbits,_ZN7ck_tile13launch_kernelIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEfSD_DpOT_,comdat
	.p2align	2, 0x0
GCC_except_table5:
.Lexception4:
	.byte	255                             # @LPStart Encoding = omit
	.byte	3                               # @TType Encoding = udata4
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end4-.Lcst_begin4
.Lcst_begin4:
	.uleb128 .Lfunc_begin4-.Lfunc_begin4    # >> Call Site 1 <<
	.uleb128 .Ltmp138-.Lfunc_begin4         #   Call between .Lfunc_begin4 and .Ltmp138
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp138-.Lfunc_begin4         # >> Call Site 2 <<
	.uleb128 .Ltmp139-.Ltmp138              #   Call between .Ltmp138 and .Ltmp139
	.uleb128 .Ltmp140-.Lfunc_begin4         #     jumps to .Ltmp140
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp141-.Lfunc_begin4         # >> Call Site 3 <<
	.uleb128 .Ltmp142-.Ltmp141              #   Call between .Ltmp141 and .Ltmp142
	.uleb128 .Ltmp152-.Lfunc_begin4         #     jumps to .Ltmp152
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp143-.Lfunc_begin4         # >> Call Site 4 <<
	.uleb128 .Ltmp144-.Ltmp143              #   Call between .Ltmp143 and .Ltmp144
	.uleb128 .Ltmp145-.Lfunc_begin4         #     jumps to .Ltmp145
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp146-.Lfunc_begin4         # >> Call Site 5 <<
	.uleb128 .Ltmp149-.Ltmp146              #   Call between .Ltmp146 and .Ltmp149
	.uleb128 .Ltmp152-.Lfunc_begin4         #     jumps to .Ltmp152
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp149-.Lfunc_begin4         # >> Call Site 6 <<
	.uleb128 .Ltmp150-.Ltmp149              #   Call between .Ltmp149 and .Ltmp150
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp150-.Lfunc_begin4         # >> Call Site 7 <<
	.uleb128 .Ltmp151-.Ltmp150              #   Call between .Ltmp150 and .Ltmp151
	.uleb128 .Ltmp152-.Lfunc_begin4         #     jumps to .Ltmp152
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp151-.Lfunc_begin4         # >> Call Site 8 <<
	.uleb128 .Ltmp153-.Ltmp151              #   Call between .Ltmp151 and .Ltmp153
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp153-.Lfunc_begin4         # >> Call Site 9 <<
	.uleb128 .Ltmp154-.Ltmp153              #   Call between .Ltmp153 and .Ltmp154
	.uleb128 .Ltmp155-.Lfunc_begin4         #     jumps to .Ltmp155
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp154-.Lfunc_begin4         # >> Call Site 10 <<
	.uleb128 .Lfunc_end5-.Ltmp154           #   Call between .Ltmp154 and .Lfunc_end5
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end4:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	0                               # TypeInfo 1
.Lttbase0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile15hip_check_errorE10hipError_t,"axG",@progbits,_ZN7ck_tile15hip_check_errorE10hipError_t,comdat
	.weak	_ZN7ck_tile15hip_check_errorE10hipError_t # -- Begin function _ZN7ck_tile15hip_check_errorE10hipError_t
	.p2align	4
	.type	_ZN7ck_tile15hip_check_errorE10hipError_t,@function
_ZN7ck_tile15hip_check_errorE10hipError_t: # @_ZN7ck_tile15hip_check_errorE10hipError_t
.Lfunc_begin5:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception5
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	testl	%edi, %edi
	jne	.LBB6_1
# %bb.20:                               # %if.end
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB6_1:                                # %if.then
	.cfi_def_cfa_offset 448
	movl	%edi, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp156:
	movl	$.L.str.45, %esi
	movl	$19, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp157:
# %bb.2:                                # %invoke.cont
.Ltmp158:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp159:
# %bb.3:                                # %invoke.cont1
.Ltmp160:
	leaq	40(%rsp), %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp161:
# %bb.4:                                # %invoke.cont3
.Ltmp162:
	movq	%rax, %rbx
	movl	$.L.str.46, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp163:
# %bb.5:                                # %invoke.cont5
.Ltmp164:
	movl	$.L.str.47, %esi
	movl	$74, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp165:
# %bb.6:                                # %invoke.cont7
.Ltmp166:
	movl	$.L.str.48, %esi
	movl	$2, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp167:
# %bb.7:                                # %invoke.cont9
.Ltmp168:
	movq	%rbx, %rdi
	movl	$18, %esi
	callq	_ZNSolsEi
.Ltmp169:
# %bb.8:                                # %invoke.cont11
.Ltmp170:
	movq	%rax, %rbx
	movl	$.L.str.49, %esi
	movl	$13, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp171:
# %bb.9:                                # %invoke.cont13
.Ltmp172:
	movl	$.L__func__._ZN7ck_tile15hip_check_errorE10hipError_t, %esi
	movl	$15, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp173:
# %bb.10:                               # %invoke.cont15
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp175:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp176:
# %bb.11:                               # %invoke.cont18
	movb	$1, %bpl
.Ltmp178:
	leaq	8(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp179:
# %bb.12:                               # %invoke.cont20
	xorl	%ebp, %ebp
.Ltmp180:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp181:
# %bb.21:                               # %unreachable
.LBB6_15:                               # %lpad19
.Ltmp182:
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB6_16
# %bb.17:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB6_18
.LBB6_19:                               # %ehcleanup22
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB6_16:                               # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB6_19
	jmp	.LBB6_18
.LBB6_14:                               # %ehcleanup.thread
.Ltmp177:
	movq	%rax, %r14
.LBB6_18:                               # %cleanup.action
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB6_13:                               # %lpad
.Ltmp174:
	movq	%rax, %r14
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end6:
	.size	_ZN7ck_tile15hip_check_errorE10hipError_t, .Lfunc_end6-_ZN7ck_tile15hip_check_errorE10hipError_t
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile15hip_check_errorE10hipError_t,"aG",@progbits,_ZN7ck_tile15hip_check_errorE10hipError_t,comdat
	.p2align	2, 0x0
GCC_except_table6:
.Lexception5:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end5-.Lcst_begin5
.Lcst_begin5:
	.uleb128 .Lfunc_begin5-.Lfunc_begin5    # >> Call Site 1 <<
	.uleb128 .Ltmp156-.Lfunc_begin5         #   Call between .Lfunc_begin5 and .Ltmp156
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp156-.Lfunc_begin5         # >> Call Site 2 <<
	.uleb128 .Ltmp173-.Ltmp156              #   Call between .Ltmp156 and .Ltmp173
	.uleb128 .Ltmp174-.Lfunc_begin5         #     jumps to .Ltmp174
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp173-.Lfunc_begin5         # >> Call Site 3 <<
	.uleb128 .Ltmp175-.Ltmp173              #   Call between .Ltmp173 and .Ltmp175
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp175-.Lfunc_begin5         # >> Call Site 4 <<
	.uleb128 .Ltmp176-.Ltmp175              #   Call between .Ltmp175 and .Ltmp176
	.uleb128 .Ltmp177-.Lfunc_begin5         #     jumps to .Ltmp177
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp178-.Lfunc_begin5         # >> Call Site 5 <<
	.uleb128 .Ltmp181-.Ltmp178              #   Call between .Ltmp178 and .Ltmp181
	.uleb128 .Ltmp182-.Lfunc_begin5         #     jumps to .Ltmp182
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp181-.Lfunc_begin5         # >> Call Site 6 <<
	.uleb128 .Lfunc_end6-.Ltmp181           #   Call between .Ltmp181 and .Lfunc_end6
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end5:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNK7ck_tile9DeviceMem10FromDeviceEPvm,"axG",@progbits,_ZNK7ck_tile9DeviceMem10FromDeviceEPvm,comdat
	.weak	_ZNK7ck_tile9DeviceMem10FromDeviceEPvm # -- Begin function _ZNK7ck_tile9DeviceMem10FromDeviceEPvm
	.p2align	4
	.type	_ZNK7ck_tile9DeviceMem10FromDeviceEPvm,@function
_ZNK7ck_tile9DeviceMem10FromDeviceEPvm: # @_ZNK7ck_tile9DeviceMem10FromDeviceEPvm
.Lfunc_begin6:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception6
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	(%rdi), %rax
	testq	%rax, %rax
	je	.LBB7_19
# %bb.1:                                # %do.body
	movq	%rsi, %rdi
	movq	%rax, %rsi
	movl	$2, %ecx
	callq	hipMemcpy
	testl	%eax, %eax
	jne	.LBB7_2
.LBB7_19:                               # %if.end25
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB7_2:                                # %if.then3
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp183:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp184:
# %bb.3:                                # %invoke.cont
.Ltmp185:
	leaq	40(%rsp), %rdi
	movl	$.L.str.38, %esi
	movl	$72, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp186:
# %bb.4:                                # %invoke.cont5
.Ltmp187:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp188:
# %bb.5:                                # %invoke.cont7
.Ltmp189:
	leaq	40(%rsp), %rdi
	movl	$131, %esi
	callq	_ZNSolsEi
.Ltmp190:
# %bb.6:                                # %invoke.cont9
.Ltmp191:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp192:
# %bb.7:                                # %invoke.cont11
.Ltmp193:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp194:
# %bb.8:                                # %invoke.cont13
.Ltmp195:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp196:
# %bb.9:                                # %invoke.cont15
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp198:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp199:
# %bb.10:                               # %invoke.cont18
	movb	$1, %bpl
.Ltmp201:
	leaq	8(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp202:
# %bb.11:                               # %invoke.cont20
	xorl	%ebp, %ebp
.Ltmp203:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp204:
# %bb.20:                               # %unreachable
.LBB7_14:                               # %lpad19
.Ltmp205:
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB7_15
# %bb.16:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB7_17
.LBB7_18:                               # %ehcleanup22
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB7_15:                               # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB7_18
	jmp	.LBB7_17
.LBB7_13:                               # %ehcleanup.thread
.Ltmp200:
	movq	%rax, %r14
.LBB7_17:                               # %cleanup.action
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB7_12:                               # %lpad
.Ltmp197:
	movq	%rax, %r14
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end7:
	.size	_ZNK7ck_tile9DeviceMem10FromDeviceEPvm, .Lfunc_end7-_ZNK7ck_tile9DeviceMem10FromDeviceEPvm
	.cfi_endproc
	.section	.gcc_except_table._ZNK7ck_tile9DeviceMem10FromDeviceEPvm,"aG",@progbits,_ZNK7ck_tile9DeviceMem10FromDeviceEPvm,comdat
	.p2align	2, 0x0
GCC_except_table7:
.Lexception6:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end6-.Lcst_begin6
.Lcst_begin6:
	.uleb128 .Lfunc_begin6-.Lfunc_begin6    # >> Call Site 1 <<
	.uleb128 .Ltmp183-.Lfunc_begin6         #   Call between .Lfunc_begin6 and .Ltmp183
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp183-.Lfunc_begin6         # >> Call Site 2 <<
	.uleb128 .Ltmp196-.Ltmp183              #   Call between .Ltmp183 and .Ltmp196
	.uleb128 .Ltmp197-.Lfunc_begin6         #     jumps to .Ltmp197
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp196-.Lfunc_begin6         # >> Call Site 3 <<
	.uleb128 .Ltmp198-.Ltmp196              #   Call between .Ltmp196 and .Ltmp198
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp198-.Lfunc_begin6         # >> Call Site 4 <<
	.uleb128 .Ltmp199-.Ltmp198              #   Call between .Ltmp198 and .Ltmp199
	.uleb128 .Ltmp200-.Lfunc_begin6         #     jumps to .Ltmp200
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp201-.Lfunc_begin6         # >> Call Site 5 <<
	.uleb128 .Ltmp204-.Ltmp201              #   Call between .Ltmp201 and .Ltmp204
	.uleb128 .Ltmp205-.Lfunc_begin6         #     jumps to .Ltmp205
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp204-.Lfunc_begin6         # >> Call Site 6 <<
	.uleb128 .Lfunc_end7-.Ltmp204           #   Call between .Ltmp204 and .Lfunc_end7
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end6:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile9DeviceMemD2Ev,"axG",@progbits,_ZN7ck_tile9DeviceMemD2Ev,comdat
	.weak	_ZN7ck_tile9DeviceMemD2Ev       # -- Begin function _ZN7ck_tile9DeviceMemD2Ev
	.p2align	4
	.type	_ZN7ck_tile9DeviceMemD2Ev,@function
_ZN7ck_tile9DeviceMemD2Ev:              # @_ZN7ck_tile9DeviceMemD2Ev
.Lfunc_begin7:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception7
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	subq	$408, %rsp                      # imm = 0x198
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	(%rdi), %rdi
	testq	%rdi, %rdi
	je	.LBB8_28
# %bb.1:                                # %do.body
.Ltmp206:
	callq	hipFree
.Ltmp207:
# %bb.2:                                # %invoke.cont
	movl	%eax, %ebx
	testl	%eax, %eax
	jne	.LBB8_3
.LBB8_28:                               # %if.end35
	addq	$408, %rsp                      # imm = 0x198
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB8_3:                                # %if.then3
	.cfi_def_cfa_offset 448
.Ltmp209:
	leaq	32(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp210:
# %bb.4:                                # %invoke.cont5
.Ltmp212:
	leaq	32(%rsp), %rdi
	movl	$.L.str.37, %esi
	movl	$21, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp213:
# %bb.5:                                # %invoke.cont7
.Ltmp214:
	leaq	32(%rsp), %rdi
	movl	$.L.str.38, %esi
	movl	$72, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp215:
# %bb.6:                                # %invoke.cont9
.Ltmp216:
	leaq	32(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp217:
# %bb.7:                                # %invoke.cont11
.Ltmp218:
	leaq	32(%rsp), %rdi
	movl	$182, %esi
	callq	_ZNSolsEi
.Ltmp219:
# %bb.8:                                # %invoke.cont13
.Ltmp220:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp221:
# %bb.9:                                # %invoke.cont15
.Ltmp222:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp223:
# %bb.10:                               # %invoke.cont17
.Ltmp224:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp225:
# %bb.11:                               # %invoke.cont19
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp227:
	movq	%rsp, %rdi
	leaq	32(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp228:
# %bb.12:                               # %invoke.cont22
	movb	$1, %bpl
.Ltmp230:
	movq	%rsp, %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp231:
# %bb.13:                               # %invoke.cont24
	xorl	%ebp, %ebp
.Ltmp232:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp233:
# %bb.31:                               # %unreachable
.LBB8_19:                               # %lpad23
.Ltmp234:
	movq	%rdx, %r15
	movq	%rax, %rbx
	movq	(%rsp), %rdi
	leaq	16(%rsp), %rax
	cmpq	%rax, %rdi
	je	.LBB8_21
# %bb.20:                               # %ehcleanup
	movq	16(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
.LBB8_21:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB8_22
	jmp	.LBB8_23
.LBB8_18:                               # %ehcleanup.thread
.Ltmp229:
	movq	%rdx, %r15
	movq	%rax, %rbx
.LBB8_22:                               # %cleanup.action
	movq	%r14, %rdi
	callq	__cxa_free_exception
	jmp	.LBB8_23
.LBB8_17:                               # %lpad6
.Ltmp226:
	movq	%rdx, %r15
	movq	%rax, %rbx
.LBB8_23:                               # %ehcleanup26
	leaq	32(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	jmp	.LBB8_24
.LBB8_16:                               # %lpad4
.Ltmp211:
	jmp	.LBB8_15
.LBB8_14:                               # %lpad
.Ltmp208:
.LBB8_15:                               # %ehcleanup28
	movq	%rdx, %r15
	movq	%rax, %rbx
.LBB8_24:                               # %ehcleanup28
	movq	%rbx, %rdi
	cmpl	$2, %r15d
	jne	.LBB8_30
# %bb.25:                               # %catch
	callq	__cxa_begin_catch
	movq	(%rax), %rcx
	movq	%rax, %rdi
	callq	*16(%rcx)
.Ltmp235:
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp236:
# %bb.26:                               # %invoke.cont30
.Ltmp237:
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp238:
# %bb.27:                               # %invoke.cont32
.Ltmp239:
	callq	__cxa_end_catch
.Ltmp240:
	jmp	.LBB8_28
.LBB8_29:                               # %terminate.lpad
.Ltmp241:
	movq	%rax, %rdi
.LBB8_30:                               # %terminate.handler
	callq	__clang_call_terminate
.Lfunc_end8:
	.size	_ZN7ck_tile9DeviceMemD2Ev, .Lfunc_end8-_ZN7ck_tile9DeviceMemD2Ev
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9DeviceMemD2Ev,"aG",@progbits,_ZN7ck_tile9DeviceMemD2Ev,comdat
	.p2align	2, 0x0
GCC_except_table8:
.Lexception7:
	.byte	255                             # @LPStart Encoding = omit
	.byte	3                               # @TType Encoding = udata4
	.uleb128 .Lttbase1-.Lttbaseref1
.Lttbaseref1:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end7-.Lcst_begin7
.Lcst_begin7:
	.uleb128 .Ltmp206-.Lfunc_begin7         # >> Call Site 1 <<
	.uleb128 .Ltmp207-.Ltmp206              #   Call between .Ltmp206 and .Ltmp207
	.uleb128 .Ltmp208-.Lfunc_begin7         #     jumps to .Ltmp208
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp209-.Lfunc_begin7         # >> Call Site 2 <<
	.uleb128 .Ltmp210-.Ltmp209              #   Call between .Ltmp209 and .Ltmp210
	.uleb128 .Ltmp211-.Lfunc_begin7         #     jumps to .Ltmp211
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp212-.Lfunc_begin7         # >> Call Site 3 <<
	.uleb128 .Ltmp225-.Ltmp212              #   Call between .Ltmp212 and .Ltmp225
	.uleb128 .Ltmp226-.Lfunc_begin7         #     jumps to .Ltmp226
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp225-.Lfunc_begin7         # >> Call Site 4 <<
	.uleb128 .Ltmp227-.Ltmp225              #   Call between .Ltmp225 and .Ltmp227
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp227-.Lfunc_begin7         # >> Call Site 5 <<
	.uleb128 .Ltmp228-.Ltmp227              #   Call between .Ltmp227 and .Ltmp228
	.uleb128 .Ltmp229-.Lfunc_begin7         #     jumps to .Ltmp229
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp230-.Lfunc_begin7         # >> Call Site 6 <<
	.uleb128 .Ltmp233-.Ltmp230              #   Call between .Ltmp230 and .Ltmp233
	.uleb128 .Ltmp234-.Lfunc_begin7         #     jumps to .Ltmp234
	.byte	3                               #   On action: 2
	.uleb128 .Ltmp233-.Lfunc_begin7         # >> Call Site 7 <<
	.uleb128 .Ltmp235-.Ltmp233              #   Call between .Ltmp233 and .Ltmp235
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp235-.Lfunc_begin7         # >> Call Site 8 <<
	.uleb128 .Ltmp240-.Ltmp235              #   Call between .Ltmp235 and .Ltmp240
	.uleb128 .Ltmp241-.Lfunc_begin7         #     jumps to .Ltmp241
	.byte	1                               #   On action: 1
.Lcst_end7:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.byte	2                               # >> Action Record 2 <<
                                        #   Catch TypeInfo 2
	.byte	125                             #   Continue to action 1
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.long	_ZTISt13runtime_error           # TypeInfo 2
	.long	0                               # TypeInfo 1
.Lttbase1:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_,"axG",@progbits,_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_,comdat
	.weak	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_ # -- Begin function _ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
	.p2align	4
	.type	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_,@function
_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_: # @_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
.Lfunc_begin8:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception8
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$480, %rsp                      # imm = 0x1E0
	.cfi_def_cfa_offset 512
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	8(%rsi), %rax
	movl	16(%rsi), %esi
	movq	20(%rbx), %rdx
	movl	28(%rbx), %ecx
	movq	32(%rbx), %r8
	movq	(%rdi), %r9
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	__hipPushCallConfiguration
	testl	%eax, %eax
	je	.LBB9_1
# %bb.2:                                # %_ZZN7ck_tile11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S7_mDpT2_ENKUlRKNS_13stream_configEE_clESC_.exit
	.cfi_escape 0x2e, 0x00
	callq	hipPeekAtLastError
	testl	%eax, %eax
	jne	.LBB9_3
	jmp	.LBB9_21
.LBB9_1:                                # %kcall.configok.i
	movq	40(%rbx), %rax
	movq	48(%rbx), %rcx
	movl	56(%rbx), %edx
	movl	60(%rbx), %esi
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movl	%edx, 12(%rsp)
	movl	%esi, 8(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 120(%rsp)
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	64(%rsp), %rsi
	leaq	56(%rsp), %rdx
	leaq	48(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	16(%rsp), %rsi
	movl	24(%rsp), %edx
	movq	64(%rsp), %rcx
	movl	72(%rsp), %r8d
	.cfi_escape 0x2e, 0x10
	leaq	96(%rsp), %r9
	movl	$_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_, %edi
	pushq	48(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	64(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	.cfi_escape 0x2e, 0x00
	callq	hipPeekAtLastError
	testl	%eax, %eax
	je	.LBB9_21
.LBB9_3:                                # %do.body
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
	testl	%eax, %eax
	jne	.LBB9_4
.LBB9_21:                               # %if.end25
	addq	$480, %rsp                      # imm = 0x1E0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB9_4:                                # %if.then3
	.cfi_def_cfa_offset 512
	movl	%eax, %ebx
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp242:
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp243:
# %bb.5:                                # %invoke.cont
.Ltmp244:
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	movl	$.L.str.43, %esi
	movl	$72, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp245:
# %bb.6:                                # %invoke.cont5
.Ltmp246:
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp247:
# %bb.7:                                # %invoke.cont7
.Ltmp248:
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	movl	$83, %esi
	callq	_ZNSolsEi
.Ltmp249:
# %bb.8:                                # %invoke.cont9
.Ltmp250:
	movq	%rax, %r14
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp251:
# %bb.9:                                # %invoke.cont11
.Ltmp252:
	.cfi_escape 0x2e, 0x00
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp253:
# %bb.10:                               # %invoke.cont13
.Ltmp254:
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp255:
# %bb.11:                               # %invoke.cont15
	.cfi_escape 0x2e, 0x00
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp257:
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	leaq	96(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp258:
# %bb.12:                               # %invoke.cont18
	movb	$1, %bpl
.Ltmp260:
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp261:
# %bb.13:                               # %invoke.cont20
	xorl	%ebp, %ebp
.Ltmp262:
	.cfi_escape 0x2e, 0x00
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp263:
# %bb.22:                               # %unreachable
.LBB9_16:                               # %lpad19
.Ltmp264:
	movq	%rax, %r14
	movq	16(%rsp), %rdi
	leaq	32(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB9_17
# %bb.18:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB9_19
.LBB9_20:                               # %ehcleanup22
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB9_17:                               # %ehcleanup
	movq	32(%rsp), %rsi
	incq	%rsi
	.cfi_escape 0x2e, 0x00
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB9_20
	jmp	.LBB9_19
.LBB9_15:                               # %ehcleanup.thread
.Ltmp259:
	movq	%rax, %r14
.LBB9_19:                               # %cleanup.action
	.cfi_escape 0x2e, 0x00
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB9_14:                               # %lpad
.Ltmp256:
	movq	%rax, %r14
	.cfi_escape 0x2e, 0x00
	leaq	96(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end9:
	.size	_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_, .Lfunc_end9-_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_,"aG",@progbits,_ZN7ck_tile16launch_and_checkIJZNS_11make_kernelILi256Ev25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEDaT1_4dim3S8_mDpT2_EUlRKNS_13stream_configEE_EEEvSD_DpOT_,comdat
	.p2align	2, 0x0
GCC_except_table9:
.Lexception8:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end8-.Lcst_begin8
.Lcst_begin8:
	.uleb128 .Lfunc_begin8-.Lfunc_begin8    # >> Call Site 1 <<
	.uleb128 .Ltmp242-.Lfunc_begin8         #   Call between .Lfunc_begin8 and .Ltmp242
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp242-.Lfunc_begin8         # >> Call Site 2 <<
	.uleb128 .Ltmp255-.Ltmp242              #   Call between .Ltmp242 and .Ltmp255
	.uleb128 .Ltmp256-.Lfunc_begin8         #     jumps to .Ltmp256
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp255-.Lfunc_begin8         # >> Call Site 3 <<
	.uleb128 .Ltmp257-.Ltmp255              #   Call between .Ltmp255 and .Ltmp257
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp257-.Lfunc_begin8         # >> Call Site 4 <<
	.uleb128 .Ltmp258-.Ltmp257              #   Call between .Ltmp257 and .Ltmp258
	.uleb128 .Ltmp259-.Lfunc_begin8         #     jumps to .Ltmp259
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp260-.Lfunc_begin8         # >> Call Site 5 <<
	.uleb128 .Ltmp263-.Ltmp260              #   Call between .Ltmp260 and .Ltmp263
	.uleb128 .Ltmp264-.Lfunc_begin8         #     jumps to .Ltmp264
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp263-.Lfunc_begin8         # >> Call Site 6 <<
	.uleb128 .Lfunc_end9-.Ltmp263           #   Call between .Ltmp263 and .Lfunc_end9
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end8:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile9gpu_timerC2Ev,"axG",@progbits,_ZN7ck_tile9gpu_timerC2Ev,comdat
	.weak	_ZN7ck_tile9gpu_timerC2Ev       # -- Begin function _ZN7ck_tile9gpu_timerC2Ev
	.p2align	4
	.type	_ZN7ck_tile9gpu_timerC2Ev,@function
_ZN7ck_tile9gpu_timerC2Ev:              # @_ZN7ck_tile9gpu_timerC2Ev
.Lfunc_begin9:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception9
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	callq	hipEventCreate
	testl	%eax, %eax
	jne	.LBB10_3
# %bb.1:                                # %if.end
	addq	$8, %rbx
	movq	%rbx, %rdi
	callq	hipEventCreate
	testl	%eax, %eax
	jne	.LBB10_13
# %bb.2:                                # %if.end59
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB10_3:                               # %if.then
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp265:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp266:
# %bb.4:                                # %invoke.cont
.Ltmp267:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp268:
# %bb.5:                                # %invoke.cont3
.Ltmp269:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp270:
# %bb.6:                                # %invoke.cont5
.Ltmp271:
	leaq	40(%rsp), %rdi
	movl	$18, %esi
	callq	_ZNSolsEi
.Ltmp272:
# %bb.7:                                # %invoke.cont7
.Ltmp273:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp274:
# %bb.8:                                # %invoke.cont9
.Ltmp275:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp276:
# %bb.9:                                # %invoke.cont11
.Ltmp277:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp278:
# %bb.10:                               # %invoke.cont13
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp280:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp281:
# %bb.11:                               # %invoke.cont16
	movb	$1, %bpl
.Ltmp283:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp284:
# %bb.12:                               # %invoke.cont18
	xorl	%ebp, %ebp
.Ltmp285:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp286:
	jmp	.LBB10_23
.LBB10_13:                              # %if.then27
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp288:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp289:
# %bb.14:                               # %invoke.cont30
.Ltmp290:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp291:
# %bb.15:                               # %invoke.cont32
.Ltmp292:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp293:
# %bb.16:                               # %invoke.cont34
.Ltmp294:
	leaq	40(%rsp), %rdi
	movl	$19, %esi
	callq	_ZNSolsEi
.Ltmp295:
# %bb.17:                               # %invoke.cont36
.Ltmp296:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp297:
# %bb.18:                               # %invoke.cont38
.Ltmp298:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp299:
# %bb.19:                               # %invoke.cont40
.Ltmp300:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp301:
# %bb.20:                               # %invoke.cont42
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp303:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp304:
# %bb.21:                               # %invoke.cont47
	movb	$1, %bpl
.Ltmp306:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp307:
# %bb.22:                               # %invoke.cont49
	xorl	%ebp, %ebp
.Ltmp308:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp309:
.LBB10_23:                              # %unreachable
.LBB10_24:                              # %lpad48
.Ltmp310:
	jmp	.LBB10_27
.LBB10_25:                              # %ehcleanup52.thread
.Ltmp305:
	jmp	.LBB10_32
.LBB10_26:                              # %lpad17
.Ltmp287:
.LBB10_27:                              # %lpad17
	movq	%rax, %rbx
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB10_30
# %bb.28:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB10_33
	jmp	.LBB10_29
.LBB10_30:                              # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	jne	.LBB10_33
.LBB10_29:                              # %ehcleanup20
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB10_31:                              # %ehcleanup.thread
.Ltmp282:
.LBB10_32:                              # %cleanup.action
	movq	%rax, %rbx
.LBB10_33:                              # %cleanup.action
	movq	%r14, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB10_34:                              # %lpad29
.Ltmp302:
	jmp	.LBB10_36
.LBB10_35:                              # %lpad
.Ltmp279:
.LBB10_36:                              # %lpad
	movq	%rax, %rbx
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end10:
	.size	_ZN7ck_tile9gpu_timerC2Ev, .Lfunc_end10-_ZN7ck_tile9gpu_timerC2Ev
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9gpu_timerC2Ev,"aG",@progbits,_ZN7ck_tile9gpu_timerC2Ev,comdat
	.p2align	2, 0x0
GCC_except_table10:
.Lexception9:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end9-.Lcst_begin9
.Lcst_begin9:
	.uleb128 .Lfunc_begin9-.Lfunc_begin9    # >> Call Site 1 <<
	.uleb128 .Ltmp265-.Lfunc_begin9         #   Call between .Lfunc_begin9 and .Ltmp265
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp265-.Lfunc_begin9         # >> Call Site 2 <<
	.uleb128 .Ltmp278-.Ltmp265              #   Call between .Ltmp265 and .Ltmp278
	.uleb128 .Ltmp279-.Lfunc_begin9         #     jumps to .Ltmp279
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp278-.Lfunc_begin9         # >> Call Site 3 <<
	.uleb128 .Ltmp280-.Ltmp278              #   Call between .Ltmp278 and .Ltmp280
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp280-.Lfunc_begin9         # >> Call Site 4 <<
	.uleb128 .Ltmp281-.Ltmp280              #   Call between .Ltmp280 and .Ltmp281
	.uleb128 .Ltmp282-.Lfunc_begin9         #     jumps to .Ltmp282
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp283-.Lfunc_begin9         # >> Call Site 5 <<
	.uleb128 .Ltmp286-.Ltmp283              #   Call between .Ltmp283 and .Ltmp286
	.uleb128 .Ltmp287-.Lfunc_begin9         #     jumps to .Ltmp287
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp286-.Lfunc_begin9         # >> Call Site 6 <<
	.uleb128 .Ltmp288-.Ltmp286              #   Call between .Ltmp286 and .Ltmp288
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp288-.Lfunc_begin9         # >> Call Site 7 <<
	.uleb128 .Ltmp301-.Ltmp288              #   Call between .Ltmp288 and .Ltmp301
	.uleb128 .Ltmp302-.Lfunc_begin9         #     jumps to .Ltmp302
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp301-.Lfunc_begin9         # >> Call Site 8 <<
	.uleb128 .Ltmp303-.Ltmp301              #   Call between .Ltmp301 and .Ltmp303
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp303-.Lfunc_begin9         # >> Call Site 9 <<
	.uleb128 .Ltmp304-.Ltmp303              #   Call between .Ltmp303 and .Ltmp304
	.uleb128 .Ltmp305-.Lfunc_begin9         #     jumps to .Ltmp305
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp306-.Lfunc_begin9         # >> Call Site 10 <<
	.uleb128 .Ltmp309-.Ltmp306              #   Call between .Ltmp306 and .Ltmp309
	.uleb128 .Ltmp310-.Lfunc_begin9         #     jumps to .Ltmp310
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp309-.Lfunc_begin9         # >> Call Site 11 <<
	.uleb128 .Lfunc_end10-.Ltmp309          #   Call between .Ltmp309 and .Lfunc_end10
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end9:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile9gpu_timerD2Ev,"axG",@progbits,_ZN7ck_tile9gpu_timerD2Ev,comdat
	.weak	_ZN7ck_tile9gpu_timerD2Ev       # -- Begin function _ZN7ck_tile9gpu_timerD2Ev
	.p2align	4
	.type	_ZN7ck_tile9gpu_timerD2Ev,@function
_ZN7ck_tile9gpu_timerD2Ev:              # @_ZN7ck_tile9gpu_timerD2Ev
.Lfunc_begin10:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception10
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	movq	(%rdi), %rdi
	callq	hipEventDestroy
	testl	%eax, %eax
	jne	.LBB11_3
# %bb.1:                                # %if.end
	movq	8(%rbx), %rdi
	callq	hipEventDestroy
	testl	%eax, %eax
	jne	.LBB11_13
# %bb.2:                                # %if.end59
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB11_3:                               # %if.then
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp311:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp312:
# %bb.4:                                # %invoke.cont
.Ltmp313:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp314:
# %bb.5:                                # %invoke.cont3
.Ltmp315:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp316:
# %bb.6:                                # %invoke.cont5
.Ltmp317:
	leaq	40(%rsp), %rdi
	movl	$24, %esi
	callq	_ZNSolsEi
.Ltmp318:
# %bb.7:                                # %invoke.cont7
.Ltmp319:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp320:
# %bb.8:                                # %invoke.cont9
.Ltmp321:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp322:
# %bb.9:                                # %invoke.cont11
.Ltmp323:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp324:
# %bb.10:                               # %invoke.cont13
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp326:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp327:
# %bb.11:                               # %invoke.cont16
	movb	$1, %bpl
.Ltmp329:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp330:
# %bb.12:                               # %invoke.cont18
	xorl	%ebp, %ebp
.Ltmp331:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp332:
	jmp	.LBB11_23
.LBB11_13:                              # %if.then27
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp334:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp335:
# %bb.14:                               # %invoke.cont30
.Ltmp336:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp337:
# %bb.15:                               # %invoke.cont32
.Ltmp338:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp339:
# %bb.16:                               # %invoke.cont34
.Ltmp340:
	leaq	40(%rsp), %rdi
	movl	$25, %esi
	callq	_ZNSolsEi
.Ltmp341:
# %bb.17:                               # %invoke.cont36
.Ltmp342:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp343:
# %bb.18:                               # %invoke.cont38
.Ltmp344:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp345:
# %bb.19:                               # %invoke.cont40
.Ltmp346:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp347:
# %bb.20:                               # %invoke.cont42
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp349:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp350:
# %bb.21:                               # %invoke.cont47
	movb	$1, %bpl
.Ltmp352:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp353:
# %bb.22:                               # %invoke.cont49
	xorl	%ebp, %ebp
.Ltmp354:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp355:
.LBB11_23:                              # %unreachable
.LBB11_24:                              # %lpad48
.Ltmp356:
	jmp	.LBB11_27
.LBB11_25:                              # %ehcleanup52.thread
.Ltmp351:
	jmp	.LBB11_32
.LBB11_26:                              # %lpad17
.Ltmp333:
.LBB11_27:                              # %lpad17
	movq	%rax, %rbx
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB11_30
# %bb.28:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB11_33
	jmp	.LBB11_29
.LBB11_30:                              # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	jne	.LBB11_33
.LBB11_29:                              # %ehcleanup20
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB11_31:                              # %ehcleanup.thread
.Ltmp328:
.LBB11_32:                              # %cleanup.action
	movq	%rax, %rbx
.LBB11_33:                              # %cleanup.action
	movq	%r14, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB11_34:                              # %lpad29
.Ltmp348:
	jmp	.LBB11_36
.LBB11_35:                              # %lpad
.Ltmp325:
.LBB11_36:                              # %lpad
	movq	%rax, %rbx
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end11:
	.size	_ZN7ck_tile9gpu_timerD2Ev, .Lfunc_end11-_ZN7ck_tile9gpu_timerD2Ev
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9gpu_timerD2Ev,"aG",@progbits,_ZN7ck_tile9gpu_timerD2Ev,comdat
	.p2align	2, 0x0
GCC_except_table11:
.Lexception10:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end10-.Lcst_begin10
.Lcst_begin10:
	.uleb128 .Lfunc_begin10-.Lfunc_begin10  # >> Call Site 1 <<
	.uleb128 .Ltmp311-.Lfunc_begin10        #   Call between .Lfunc_begin10 and .Ltmp311
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp311-.Lfunc_begin10        # >> Call Site 2 <<
	.uleb128 .Ltmp324-.Ltmp311              #   Call between .Ltmp311 and .Ltmp324
	.uleb128 .Ltmp325-.Lfunc_begin10        #     jumps to .Ltmp325
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp324-.Lfunc_begin10        # >> Call Site 3 <<
	.uleb128 .Ltmp326-.Ltmp324              #   Call between .Ltmp324 and .Ltmp326
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp326-.Lfunc_begin10        # >> Call Site 4 <<
	.uleb128 .Ltmp327-.Ltmp326              #   Call between .Ltmp326 and .Ltmp327
	.uleb128 .Ltmp328-.Lfunc_begin10        #     jumps to .Ltmp328
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp329-.Lfunc_begin10        # >> Call Site 5 <<
	.uleb128 .Ltmp332-.Ltmp329              #   Call between .Ltmp329 and .Ltmp332
	.uleb128 .Ltmp333-.Lfunc_begin10        #     jumps to .Ltmp333
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp332-.Lfunc_begin10        # >> Call Site 6 <<
	.uleb128 .Ltmp334-.Ltmp332              #   Call between .Ltmp332 and .Ltmp334
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp334-.Lfunc_begin10        # >> Call Site 7 <<
	.uleb128 .Ltmp347-.Ltmp334              #   Call between .Ltmp334 and .Ltmp347
	.uleb128 .Ltmp348-.Lfunc_begin10        #     jumps to .Ltmp348
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp347-.Lfunc_begin10        # >> Call Site 8 <<
	.uleb128 .Ltmp349-.Ltmp347              #   Call between .Ltmp347 and .Ltmp349
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp349-.Lfunc_begin10        # >> Call Site 9 <<
	.uleb128 .Ltmp350-.Ltmp349              #   Call between .Ltmp349 and .Ltmp350
	.uleb128 .Ltmp351-.Lfunc_begin10        #     jumps to .Ltmp351
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp352-.Lfunc_begin10        # >> Call Site 10 <<
	.uleb128 .Ltmp355-.Ltmp352              #   Call between .Ltmp352 and .Ltmp355
	.uleb128 .Ltmp356-.Lfunc_begin10        #     jumps to .Ltmp356
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp355-.Lfunc_begin10        # >> Call Site 11 <<
	.uleb128 .Lfunc_end11-.Ltmp355          #   Call between .Ltmp355 and .Lfunc_end11
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end10:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,"axG",@progbits,_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,comdat
	.weak	_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_ # -- Begin function _ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.p2align	4
	.type	_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,@function
_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_: # @_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	movl	%ecx, 8(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end12:
	.size	_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_, .Lfunc_end12-_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN7ck_tile9gpu_timer5startERKP12ihipStream_t,"axG",@progbits,_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t,comdat
	.weak	_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t # -- Begin function _ZN7ck_tile9gpu_timer5startERKP12ihipStream_t
	.p2align	4
	.type	_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t,@function
_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t: # @_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t
.Lfunc_begin11:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception11
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rsi, %rbx
	movq	%rdi, %r14
	movq	(%rsi), %rdi
	callq	hipStreamSynchronize
	testl	%eax, %eax
	jne	.LBB13_3
# %bb.1:                                # %if.end
	movq	(%r14), %rdi
	movq	(%rbx), %rsi
	callq	hipEventRecord
	testl	%eax, %eax
	jne	.LBB13_13
# %bb.2:                                # %if.end59
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB13_3:                               # %if.then
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp357:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp358:
# %bb.4:                                # %invoke.cont
.Ltmp359:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp360:
# %bb.5:                                # %invoke.cont3
.Ltmp361:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp362:
# %bb.6:                                # %invoke.cont5
.Ltmp363:
	leaq	40(%rsp), %rdi
	movl	$30, %esi
	callq	_ZNSolsEi
.Ltmp364:
# %bb.7:                                # %invoke.cont7
.Ltmp365:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp366:
# %bb.8:                                # %invoke.cont9
.Ltmp367:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp368:
# %bb.9:                                # %invoke.cont11
.Ltmp369:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp370:
# %bb.10:                               # %invoke.cont13
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp372:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp373:
# %bb.11:                               # %invoke.cont16
	movb	$1, %bpl
.Ltmp375:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp376:
# %bb.12:                               # %invoke.cont18
	xorl	%ebp, %ebp
.Ltmp377:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp378:
	jmp	.LBB13_23
.LBB13_13:                              # %if.then27
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp380:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp381:
# %bb.14:                               # %invoke.cont30
.Ltmp382:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp383:
# %bb.15:                               # %invoke.cont32
.Ltmp384:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp385:
# %bb.16:                               # %invoke.cont34
.Ltmp386:
	leaq	40(%rsp), %rdi
	movl	$31, %esi
	callq	_ZNSolsEi
.Ltmp387:
# %bb.17:                               # %invoke.cont36
.Ltmp388:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp389:
# %bb.18:                               # %invoke.cont38
.Ltmp390:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp391:
# %bb.19:                               # %invoke.cont40
.Ltmp392:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp393:
# %bb.20:                               # %invoke.cont42
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp395:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp396:
# %bb.21:                               # %invoke.cont47
	movb	$1, %bpl
.Ltmp398:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp399:
# %bb.22:                               # %invoke.cont49
	xorl	%ebp, %ebp
.Ltmp400:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp401:
.LBB13_23:                              # %unreachable
.LBB13_24:                              # %lpad48
.Ltmp402:
	jmp	.LBB13_27
.LBB13_25:                              # %ehcleanup52.thread
.Ltmp397:
	jmp	.LBB13_32
.LBB13_26:                              # %lpad17
.Ltmp379:
.LBB13_27:                              # %lpad17
	movq	%rax, %rbx
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB13_30
# %bb.28:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB13_33
	jmp	.LBB13_29
.LBB13_30:                              # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	jne	.LBB13_33
.LBB13_29:                              # %ehcleanup20
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB13_31:                              # %ehcleanup.thread
.Ltmp374:
.LBB13_32:                              # %cleanup.action
	movq	%rax, %rbx
.LBB13_33:                              # %cleanup.action
	movq	%r14, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB13_34:                              # %lpad29
.Ltmp394:
	jmp	.LBB13_36
.LBB13_35:                              # %lpad
.Ltmp371:
.LBB13_36:                              # %lpad
	movq	%rax, %rbx
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end13:
	.size	_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t, .Lfunc_end13-_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9gpu_timer5startERKP12ihipStream_t,"aG",@progbits,_ZN7ck_tile9gpu_timer5startERKP12ihipStream_t,comdat
	.p2align	2, 0x0
GCC_except_table13:
.Lexception11:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end11-.Lcst_begin11
.Lcst_begin11:
	.uleb128 .Lfunc_begin11-.Lfunc_begin11  # >> Call Site 1 <<
	.uleb128 .Ltmp357-.Lfunc_begin11        #   Call between .Lfunc_begin11 and .Ltmp357
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp357-.Lfunc_begin11        # >> Call Site 2 <<
	.uleb128 .Ltmp370-.Ltmp357              #   Call between .Ltmp357 and .Ltmp370
	.uleb128 .Ltmp371-.Lfunc_begin11        #     jumps to .Ltmp371
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp370-.Lfunc_begin11        # >> Call Site 3 <<
	.uleb128 .Ltmp372-.Ltmp370              #   Call between .Ltmp370 and .Ltmp372
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp372-.Lfunc_begin11        # >> Call Site 4 <<
	.uleb128 .Ltmp373-.Ltmp372              #   Call between .Ltmp372 and .Ltmp373
	.uleb128 .Ltmp374-.Lfunc_begin11        #     jumps to .Ltmp374
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp375-.Lfunc_begin11        # >> Call Site 5 <<
	.uleb128 .Ltmp378-.Ltmp375              #   Call between .Ltmp375 and .Ltmp378
	.uleb128 .Ltmp379-.Lfunc_begin11        #     jumps to .Ltmp379
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp378-.Lfunc_begin11        # >> Call Site 6 <<
	.uleb128 .Ltmp380-.Ltmp378              #   Call between .Ltmp378 and .Ltmp380
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp380-.Lfunc_begin11        # >> Call Site 7 <<
	.uleb128 .Ltmp393-.Ltmp380              #   Call between .Ltmp380 and .Ltmp393
	.uleb128 .Ltmp394-.Lfunc_begin11        #     jumps to .Ltmp394
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp393-.Lfunc_begin11        # >> Call Site 8 <<
	.uleb128 .Ltmp395-.Ltmp393              #   Call between .Ltmp393 and .Ltmp395
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp395-.Lfunc_begin11        # >> Call Site 9 <<
	.uleb128 .Ltmp396-.Ltmp395              #   Call between .Ltmp395 and .Ltmp396
	.uleb128 .Ltmp397-.Lfunc_begin11        #     jumps to .Ltmp397
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp398-.Lfunc_begin11        # >> Call Site 10 <<
	.uleb128 .Ltmp401-.Ltmp398              #   Call between .Ltmp398 and .Ltmp401
	.uleb128 .Ltmp402-.Lfunc_begin11        #     jumps to .Ltmp402
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp401-.Lfunc_begin11        # >> Call Site 11 <<
	.uleb128 .Lfunc_end13-.Ltmp401          #   Call between .Ltmp401 and .Lfunc_end13
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end11:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t,"axG",@progbits,_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t,comdat
	.weak	_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t # -- Begin function _ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t
	.p2align	4
	.type	_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t,@function
_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t: # @_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t
.Lfunc_begin12:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception12
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	movq	8(%rdi), %rdi
	movq	(%rsi), %rsi
	callq	hipEventRecord
	testl	%eax, %eax
	jne	.LBB14_3
# %bb.1:                                # %if.end
	movq	8(%rbx), %rdi
	callq	hipEventSynchronize
	testl	%eax, %eax
	jne	.LBB14_13
# %bb.2:                                # %if.end60
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB14_3:                               # %if.then
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp403:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp404:
# %bb.4:                                # %invoke.cont
.Ltmp405:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp406:
# %bb.5:                                # %invoke.cont3
.Ltmp407:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp408:
# %bb.6:                                # %invoke.cont5
.Ltmp409:
	leaq	40(%rsp), %rdi
	movl	$36, %esi
	callq	_ZNSolsEi
.Ltmp410:
# %bb.7:                                # %invoke.cont7
.Ltmp411:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp412:
# %bb.8:                                # %invoke.cont9
.Ltmp413:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp414:
# %bb.9:                                # %invoke.cont11
.Ltmp415:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp416:
# %bb.10:                               # %invoke.cont13
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp418:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp419:
# %bb.11:                               # %invoke.cont16
	movb	$1, %bpl
.Ltmp421:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp422:
# %bb.12:                               # %invoke.cont18
	xorl	%ebp, %ebp
.Ltmp423:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp424:
	jmp	.LBB14_23
.LBB14_13:                              # %if.then28
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp426:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp427:
# %bb.14:                               # %invoke.cont31
.Ltmp428:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp429:
# %bb.15:                               # %invoke.cont33
.Ltmp430:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp431:
# %bb.16:                               # %invoke.cont35
.Ltmp432:
	leaq	40(%rsp), %rdi
	movl	$37, %esi
	callq	_ZNSolsEi
.Ltmp433:
# %bb.17:                               # %invoke.cont37
.Ltmp434:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp435:
# %bb.18:                               # %invoke.cont39
.Ltmp436:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp437:
# %bb.19:                               # %invoke.cont41
.Ltmp438:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp439:
# %bb.20:                               # %invoke.cont43
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %r14
.Ltmp441:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp442:
# %bb.21:                               # %invoke.cont48
	movb	$1, %bpl
.Ltmp444:
	leaq	8(%rsp), %rsi
	movq	%r14, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp445:
# %bb.22:                               # %invoke.cont50
	xorl	%ebp, %ebp
.Ltmp446:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%r14, %rdi
	callq	__cxa_throw
.Ltmp447:
.LBB14_23:                              # %unreachable
.LBB14_24:                              # %lpad49
.Ltmp448:
	jmp	.LBB14_27
.LBB14_25:                              # %ehcleanup53.thread
.Ltmp443:
	jmp	.LBB14_32
.LBB14_26:                              # %lpad17
.Ltmp425:
.LBB14_27:                              # %lpad17
	movq	%rax, %rbx
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB14_30
# %bb.28:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB14_33
	jmp	.LBB14_29
.LBB14_30:                              # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	jne	.LBB14_33
.LBB14_29:                              # %ehcleanup20
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB14_31:                              # %ehcleanup.thread
.Ltmp420:
.LBB14_32:                              # %cleanup.action
	movq	%rax, %rbx
.LBB14_33:                              # %cleanup.action
	movq	%r14, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.LBB14_34:                              # %lpad30
.Ltmp440:
	jmp	.LBB14_36
.LBB14_35:                              # %lpad
.Ltmp417:
.LBB14_36:                              # %lpad
	movq	%rax, %rbx
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end14:
	.size	_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t, .Lfunc_end14-_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t,"aG",@progbits,_ZN7ck_tile9gpu_timer4stopERKP12ihipStream_t,comdat
	.p2align	2, 0x0
GCC_except_table14:
.Lexception12:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end12-.Lcst_begin12
.Lcst_begin12:
	.uleb128 .Lfunc_begin12-.Lfunc_begin12  # >> Call Site 1 <<
	.uleb128 .Ltmp403-.Lfunc_begin12        #   Call between .Lfunc_begin12 and .Ltmp403
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp403-.Lfunc_begin12        # >> Call Site 2 <<
	.uleb128 .Ltmp416-.Ltmp403              #   Call between .Ltmp403 and .Ltmp416
	.uleb128 .Ltmp417-.Lfunc_begin12        #     jumps to .Ltmp417
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp416-.Lfunc_begin12        # >> Call Site 3 <<
	.uleb128 .Ltmp418-.Ltmp416              #   Call between .Ltmp416 and .Ltmp418
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp418-.Lfunc_begin12        # >> Call Site 4 <<
	.uleb128 .Ltmp419-.Ltmp418              #   Call between .Ltmp418 and .Ltmp419
	.uleb128 .Ltmp420-.Lfunc_begin12        #     jumps to .Ltmp420
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp421-.Lfunc_begin12        # >> Call Site 5 <<
	.uleb128 .Ltmp424-.Ltmp421              #   Call between .Ltmp421 and .Ltmp424
	.uleb128 .Ltmp425-.Lfunc_begin12        #     jumps to .Ltmp425
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp424-.Lfunc_begin12        # >> Call Site 6 <<
	.uleb128 .Ltmp426-.Ltmp424              #   Call between .Ltmp424 and .Ltmp426
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp426-.Lfunc_begin12        # >> Call Site 7 <<
	.uleb128 .Ltmp439-.Ltmp426              #   Call between .Ltmp426 and .Ltmp439
	.uleb128 .Ltmp440-.Lfunc_begin12        #     jumps to .Ltmp440
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp439-.Lfunc_begin12        # >> Call Site 8 <<
	.uleb128 .Ltmp441-.Ltmp439              #   Call between .Ltmp439 and .Ltmp441
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp441-.Lfunc_begin12        # >> Call Site 9 <<
	.uleb128 .Ltmp442-.Ltmp441              #   Call between .Ltmp441 and .Ltmp442
	.uleb128 .Ltmp443-.Lfunc_begin12        #     jumps to .Ltmp443
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp444-.Lfunc_begin12        # >> Call Site 10 <<
	.uleb128 .Ltmp447-.Ltmp444              #   Call between .Ltmp444 and .Ltmp447
	.uleb128 .Ltmp448-.Lfunc_begin12        #     jumps to .Ltmp448
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp447-.Lfunc_begin12        # >> Call Site 11 <<
	.uleb128 .Lfunc_end14-.Ltmp447          #   Call between .Ltmp447 and .Lfunc_end14
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end12:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNK7ck_tile9gpu_timer8durationEv,"axG",@progbits,_ZNK7ck_tile9gpu_timer8durationEv,comdat
	.weak	_ZNK7ck_tile9gpu_timer8durationEv # -- Begin function _ZNK7ck_tile9gpu_timer8durationEv
	.p2align	4
	.type	_ZNK7ck_tile9gpu_timer8durationEv,@function
_ZNK7ck_tile9gpu_timer8durationEv:      # @_ZNK7ck_tile9gpu_timer8durationEv
.Lfunc_begin13:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception13
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movl	$0, 4(%rsp)
	movq	(%rdi), %rsi
	movq	8(%rdi), %rdx
	leaq	4(%rsp), %rdi
	callq	hipEventElapsedTime
	testl	%eax, %eax
	jne	.LBB15_1
# %bb.18:                               # %if.end
	movss	4(%rsp), %xmm0                  # xmm0 = mem[0],zero,zero,zero
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB15_1:                               # %if.then
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp449:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp450:
# %bb.2:                                # %invoke.cont
.Ltmp451:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp452:
# %bb.3:                                # %invoke.cont3
.Ltmp453:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp454:
# %bb.4:                                # %invoke.cont5
.Ltmp455:
	leaq	40(%rsp), %rdi
	movl	$43, %esi
	callq	_ZNSolsEi
.Ltmp456:
# %bb.5:                                # %invoke.cont7
.Ltmp457:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp458:
# %bb.6:                                # %invoke.cont9
.Ltmp459:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp460:
# %bb.7:                                # %invoke.cont11
.Ltmp461:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp462:
# %bb.8:                                # %invoke.cont13
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp464:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp465:
# %bb.9:                                # %invoke.cont16
	movb	$1, %bpl
.Ltmp467:
	leaq	8(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp468:
# %bb.10:                               # %invoke.cont18
	xorl	%ebp, %ebp
.Ltmp469:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp470:
# %bb.19:                               # %unreachable
.LBB15_13:                              # %lpad17
.Ltmp471:
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB15_14
# %bb.15:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB15_16
.LBB15_17:                              # %ehcleanup20
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB15_14:                              # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB15_17
	jmp	.LBB15_16
.LBB15_12:                              # %ehcleanup.thread
.Ltmp466:
	movq	%rax, %r14
.LBB15_16:                              # %cleanup.action
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB15_11:                              # %lpad
.Ltmp463:
	movq	%rax, %r14
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end15:
	.size	_ZNK7ck_tile9gpu_timer8durationEv, .Lfunc_end15-_ZNK7ck_tile9gpu_timer8durationEv
	.cfi_endproc
	.section	.gcc_except_table._ZNK7ck_tile9gpu_timer8durationEv,"aG",@progbits,_ZNK7ck_tile9gpu_timer8durationEv,comdat
	.p2align	2, 0x0
GCC_except_table15:
.Lexception13:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end13-.Lcst_begin13
.Lcst_begin13:
	.uleb128 .Lfunc_begin13-.Lfunc_begin13  # >> Call Site 1 <<
	.uleb128 .Ltmp449-.Lfunc_begin13        #   Call between .Lfunc_begin13 and .Ltmp449
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp449-.Lfunc_begin13        # >> Call Site 2 <<
	.uleb128 .Ltmp462-.Ltmp449              #   Call between .Ltmp449 and .Ltmp462
	.uleb128 .Ltmp463-.Lfunc_begin13        #     jumps to .Ltmp463
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp462-.Lfunc_begin13        # >> Call Site 3 <<
	.uleb128 .Ltmp464-.Ltmp462              #   Call between .Ltmp462 and .Ltmp464
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp464-.Lfunc_begin13        # >> Call Site 4 <<
	.uleb128 .Ltmp465-.Ltmp464              #   Call between .Ltmp464 and .Ltmp465
	.uleb128 .Ltmp466-.Lfunc_begin13        #     jumps to .Ltmp466
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp467-.Lfunc_begin13        # >> Call Site 5 <<
	.uleb128 .Ltmp470-.Ltmp467              #   Call between .Ltmp467 and .Ltmp470
	.uleb128 .Ltmp471-.Lfunc_begin13        #     jumps to .Ltmp471
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp470-.Lfunc_begin13        # >> Call Site 6 <<
	.uleb128 .Lfunc_end15-.Ltmp470          #   Call between .Ltmp470 and .Lfunc_end15
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end13:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile9cpu_timer5startERKP12ihipStream_t,"axG",@progbits,_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t,comdat
	.weak	_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t # -- Begin function _ZN7ck_tile9cpu_timer5startERKP12ihipStream_t
	.p2align	4
	.type	_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t,@function
_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t: # @_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t
.Lfunc_begin14:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception14
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	movq	(%rsi), %rdi
	callq	hipStreamSynchronize
	testl	%eax, %eax
	jne	.LBB16_1
# %bb.18:                               # %if.end
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, (%rbx)
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB16_1:                               # %if.then
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp472:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp473:
# %bb.2:                                # %invoke.cont
.Ltmp474:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp475:
# %bb.3:                                # %invoke.cont3
.Ltmp476:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp477:
# %bb.4:                                # %invoke.cont5
.Ltmp478:
	leaq	40(%rsp), %rdi
	movl	$56, %esi
	callq	_ZNSolsEi
.Ltmp479:
# %bb.5:                                # %invoke.cont7
.Ltmp480:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp481:
# %bb.6:                                # %invoke.cont9
.Ltmp482:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp483:
# %bb.7:                                # %invoke.cont11
.Ltmp484:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp485:
# %bb.8:                                # %invoke.cont13
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp487:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp488:
# %bb.9:                                # %invoke.cont16
	movb	$1, %bpl
.Ltmp490:
	leaq	8(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp491:
# %bb.10:                               # %invoke.cont18
	xorl	%ebp, %ebp
.Ltmp492:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp493:
# %bb.19:                               # %unreachable
.LBB16_13:                              # %lpad17
.Ltmp494:
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB16_14
# %bb.15:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB16_16
.LBB16_17:                              # %ehcleanup20
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB16_14:                              # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB16_17
	jmp	.LBB16_16
.LBB16_12:                              # %ehcleanup.thread
.Ltmp489:
	movq	%rax, %r14
.LBB16_16:                              # %cleanup.action
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB16_11:                              # %lpad
.Ltmp486:
	movq	%rax, %r14
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end16:
	.size	_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t, .Lfunc_end16-_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9cpu_timer5startERKP12ihipStream_t,"aG",@progbits,_ZN7ck_tile9cpu_timer5startERKP12ihipStream_t,comdat
	.p2align	2, 0x0
GCC_except_table16:
.Lexception14:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end14-.Lcst_begin14
.Lcst_begin14:
	.uleb128 .Lfunc_begin14-.Lfunc_begin14  # >> Call Site 1 <<
	.uleb128 .Ltmp472-.Lfunc_begin14        #   Call between .Lfunc_begin14 and .Ltmp472
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp472-.Lfunc_begin14        # >> Call Site 2 <<
	.uleb128 .Ltmp485-.Ltmp472              #   Call between .Ltmp472 and .Ltmp485
	.uleb128 .Ltmp486-.Lfunc_begin14        #     jumps to .Ltmp486
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp485-.Lfunc_begin14        # >> Call Site 3 <<
	.uleb128 .Ltmp487-.Ltmp485              #   Call between .Ltmp485 and .Ltmp487
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp487-.Lfunc_begin14        # >> Call Site 4 <<
	.uleb128 .Ltmp488-.Ltmp487              #   Call between .Ltmp487 and .Ltmp488
	.uleb128 .Ltmp489-.Lfunc_begin14        #     jumps to .Ltmp489
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp490-.Lfunc_begin14        # >> Call Site 5 <<
	.uleb128 .Ltmp493-.Ltmp490              #   Call between .Ltmp490 and .Ltmp493
	.uleb128 .Ltmp494-.Lfunc_begin14        #     jumps to .Ltmp494
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp493-.Lfunc_begin14        # >> Call Site 6 <<
	.uleb128 .Lfunc_end16-.Ltmp493          #   Call between .Ltmp493 and .Lfunc_end16
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end14:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t,"axG",@progbits,_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t,comdat
	.weak	_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t # -- Begin function _ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t
	.p2align	4
	.type	_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t,@function
_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t: # @_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t
.Lfunc_begin15:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception15
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 448
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	movq	%rdi, %rbx
	movq	(%rsi), %rdi
	callq	hipStreamSynchronize
	testl	%eax, %eax
	jne	.LBB17_1
# %bb.18:                               # %if.end
	callq	_ZNSt6chrono3_V212system_clock3nowEv
	movq	%rax, 8(%rbx)
	addq	$416, %rsp                      # imm = 0x1A0
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB17_1:                               # %if.then
	.cfi_def_cfa_offset 448
	movl	%eax, %ebx
	leaq	40(%rsp), %r14
	movq	%r14, %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev
.Ltmp495:
	movl	$.L.str.37, %esi
	movl	$21, %edx
	movq	%r14, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp496:
# %bb.2:                                # %invoke.cont
.Ltmp497:
	leaq	40(%rsp), %rdi
	movl	$.L.str.44, %esi
	movl	$64, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp498:
# %bb.3:                                # %invoke.cont3
.Ltmp499:
	leaq	40(%rsp), %rdi
	movl	$.L.str.39, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp500:
# %bb.4:                                # %invoke.cont5
.Ltmp501:
	leaq	40(%rsp), %rdi
	movl	$62, %esi
	callq	_ZNSolsEi
.Ltmp502:
# %bb.5:                                # %invoke.cont7
.Ltmp503:
	movq	%rax, %r14
	movl	$.L.str.40, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp504:
# %bb.6:                                # %invoke.cont9
.Ltmp505:
	movl	%ebx, %edi
	callq	hipGetErrorString
.Ltmp506:
# %bb.7:                                # %invoke.cont11
.Ltmp507:
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp508:
# %bb.8:                                # %invoke.cont13
	movl	$16, %edi
	callq	__cxa_allocate_exception
	movq	%rax, %rbx
.Ltmp510:
	leaq	8(%rsp), %rdi
	leaq	40(%rsp), %rsi
	callq	_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv
.Ltmp511:
# %bb.9:                                # %invoke.cont16
	movb	$1, %bpl
.Ltmp513:
	leaq	8(%rsp), %rsi
	movq	%rbx, %rdi
	callq	_ZNSt13runtime_errorC1ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
.Ltmp514:
# %bb.10:                               # %invoke.cont18
	xorl	%ebp, %ebp
.Ltmp515:
	movl	$_ZTISt13runtime_error, %esi
	movl	$_ZNSt13runtime_errorD1Ev, %edx
	movq	%rbx, %rdi
	callq	__cxa_throw
.Ltmp516:
# %bb.19:                               # %unreachable
.LBB17_13:                              # %lpad17
.Ltmp517:
	movq	%rax, %r14
	movq	8(%rsp), %rdi
	leaq	24(%rsp), %rax
	cmpq	%rax, %rdi
	jne	.LBB17_14
# %bb.15:                               # %ehcleanup
	testb	%bpl, %bpl
	jne	.LBB17_16
.LBB17_17:                              # %ehcleanup20
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB17_14:                              # %ehcleanup
	movq	24(%rsp), %rsi
	incq	%rsi
	callq	_ZdlPvm
	testb	%bpl, %bpl
	je	.LBB17_17
	jmp	.LBB17_16
.LBB17_12:                              # %ehcleanup.thread
.Ltmp512:
	movq	%rax, %r14
.LBB17_16:                              # %cleanup.action
	movq	%rbx, %rdi
	callq	__cxa_free_exception
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.LBB17_11:                              # %lpad
.Ltmp509:
	movq	%rax, %r14
	leaq	40(%rsp), %rdi
	callq	_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev
	movq	%r14, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end17:
	.size	_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t, .Lfunc_end17-_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t
	.cfi_endproc
	.section	.gcc_except_table._ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t,"aG",@progbits,_ZN7ck_tile9cpu_timer4stopERKP12ihipStream_t,comdat
	.p2align	2, 0x0
GCC_except_table17:
.Lexception15:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end15-.Lcst_begin15
.Lcst_begin15:
	.uleb128 .Lfunc_begin15-.Lfunc_begin15  # >> Call Site 1 <<
	.uleb128 .Ltmp495-.Lfunc_begin15        #   Call between .Lfunc_begin15 and .Ltmp495
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp495-.Lfunc_begin15        # >> Call Site 2 <<
	.uleb128 .Ltmp508-.Ltmp495              #   Call between .Ltmp495 and .Ltmp508
	.uleb128 .Ltmp509-.Lfunc_begin15        #     jumps to .Ltmp509
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp508-.Lfunc_begin15        # >> Call Site 3 <<
	.uleb128 .Ltmp510-.Ltmp508              #   Call between .Ltmp508 and .Ltmp510
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp510-.Lfunc_begin15        # >> Call Site 4 <<
	.uleb128 .Ltmp511-.Ltmp510              #   Call between .Ltmp510 and .Ltmp511
	.uleb128 .Ltmp512-.Lfunc_begin15        #     jumps to .Ltmp512
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp513-.Lfunc_begin15        # >> Call Site 5 <<
	.uleb128 .Ltmp516-.Ltmp513              #   Call between .Ltmp513 and .Ltmp516
	.uleb128 .Ltmp517-.Lfunc_begin15        #     jumps to .Ltmp517
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp516-.Lfunc_begin15        # >> Call Site 6 <<
	.uleb128 .Lfunc_end17-.Ltmp516          #   Call between .Ltmp516 and .Lfunc_end17
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end15:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4                               # -- Begin function _GLOBAL__sub_I_04_row_major_xor.cpp
	.type	_GLOBAL__sub_I_04_row_major_xor.cpp,@function
_GLOBAL__sub_I_04_row_major_xor.cpp:    # @_GLOBAL__sub_I_04_row_major_xor.cpp
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
.Lfunc_end18:
	.size	_GLOBAL__sub_I_04_row_major_xor.cpp, .Lfunc_end18-_GLOBAL__sub_I_04_row_major_xor.cpp
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_53b1c7993ec8f2c7(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB19_2
# %bb.1:                                # %if
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_53b1c7993ec8f2c7(%rip)
.LBB19_2:                               # %exit
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end19:
	.size	__hip_module_ctor, .Lfunc_end19-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:                                # %entry
	movq	__hip_gpubin_handle_53b1c7993ec8f2c7(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB20_2
# %bb.1:                                # %if
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_53b1c7993ec8f2c7(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB20_2:                               # %exit
	retq
.Lfunc_end20:
	.size	__hip_module_dtor, .Lfunc_end20-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"\n\342\225\224\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\227\n"
	.size	.L.str, 162

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"\342\225\221 Production Transpose with CK Tile API            \342\225\221\n"
	.size	.L.str.1, 58

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"\342\225\221 Single-pass transpose (Plain vs XOR)             \342\225\221\n"
	.size	.L.str.2, 58

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"\342\225\232\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\235\n"
	.size	.L.str.3, 161

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"Test 2: XOR LDS"
	.size	.L.str.4, 16

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"\342\225\221 Summary                                           \342\225\221\n"
	.size	.L.str.5, 59

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"\342\225\232\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\220\342\225\235\n\n"
	.size	.L.str.6, 162

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"XOR LDS:   "
	.size	.L.str.7, 12

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"\342\234\223 PASSED"
	.size	.L.str.8, 11

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"\342\234\227 FAILED"
	.size	.L.str.9, 11

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"\n\n"
	.size	.L.str.10, 3

	.type	.L.str.11,@object               # @.str.11
.L.str.11:
	.asciz	"This is a production-ready transpose implementation.\n"
	.size	.L.str.11, 54

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"- Single-pass (no iteration amplification)\n"
	.size	.L.str.12, 44

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"- CK Tile API (tensor_view, tile_window, load_tile, store_tile)\n"
	.size	.L.str.13, 65

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"- XOR swizzling for bank conflict reduction\n\n"
	.size	.L.str.14, 46

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"\n========================================\n"
	.size	.L.str.16, 43

	.type	.L.str.17,@object               # @.str.17
.L.str.17:
	.asciz	"\n"
	.size	.L.str.17, 2

	.type	.L.str.18,@object               # @.str.18
.L.str.18:
	.asciz	"========================================\n\n"
	.size	.L.str.18, 43

	.type	.L.str.19,@object               # @.str.19
.L.str.19:
	.asciz	"Configuration:\n"
	.size	.L.str.19, 16

	.type	.L.str.20,@object               # @.str.20
.L.str.20:
	.asciz	"  Input:  ["
	.size	.L.str.20, 12

	.type	.L.str.21,@object               # @.str.21
.L.str.21:
	.asciz	", "
	.size	.L.str.21, 3

	.type	.L.str.22,@object               # @.str.22
.L.str.22:
	.asciz	"] (row-major)\n"
	.size	.L.str.22, 15

	.type	.L.str.23,@object               # @.str.23
.L.str.23:
	.asciz	"  Output: ["
	.size	.L.str.23, 12

	.type	.L.str.24,@object               # @.str.24
.L.str.24:
	.asciz	"] (transposed)\n"
	.size	.L.str.24, 16

	.type	.L.str.25,@object               # @.str.25
.L.str.25:
	.asciz	"  XOR: "
	.size	.L.str.25, 8

	.type	.L.str.26,@object               # @.str.26
.L.str.26:
	.asciz	"ENABLED"
	.size	.L.str.26, 8

	.type	.L.str.27,@object               # @.str.27
.L.str.27:
	.asciz	"  Mode: Single-pass production transpose\n\n"
	.size	.L.str.27, 43

	.type	.L.str.28,@object               # @.str.28
.L.str.28:
	.asciz	"Error at ["
	.size	.L.str.28, 11

	.type	.L.str.29,@object               # @.str.29
.L.str.29:
	.asciz	"]["
	.size	.L.str.29, 3

	.type	.L.str.30,@object               # @.str.30
.L.str.30:
	.asciz	"]: "
	.size	.L.str.30, 4

	.type	.L.str.31,@object               # @.str.31
.L.str.31:
	.asciz	"expected "
	.size	.L.str.31, 10

	.type	.L.str.32,@object               # @.str.32
.L.str.32:
	.asciz	", got "
	.size	.L.str.32, 7

	.type	.L.str.33,@object               # @.str.33
.L.str.33:
	.asciz	"Result: "
	.size	.L.str.33, 9

	.type	.L.str.34,@object               # @.str.34
.L.str.34:
	.asciz	" (verified full ["
	.size	.L.str.34, 18

	.type	.L.str.35,@object               # @.str.35
.L.str.35:
	.asciz	"])\n"
	.size	.L.str.35, 4

	.type	.L.str.37,@object               # @.str.37
.L.str.37:
	.asciz	"HIP Function Failed ("
	.size	.L.str.37, 22

	.type	.L.str.38,@object               # @.str.38
.L.str.38:
	.asciz	"/data0/aghamari/composable_kernel/include/ck_tile/host/device_memory.hpp"
	.size	.L.str.38, 73

	.type	.L.str.39,@object               # @.str.39
.L.str.39:
	.asciz	","
	.size	.L.str.39, 2

	.type	.L.str.40,@object               # @.str.40
.L.str.40:
	.asciz	") "
	.size	.L.str.40, 3

	.type	.L.str.43,@object               # @.str.43
.L.str.43:
	.asciz	"/data0/aghamari/composable_kernel/include/ck_tile/host/kernel_launch.hpp"
	.size	.L.str.43, 73

	.type	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,@object # @_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.section	.rodata._ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,"aG",@progbits,_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_,comdat
	.weak	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.p2align	3, 0x0
_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_:
	.quad	_ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.size	_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_, 8

	.type	.L.str.44,@object               # @.str.44
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.44:
	.asciz	"/data0/aghamari/composable_kernel/include/ck_tile/host/timer.hpp"
	.size	.L.str.44, 65

	.type	.L.str.45,@object               # @.str.45
.L.str.45:
	.asciz	"HIP runtime error: "
	.size	.L.str.45, 20

	.type	.L.str.46,@object               # @.str.46
.L.str.46:
	.asciz	". "
	.size	.L.str.46, 3

	.type	.L.str.47,@object               # @.str.47
.L.str.47:
	.asciz	"/data0/aghamari/composable_kernel/include/ck_tile/host/hip_check_error.hpp"
	.size	.L.str.47, 75

	.type	.L.str.48,@object               # @.str.48
.L.str.48:
	.asciz	": "
	.size	.L.str.48, 3

	.type	.L.str.49,@object               # @.str.49
.L.str.49:
	.asciz	"in function: "
	.size	.L.str.49, 14

	.type	.L__func__._ZN7ck_tile15hip_check_errorE10hipError_t,@object # @__func__._ZN7ck_tile15hip_check_errorE10hipError_t
.L__func__._ZN7ck_tile15hip_check_errorE10hipError_t:
	.asciz	"hip_check_error"
	.size	.L__func__._ZN7ck_tile15hip_check_errorE10hipError_t, 16

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_"
	.size	.L__unnamed_1, 89

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_53b1c7993ec8f2c7
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.hidden	__hip_gpubin_handle_53b1c7993ec8f2c7
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_04_row_major_xor.cpp
	.quad	__hip_module_ctor
	.type	__hip_cuid_53b1c7993ec8f2c7,@object # @__hip_cuid_53b1c7993ec8f2c7
	.bss
	.globl	__hip_cuid_53b1c7993ec8f2c7
__hip_cuid_53b1c7993ec8f2c7:
	.byte	0                               # 0x0
	.size	__hip_cuid_53b1c7993ec8f2c7, 1

	.ident	"AMD clang version 21.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25401 965357120e93d691c2c2f6b221deb863caf44a62)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZN7ck_tile21__device_stub__kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.addrsig_sym _GLOBAL__sub_I_04_row_major_xor.cpp
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
	.addrsig_sym _ZTISt13runtime_error
	.addrsig_sym _ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_
	.addrsig_sym _ZSt4cerr
	.addrsig_sym __hip_fatbin_53b1c7993ec8f2c7
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_53b1c7993ec8f2c7
