	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[66:67], s[2:3]
	s_mov_b64 s[64:65], s[0:1]
	s_add_u32 s64, s64, s7
	s_load_dwordx2 s[12:13], s[4:5], 0x0
	s_load_dwordx2 s[16:17], s[4:5], 0x8
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_load_dwordx2 s[2:3], s[4:5], 0x24
	s_load_dword s44, s[4:5], 0x48
	s_load_dword s10, s[4:5], 0x50
	s_load_dword s11, s[4:5], 0x58
	s_load_dwordx2 s[42:43], s[4:5], 0x6c
	s_load_dword s33, s[4:5], 0x84
	s_load_dwordx4 s[20:23], s[4:5], 0x98
	s_load_dwordx4 s[24:27], s[4:5], 0xac
	s_load_dwordx2 s[28:29], s[4:5], 0xbc
	s_load_dwordx2 s[30:31], s[4:5], 0xd4
	s_load_dwordx2 s[34:35], s[4:5], 0xe4
	s_load_dwordx2 s[36:37], s[4:5], 0x114
	s_load_dwordx2 s[38:39], s[4:5], 0x120
	s_load_dwordx2 s[40:41], s[4:5], 0x12c
	s_load_dwordx2 s[0:1], s[4:5], 0x13c
	s_load_dwordx2 s[14:15], s[4:5], 0x148
	s_load_dwordx2 s[18:19], s[4:5], 0x154
	s_load_dword s45, s[4:5], 0x16c
	s_load_dword s60, s[4:5], 0x180
	s_load_dword s7, s[4:5], 0x18c
	s_waitcnt lgkmcnt(0)
	s_load_dword s23, s[4:5], 0x1b0
	s_load_dword s46, s[4:5], 0x1c4
	s_load_dword s47, s[4:5], 0x1d4
	s_load_dwordx4 s[48:51], s[4:5], 0x1e0
	s_load_dwordx4 s[52:55], s[4:5], 0x1f4
	s_load_dwordx4 s[56:59], s[4:5], 0x208
	s_addc_u32 s65, s65, 0
	v_lshrrev_b32_e32 v2, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s4, s55, s6
	s_add_i32 s4, s6, s4
	s_lshr_b32 s4, s4, s59
	s_mul_i32 s5, s4, s51
	s_sub_i32 s5, s6, s5
	s_mul_hi_u32 s6, s4, s54
	s_add_i32 s6, s4, s6
	s_lshr_b32 s6, s6, s58
	s_mul_i32 s50, s6, s50
	s_sub_i32 s4, s4, s50
	s_mul_hi_u32 s50, s6, s53
	s_add_i32 s50, s6, s50
	s_lshr_b32 s50, s50, s57
	s_mul_i32 s49, s50, s49
	s_sub_i32 s6, s6, s49
	s_mul_hi_u32 s49, s50, s52
	s_add_i32 s49, s50, s49
	s_lshr_b32 s49, s49, s56
	s_mul_i32 s48, s49, s48
	s_sub_i32 s48, s50, s48
	s_mul_i32 s6, s6, s47
	s_add_i32 s6, s5, s6
	s_mul_i32 s5, s48, s46
	s_add_i32 s5, s5, s4
	v_mad_i32_i24 v25, v33, -4, v2
	s_mul_i32 s4, s49, s43
	v_add_u32_e32 v1, s4, v25
	v_mul_hi_u32 v3, v1, s10
	v_lshrrev_b32_e32 v32, 4, v0
	s_lshl_b32 s6, s6, 7
	v_lshrrev_b32_e32 v38, 6, v0
	v_add_u32_e32 v3, v1, v3
	v_lshrrev_b32_e32 v4, s11, v3
	v_mul_lo_u32 v3, v4, s44
	s_movk_i32 s43, 0xffe0
	v_mad_i32_i24 v6, v38, -4, v32
	s_mul_i32 s49, s49, s45
	v_sub_u32_e32 v5, v1, v3
	v_mad_i32_i24 v3, v32, -16, v0
	v_lshl_add_u32 v9, v3, 3, s6
	v_mad_i32_i24 v30, v2, s43, v0
	v_mul_hi_u32 v10, v9, s15
	v_add_u32_e32 v2, s49, v6
	v_mul_hi_u32 v12, v2, s39
	v_lshlrev_b32_e32 v31, 2, v33
	v_add_u32_e32 v10, v9, v10
	v_lshrrev_b32_e32 v10, s19, v10
	v_add_u32_e32 v12, v2, v12
	v_mul_hi_u32 v11, v10, s14
	v_lshrrev_b32_e32 v12, s41, v12
	v_mul_hi_u32 v14, v12, s38
	v_mul_lo_u32 v16, v12, s37
	v_add_u32_e32 v11, v10, v11
	v_lshrrev_b32_e32 v11, s18, v11
	v_add_u32_e32 v14, v12, v14
	v_mul_lo_u32 v15, v11, s0
	v_lshrrev_b32_e32 v39, s40, v14
	v_mul_lo_u32 v14, v39, s36
	v_sub_u32_e32 v41, v2, v16
	v_sub_u32_e32 v15, v10, v15
	v_mul_lo_u32 v11, v11, s30
	v_sub_u32_e32 v42, v12, v14
	v_mul_lo_u32 v12, v15, s34
	v_mul_lo_u32 v14, v41, s35
	v_mul_lo_u32 v15, v42, s31
	v_lshl_or_b32 v8, v4, 3, v31
	v_mul_lo_u32 v8, v8, s2
	v_mul_lo_u32 v13, v5, s3
	v_add_u32_e32 v50, v14, v12
	v_lshlrev_b32_e32 v40, 1, v38
	v_mul_lo_u32 v10, v10, s1
	v_add_u32_e32 v51, v15, v11
	v_subrev_u32_e32 v11, s28, v50
	s_lshl_b32 s4, s5, 8
	v_lshl_or_b32 v16, v39, 3, v40
	v_subrev_u32_e32 v12, s25, v51
	v_mul_lo_u32 v11, v11, s22
	v_lshl_add_u32 v7, v30, 3, s4
	v_mul_lo_u32 v14, v16, s20
	v_mul_lo_u32 v12, v12, s21
	s_sub_i32 s27, s27, s29
	v_add3_u32 v7, v7, v8, v13
	v_cmp_le_i32_e32 vcc, s28, v50
	v_cmp_gt_i32_e64 s[0:1], s27, v50
	s_sub_i32 s24, s24, s26
	v_sub_u32_e32 v8, v9, v10
	v_lshlrev_b32_e32 v9, 1, v7
	v_add_u32_e32 v7, s2, v7
	s_and_b64 s[46:47], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v51
	v_cmp_gt_i32_e64 s[0:1], s24, v51
	v_add_u32_e32 v8, v8, v11
	v_lshlrev_b32_e32 v23, 1, v7
	v_add_u32_e32 v7, s2, v7
	s_and_b64 s[0:1], vcc, s[0:1]
	s_brev_b32 s26, -2
	v_add3_u32 v8, v8, v14, v12
	v_lshlrev_b32_e32 v24, 1, v7
	v_add_u32_e32 v44, s2, v7
	v_mov_b32_e32 v7, s26
	s_and_b64 s[0:1], s[46:47], s[0:1]
	s_lshl_b32 s14, s33, 1
	s_mov_b32 s15, 0x20000
	v_cndmask_b32_e64 v7, v7, 0, s[0:1]
	v_add_u32_e32 v49, s20, v8
	v_lshlrev_b32_e32 v43, 1, v44
	s_lshl_b32 s18, s60, 1
	s_mov_b32 s19, s15
	v_lshl_add_u32 v45, v8, 1, v7
	v_lshl_add_u32 v46, v49, 1, v7
	buffer_load_dwordx4 v[7:10], v9, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[11:14], v23, s[12:15], 0 offen
	buffer_load_dwordx4 v[15:18], v24, s[12:15], 0 offen
	buffer_load_dwordx4 v[19:22], v43, s[12:15], 0 offen
	buffer_load_dwordx4 v[26:29], v45, s[16:19], 0 offen
	buffer_load_dwordx4 v[34:37], v46, s[16:19], 0 offen
	s_movk_i32 s29, 0x880
	s_movk_i32 s0, 0x44
	s_movk_i32 s1, 0x440
	v_mul_lo_u32 v6, v6, s1
	v_mul_lo_u32 v3, v3, s0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a16, 0
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v23, v7, v11 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v7, v11, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v24, v15, v19 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v15, v19, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v15, v8, v12 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v8, v12, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v16, v20 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v16, v16, v20, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v19, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v17, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v17, v17, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v20, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	v_mul_lo_u32 v14, v25, s29
	v_mul_lo_u32 v21, v30, s0
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v18, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v18, v18, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	v_or_b32_e32 v14, v14, v31
	v_add_lshl_u32 v25, v14, v21, 1
	ds_write_b64 v25, v[23:24]
	ds_write2_b32 v25, v7, v11 offset0:4 offset1:5
	ds_write2_b32 v25, v15, v12 offset0:8 offset1:9
	ds_write2_b32 v25, v8, v16 offset0:12 offset1:13
	ds_write2_b32 v25, v19, v13 offset0:16 offset1:17
	ds_write2_b32 v25, v9, v17 offset0:20 offset1:21
	ds_write2_b32 v25, v20, v30 offset0:24 offset1:25
	ds_write2_b32 v25, v10, v18 offset0:28 offset1:29
	v_and_b32_e32 v15, 63, v0
	v_and_b32_e32 v16, 32, v0
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v26, v34 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v26, v34, op_sel:[1, 1] 
             
	;;#ASMEND
	v_sub_u32_e32 v15, v15, v16
	v_lshlrev_b32_e32 v34, 5, v33
	v_add_u32_e32 v17, v15, v34
	v_ashrrev_i16_e32 v18, 15, v17
	v_lshrrev_b16_e32 v18, 13, v18
	v_add_u16_e32 v18, v17, v18
	v_ashrrev_i16_e32 v19, 3, v18
	v_and_b32_e32 v18, -8, v18
	v_sub_u16_e32 v18, v17, v18
	v_and_b32_e32 v16, 2, v32
	v_bfe_i32 v19, v19, 0, 16
	v_bfe_i32 v18, v18, 0, 16
	v_mul_u32_u24_e32 v20, s29, v16
	v_mul_i32_i24_e32 v21, s0, v19
	v_lshlrev_b32_e32 v22, 3, v18
	v_add3_u32 v20, v21, v20, v22
	v_add_u32_e32 v22, 4, v1
	v_mul_hi_u32 v23, v22, s10
	v_add_u32_e32 v30, 4, v2
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v27, v35 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v27, v35, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v23, v22, v23
	v_lshrrev_b32_e32 v47, s11, v23
	v_mul_lo_u32 v23, v47, s44
	v_mul_hi_u32 v27, v30, s39
	v_sub_u32_e32 v4, v47, v4
	v_lshl_add_u32 v4, v4, 3, -3
	v_sub_u32_e32 v46, v22, v23
	v_add_u32_e32 v22, v30, v27
	v_lshrrev_b32_e32 v22, s41, v22
	v_mul_hi_u32 v23, v22, s38
	v_sub_u32_e32 v5, v46, v5
	v_mul_lo_u32 v27, v22, s37
	v_mad_i32_i24 v21, v33, -2, v38
	v_add_u32_e32 v23, v22, v23
	v_lshrrev_b32_e32 v43, s40, v23
	v_mul_lo_u32 v4, v4, s2
	v_mul_lo_u32 v5, v5, s3
	v_mul_lo_u32 v23, v43, s36
	v_lshl_add_u32 v35, v21, 5, v15
	v_ashrrev_i32_e32 v15, 31, v35
	v_lshrrev_b32_e32 v15, 29, v15
	v_sub_u32_e32 v45, v30, v27
	v_add_u32_e32 v15, v35, v15
	v_add3_u32 v48, v5, v4, v44
	v_sub_u32_e32 v44, v22, v23
	v_sub_u32_e32 v4, v45, v41
	v_ashrrev_i32_e32 v21, 3, v15
	v_mul_lo_u32 v4, v4, s35
	v_sub_u32_e32 v22, v44, v42
	v_mul_lo_u32 v24, v21, s0
	v_mul_lo_u32 v22, v22, s31
	v_and_b32_e32 v15, -8, v15
	v_sub_u32_e32 v5, v43, v39
	v_sub_u32_e32 v15, v35, v15
	v_lshl_add_u32 v5, v5, 3, -1
	v_mul_u32_u24_e32 v16, s1, v16
	v_lshlrev_b32_e32 v26, 3, v15
	v_mul_lo_u32 v5, v5, s20
	v_mul_lo_u32 v23, v4, s22
	v_add3_u32 v16, v24, v16, v26
	v_mul_lo_u32 v24, v22, s21
	s_movk_i32 s1, 0x4400
	v_add_u32_e32 v5, v5, v23
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v28, v36 
             
	;;#ASMEND
	v_add3_u32 v49, v5, v24, v49
	v_or_b32_e32 v5, v6, v40
	v_add_lshl_u32 v3, v5, v3, 1
	v_add_u32_e32 v32, s1, v3
	v_add_u32_e32 v3, 64, v35
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v28, v36, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v36, v4, v50
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshrrev_b32_e32 v4, 29, v4
	v_add_u32_e32 v4, v3, v4
	v_ashrrev_i32_e32 v5, 3, v4
	v_sub_u32_e32 v5, v5, v21
	v_mul_lo_u32 v5, v5, s0
	v_and_b32_e32 v4, 0xffffff8, v4
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v3, v3, v15
	v_lshl_add_u32 v31, v16, 1, s1
	v_lshl_add_u32 v3, v3, 3, v5
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v29, v37 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v29, v37, op_sel:[1, 1] 
             
	;;#ASMEND
	v_lshl_add_u32 v29, v3, 1, v31
	v_add_u32_e32 v3, 64, v17
	v_add_u32_e32 v5, 0x80, v17
	v_add_u32_e32 v6, 0xc0, v17
	v_lshrrev_b32_e32 v3, 3, v3
	v_lshrrev_b32_e32 v5, 3, v5
	v_lshrrev_b32_e32 v6, 3, v6
	v_sub_u32_e32 v3, v3, v19
	v_sub_u32_e32 v5, v5, v19
	v_sub_u32_e32 v6, v6, v19
	v_mul_lo_u32 v3, v3, s0
	v_mul_lo_u32 v5, v5, s0
	v_mul_lo_u32 v6, v6, s0
	v_and_b32_e32 v4, 7, v17
	v_sub_u32_e32 v4, v4, v18
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a48, 0
	v_lshl_add_u32 v4, v4, 3, v20
	v_add_u32_e32 v39, 8, v2
	v_add_u32_e32 v40, 8, v1
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	s_mov_b32 s43, 0
	s_mov_b32 s4, s39
	v_add_u32_e32 v37, v22, v51
	v_lshlrev_b32_e32 v38, 1, v20
	v_add_lshl_u32 v28, v4, v3, 1
	v_add_lshl_u32 v27, v4, v5, 1
	v_add_lshl_u32 v26, v4, v6, 1
	s_add_i32 s30, s42, -4
	s_sub_i32 s33, 0, s37
	s_sub_i32 s34, 0, s44
	s_movk_i32 s39, 0x1100
	v_mov_b32_e32 v41, v40
	v_mov_b32_e32 v42, v39
	ds_write2_b32 v32, v7, v8 offset1:4
	ds_write2_b32 v32, v9, v10 offset0:8 offset1:12
	ds_write2_b32 v32, v11, v12 offset0:16 offset1:20
	ds_write2_b32 v32, v13, v14 offset0:24 offset1:28

	;s_lshl_b32 s42, s2, 1

BB0_1:                                  ; %_ZZN2ck22move_tensor_coordinateINS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS2_IJiiiEEELb0EEENS3_INS2_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESB_NS_23Merge_v2_magic_divisionINS2_IJiiEEEEESB_NSA_IS7_EENS3_ISD_Lb0EEESB_SF_EEENS2_IJNS_8SequenceIJLi0EEEENSI_IJLi1EEEENSI_IJLi2EEEENSI_IJLi3EEEENSI_IJLi4ELi6EEEENSI_IJLi7EEEENSI_IJLi5EEEENSI_IJLi8EEEENSI_IJLi9EEEENSI_IJLi10EEEEEEENS2_IJNSI_IJLi1ELi2ELi3EEEENSI_IJLi4ELi5EEEENSI_IJLi6EEEESO_SQ_SR_SS_NSI_IJLi11ELi12EEEENSI_IJLi13EEEENSI_IJLi14EEEEEEENSI_IJLi11ELi12ELi13ELi14EEEEiEENS_16TensorCoordinateILi15EKS11_EENS_20TensorCoordinateStepILi10ELi4ENSI_IJLi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0EEEEEEEEvRKT_RT0_RKT1_ENKUlS19_E_clINS6_IiLi9EEEEEDaS19_.exit.i.i.i.i.i315.i
                                        ; =>This Inner Loop Header: Depth=1
	v_lshlrev_b32_e32 v13, 1, v48
	buffer_load_dwordx4 v[1:4], v13, s[12:15], 0 offen
	s_lshl_b32 s42, s2, 1
	buffer_load_dwordx4 v[5:8], v13, s[12:15], s42 offen
	s_lshl_b32 s42, s2, 2
	buffer_load_dwordx4 v[9:12], v13, s[12:15], s42 offen
	s_mul_i32 s42, s2, 6
	buffer_load_dwordx4 v[21:24], v13, s[12:15], s42 offen
	s_mul_i32 s42, s2, 3
	v_add_u32_e32 v48, s42, v48


	;v_add_u32_e32 v9, s2, v48
	;v_add_u32_e32 v13, s2, v9
	v_cmp_le_i32_e32 vcc, s28, v36
	v_cmp_gt_i32_e64 s[0:1], s27, v36
	;v_lshlrev_b32_e32 v1, 1, v48
	;v_add_u32_e32 v48, s2, v13
	s_and_b64 s[44:45], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v37
	v_cmp_gt_i32_e64 s[0:1], s24, v37
	;v_lshlrev_b32_e32 v5, 1, v9
	;v_lshlrev_b32_e32 v9, 1, v13
	;v_lshlrev_b32_e32 v13, 1, v48
	s_and_b64 s[0:1], vcc, s[0:1]
	;buffer_load_dwordx4 v[1:4], v1, s[12:15], 0 offen
	s_and_b64 s[0:1], s[0:1], s[44:45]
	;buffer_load_dwordx4 v[5:8], v5, s[12:15], 0 offen
	v_add_u32_e32 v55, s39, v38
	;buffer_load_dwordx4 v[9:12], v9, s[12:15], 0 offen
	v_add_u32_e32 v71, s29, v29
	;buffer_load_dwordx4 v[21:24], v13, s[12:15], 0 offen
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	v_mov_b32_e32 v13, s26
	ds_read2_b64 v[51:54], v38 offset1:1
	v_cndmask_b32_e64 v17, v13, 0, s[0:1]
	v_lshl_add_u32 v13, v49, 1, v17
	v_add_u32_e32 v49, s20, v49
	v_lshl_add_u32 v17, v49, 1, v17
	buffer_load_dwordx4 v[13:16], v13, s[16:19], 0 offen
	v_add_u32_e32 v63, s29, v31
	buffer_load_dwordx4 v[17:20], v17, s[16:19], 0 offen
	ds_read2_b64 v[55:58], v55 offset1:1
	ds_read2_b64 v[59:62], v31 offset1:1
	ds_read2_b64 v[67:70], v29 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[112:127], v[51:52], v[59:60], a[112:127]
	ds_read2_b64 v[71:74], v71 offset1:1
	ds_read2_b64 v[63:66], v63 offset1:1
	v_mul_hi_u32 v75, s10, v41
	v_mul_hi_u32 v50, s4, v42
	v_add_u32_e32 v30, 4, v30
	v_add_u32_e32 v42, 4, v42
	v_add_u32_e32 v41, 4, v41
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[96:111], v[51:52], v[67:68], a[96:111]
	v_mfma_f32_32x32x8f16 a[112:127], v[53:54], v[61:62], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[53:54], v[69:70], a[96:111]
	ds_read2_b64 v[51:54], v28 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[80:95], v[51:52], v[59:60], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[51:52], v[67:68], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[55:56], v[71:72], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[53:54], v[61:62], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[53:54], v[69:70], a[64:79]
	ds_read2_b64 v[51:54], v27 offset1:1
	v_mfma_f32_32x32x8f16 a[112:127], v[55:56], v[63:64], a[112:127]
	v_add_u32_e32 v55, s39, v28
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[51:52], v[59:60], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[51:52], v[67:68], a[16:31]
	v_mfma_f32_32x32x8f16 a[112:127], v[57:58], v[65:66], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[57:58], v[73:74], a[96:111]
	ds_read2_b64 v[55:58], v55 offset1:1
	v_mfma_f32_32x32x8f16 a[32:47], v[53:54], v[61:62], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[53:54], v[69:70], a[16:31]
	ds_read2_b64 v[51:54], v26 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[80:95], v[55:56], v[63:64], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[55:56], v[71:72], a[64:79]
	v_add_u32_e32 v55, s39, v27
	v_mfma_f32_32x32x8f16 a[80:95], v[57:58], v[65:66], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[57:58], v[73:74], a[64:79]
	ds_read2_b64 v[55:58], v55 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[0:15], v[51:52], v[59:60], a[0:15]
	v_mfma_f32_32x32x8f16 a[48:63], v[51:52], v[67:68], a[48:63]
	v_add3_u32 v51, v40, v75, s43
	v_lshrrev_b32_e32 v51, s11, v51
	v_mul_lo_u32 v52, s34, v51
	v_sub_u32_e32 v47, v51, v47
	v_lshl_add_u32 v47, v47, 3, -3
	v_mul_lo_u32 v47, v47, s2
	v_sub_u32_e32 v46, v52, v46
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[55:56], v[63:64], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[55:56], v[71:72], a[16:31]
	v_add_u32_e32 v55, s39, v26
	v_mfma_f32_32x32x8f16 a[0:15], v[53:54], v[61:62], a[0:15]
	v_mfma_f32_32x32x8f16 a[48:63], v[53:54], v[69:70], a[48:63]
	v_add_u32_e32 v53, s43, v40
	v_add_u32_e32 v46, v53, v46
	v_mul_lo_u32 v46, v46, s3
	v_add_u32_e32 v54, v53, v52
	v_add3_u32 v48, v47, v48, v46
	v_add3_u32 v46, v39, v50, s43
	v_lshrrev_b32_e32 v46, s41, v46
	v_mul_lo_u32 v50, s33, v46
	v_mul_lo_u32 v47, v46, s37
	v_sub_u32_e32 v45, v50, v45
	v_mul_hi_u32 v50, v46, s38
	v_mfma_f32_32x32x8f16 a[32:47], v[57:58], v[65:66], a[32:47]
	v_add3_u32 v45, v39, s43, v45
	v_mul_lo_u32 v45, v45, s35
	v_add_u32_e32 v50, v46, v50
	v_lshrrev_b32_e32 v50, s40, v50
	v_mul_lo_u32 v52, v50, s36
	v_sub_u32_e32 v43, v50, v43
	v_lshl_add_u32 v43, v43, 3, -1
	v_add_u32_e32 v36, v45, v36
	v_sub_u32_e32 v46, v46, v52
	v_sub_u32_e32 v44, v46, v44
	v_mul_lo_u32 v44, v44, s31
	v_mul_lo_u32 v45, v45, s22
	v_mul_lo_u32 v43, v43, s20
	v_sub_u32_e32 v47, v30, v47
	v_add_u32_e32 v37, v44, v37
	v_mfma_f32_32x32x8f16 a[16:31], v[57:58], v[73:74], a[16:31]
	ds_read2_b64 v[55:58], v55 offset1:1
	v_mul_lo_u32 v44, v44, s21
	v_add_u32_e32 v45, v45, v49
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_add_i32 s43, s43, 4
	v_add3_u32 v49, v45, v43, v44
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v43, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v44, v9, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v9, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v21, v3, v7 
             
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[0:15], v[55:56], v[63:64], a[0:15]
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v11, v23 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v11, v23, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v22, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v24 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v12, v24, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write_b64 v25, v[43:44]
	ds_write2_b32 v25, v1, v5 offset0:4 offset1:5
	ds_write2_b32 v25, v9, v6 offset0:8 offset1:9
	ds_write2_b32 v25, v2, v10 offset0:12 offset1:13
	ds_write2_b32 v25, v21, v7 offset0:16 offset1:17
	ds_write2_b32 v25, v3, v11 offset0:20 offset1:21
	ds_write2_b32 v25, v22, v8 offset0:24 offset1:25
	ds_write2_b32 v25, v4, v12 offset0:28 offset1:29
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v13, v17 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v13, v17, op_sel:[1, 1] 
             
	;;#ASMEND
	s_cmp_lt_i32 s43, s30
	v_mov_b32_e32 v43, v50
	v_mov_b32_e32 v44, v46
	v_mov_b32_e32 v45, v47
	v_mov_b32_e32 v46, v54
	v_mfma_f32_32x32x8f16 a[48:63], v[55:56], v[71:72], a[48:63]
	v_mov_b32_e32 v47, v51
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v14, v18 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v14, v18, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v15, v19 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v15, v19, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v16, v20 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v16, v20, op_sel:[1, 1] 
             
	;;#ASMEND
	v_mfma_f32_32x32x8f16 a[0:15], v[57:58], v[65:66], a[0:15]
	ds_write2_b32 v32, v1, v2 offset1:4
	ds_write2_b32 v32, v3, v4 offset0:8 offset1:12
	ds_write2_b32 v32, v5, v6 offset0:16 offset1:20
	ds_write2_b32 v32, v7, v8 offset0:24 offset1:28
	
	v_mfma_f32_32x32x8f16 a[48:63], v[57:58], v[73:74], a[48:63]
	s_cbranch_scc1 BB0_1
; %bb.2:                                ; %_ZZN2ck23Merge_v2_magic_divisionINS_5TupleIJNS_17integral_constantIiLi4EEENS2_IiLi2EEEiiiEEEEC1ERKS5_ENKUlT_E_clIS4_EEDaS9_.exit.i.i.i.i.i.i.i.i
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v38 offset1:1
	s_movk_i32 s0, 0x1100
	v_add_u32_e32 v5, s0, v38
	ds_read2_b64 v[5:8], v5 offset1:1
	ds_read2_b64 v[9:12], v31 offset1:1
	ds_read2_b64 v[17:20], v29 offset1:1
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[112:127], v[1:2], v[9:10], a[112:127]
	s_movk_i32 s1, 0x880
	v_add_u32_e32 v13, s1, v31
	v_add_u32_e32 v21, s1, v29
	ds_read2_b64 v[13:16], v13 offset1:1
	ds_read2_b64 v[21:24], v21 offset1:1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[96:111], v[1:2], v[17:18], a[96:111]
	v_mfma_f32_32x32x8f16 a[112:127], v[3:4], v[11:12], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[3:4], v[19:20], a[96:111]
	ds_read2_b64 v[1:4], v28 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[80:95], v[1:2], v[9:10], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[1:2], v[17:18], a[64:79]
	v_mfma_f32_32x32x8f16 a[80:95], v[3:4], v[11:12], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[3:4], v[19:20], a[64:79]
	ds_read2_b64 v[1:4], v27 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[1:2], v[9:10], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[17:18], a[16:31]
	v_mfma_f32_32x32x8f16 a[112:127], v[5:6], v[13:14], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[5:6], v[21:22], a[96:111]
	v_add_u32_e32 v5, s0, v28
	ds_read2_b64 v[28:31], v5 offset1:1
	v_mfma_f32_32x32x8f16 a[32:47], v[3:4], v[11:12], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[19:20], a[16:31]
	ds_read2_b64 v[1:4], v26 offset1:1
	v_add_u32_e32 v5, s0, v27
	ds_read2_b64 v[36:39], v5 offset1:1
	v_add_u32_e32 v5, s0, v26
	ds_read2_b64 v[40:43], v5 offset1:1
	s_movk_i32 s0, 0x80
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[17:18], a[48:63]
	v_mfma_f32_32x32x8f16 a[80:95], v[28:29], v[13:14], a[80:95]
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[19:20], a[48:63]
	v_mfma_f32_32x32x8f16 a[0:15], v[1:2], v[9:10], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[40:41], v[21:22], a[48:63]
	v_mfma_f32_32x32x8f16 a[80:95], v[30:31], v[15:16], a[80:95]
	v_mfma_f32_32x32x8f16 a[0:15], v[3:4], v[11:12], a[0:15]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v3, a80              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:68 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a81              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:72 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a82              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:76 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a83              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[48:63], v[42:43], v[23:24], a[48:63]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:80 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a84              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:84 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a85              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:88 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a86              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:92 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a87              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[64:79], v[28:29], v[21:22], a[64:79]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:96 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a88              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:100 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a89              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:104 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a90              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:108 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a91              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[32:47], v[36:37], v[13:14], a[32:47]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:112 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a92              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:116 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a93              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:120 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a94              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:124 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a95              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[16:31], v[36:37], v[21:22], a[16:31]
	v_mul_i32_i24_e32 v36, 0xffffffe0, v33
	buffer_store_dword v3, off, s[64:67], 0 offset:128 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a48              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a49              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a50              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a51              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[0:15], v[40:41], v[13:14], a[0:15]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a52              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a53              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a54              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a55              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[32:47], v[38:39], v[15:16], a[32:47]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a56              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a57              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a58              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a59              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[96:111], v[7:8], v[23:24], a[96:111]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a60              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a61              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a62              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a63              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[16:31], v[38:39], v[23:24], a[16:31]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[48:63], v[7:8], v[15:16], a[112:127]
	v_mfma_f32_32x32x8f16 a[80:95], v[30:31], v[23:24], a[64:79]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v17, a48
	v_accvgpr_read_b32 v18, a49
	v_accvgpr_read_b32 v19, a50
	v_accvgpr_read_b32 v20, a51
	v_accvgpr_read_b32 v21, a52
	v_accvgpr_read_b32 v22, a53
	v_accvgpr_read_b32 v23, a54
	v_accvgpr_read_b32 v24, a55
	v_accvgpr_read_b32 v25, a56
	v_accvgpr_read_b32 v26, a57
	v_accvgpr_read_b32 v27, a58
	v_accvgpr_read_b32 v28, a59
	v_accvgpr_read_b32 v29, a60
	v_mfma_f32_32x32x8f16 a[64:79], v[42:43], v[15:16], a[0:15]
	v_accvgpr_read_b32 v30, a61
	v_accvgpr_read_b32 v31, a62
	v_accvgpr_read_b32 v32, a63
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_4
; %bb.3:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s5, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s7
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v50, v2, v4, v1
	v_add3_u32 v49, s6, v2, v3
BB0_4:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES18_S1D_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v51, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v17
	ds_write_b16 v51, v0 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a96
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a97
	v_accvgpr_read_b32 v3, a98
	v_accvgpr_read_b32 v4, a99
	v_accvgpr_read_b32 v5, a100
	v_accvgpr_read_b32 v6, a101
	v_accvgpr_read_b32 v7, a102
	v_accvgpr_read_b32 v8, a103
	v_accvgpr_read_b32 v9, a104
	v_accvgpr_read_b32 v10, a105
	v_accvgpr_read_b32 v11, a106
	v_accvgpr_read_b32 v12, a107
	v_accvgpr_read_b32 v13, a108
	v_accvgpr_read_b32 v14, a109
	v_accvgpr_read_b32 v15, a110
	v_accvgpr_read_b32 v16, a111
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_6
; %bb.5:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v50
	ds_read2_b64 v[17:20], v0 offset1:1
	ds_read2_b64 v[21:24], v0 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[17:20], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v17, 8, v49
	v_lshlrev_b32_e32 v18, 1, v17
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[21:24], v18, s[8:11], 0 offen
	v_add_lshl_u32 v25, v17, s7, 1
	ds_read2_b64 v[17:20], v0 offset0:18 offset1:19
	ds_read2_b64 v[21:24], v0 offset0:16 offset1:17
	v_add_lshl_u32 v0, v49, s7, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[17:20], v25, s[8:11], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[21:24], v0, s[8:11], 0 offen
BB0_6:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v17, a80
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v18, a81
	v_accvgpr_read_b32 v19, a82
	v_accvgpr_read_b32 v20, a83
	v_accvgpr_read_b32 v21, a84
	v_accvgpr_read_b32 v22, a85
	v_accvgpr_read_b32 v23, a86
	v_accvgpr_read_b32 v24, a87
	v_accvgpr_read_b32 v25, a88
	v_accvgpr_read_b32 v26, a89
	v_accvgpr_read_b32 v27, a90
	v_accvgpr_read_b32 v28, a91
	v_accvgpr_read_b32 v29, a92
	v_accvgpr_read_b32 v30, a93
	v_accvgpr_read_b32 v31, a94
	v_accvgpr_read_b32 v32, a95
	v_cvt_f16_f32_e32 v3, v13
	s_mul_i32 s2, s7, 63
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_8
; %bb.7:                                ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, s7, v49
	v_lshlrev_b32_e32 v1, 1, v0
	v_add_u32_e32 v49, s2, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
BB0_8:                                  ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v4, off, s[64:67], 0 offset:68 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v0, v17
	v_cvt_f16_f32_e32 v1, v18
	v_cvt_f16_f32_e32 v2, v19
	v_cvt_f16_f32_e32 v3, v20
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:124 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:128 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v23
	v_cvt_f16_f32_e32 v2, v22
	v_cvt_f16_f32_e32 v3, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v1, v26
	v_cvt_f16_f32_e32 v2, v27
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v31
	v_cvt_f16_f32_e32 v2, v30
	v_cvt_f16_f32_e32 v3, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v4             ;  Reload Reuse
	s_nop 2
	v_accvgpr_read_b32 v48, a15
	v_accvgpr_read_b32 v47, a14
	v_accvgpr_read_b32 v46, a13
	v_accvgpr_read_b32 v45, a12
	v_accvgpr_read_b32 v44, a11
	v_accvgpr_read_b32 v43, a10
	v_accvgpr_read_b32 v42, a9
	v_accvgpr_read_b32 v41, a8
	v_accvgpr_read_b32 v40, a7
	v_accvgpr_read_b32 v39, a6
	v_accvgpr_read_b32 v38, a5
	v_accvgpr_read_b32 v37, a4
	v_accvgpr_read_b32 v36, a3
	v_accvgpr_read_b32 v35, a2
	v_accvgpr_read_b32 v34, a1
	v_accvgpr_read_b32 v33, a0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_10
; %bb.9:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i141.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v49, s7, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_10:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i234.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v33
	v_cvt_f16_f32_e32 v17, v34
	v_cvt_f16_f32_e32 v18, v35
	v_cvt_f16_f32_e32 v19, v36
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v40
	v_cvt_f16_f32_e32 v17, v39
	v_cvt_f16_f32_e32 v18, v38
	v_cvt_f16_f32_e32 v19, v37
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v41
	v_cvt_f16_f32_e32 v17, v42
	v_cvt_f16_f32_e32 v18, v43
	v_cvt_f16_f32_e32 v19, v44
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v48
	v_cvt_f16_f32_e32 v17, v47
	v_accvgpr_read_b32 v0, a32
	v_cvt_f16_f32_e32 v18, v46
	v_accvgpr_read_b32 v1, a33
	v_accvgpr_read_b32 v2, a34
	v_accvgpr_read_b32 v3, a35
	v_accvgpr_read_b32 v4, a36
	v_accvgpr_read_b32 v5, a37
	v_accvgpr_read_b32 v6, a38
	v_accvgpr_read_b32 v7, a39
	v_accvgpr_read_b32 v8, a40
	v_accvgpr_read_b32 v9, a41
	v_accvgpr_read_b32 v10, a42
	v_accvgpr_read_b32 v11, a43
	v_accvgpr_read_b32 v12, a44
	v_accvgpr_read_b32 v13, a45
	v_accvgpr_read_b32 v14, a46
	v_accvgpr_read_b32 v15, a47
	v_cvt_f16_f32_e32 v19, v45
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_12
; %bb.11:                               ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i278.i.i.i
	v_lshlrev_b32_e32 v24, 1, v50
	ds_read2_b64 v[16:19], v24 offset1:1
	ds_read2_b64 v[20:23], v24 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v49
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read2_b64 v[16:19], v24 offset0:18 offset1:19
	ds_read2_b64 v[20:23], v24 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s7, v49
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v49, s2, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
BB0_12:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I303.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v47, a31
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v46, a30
	v_accvgpr_read_b32 v45, a29
	v_accvgpr_read_b32 v44, a28
	v_accvgpr_read_b32 v43, a27
	v_accvgpr_read_b32 v42, a26
	v_accvgpr_read_b32 v41, a25
	v_accvgpr_read_b32 v40, a24
	v_accvgpr_read_b32 v39, a23
	v_accvgpr_read_b32 v38, a22
	v_accvgpr_read_b32 v37, a21
	v_accvgpr_read_b32 v36, a20
	v_accvgpr_read_b32 v35, a19
	v_accvgpr_read_b32 v34, a18
	v_accvgpr_read_b32 v33, a17
	v_accvgpr_read_b32 v32, a16
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_14
; %bb.13:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i444.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v49, s7, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_14:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i537.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v4, off, s[64:67], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v33
	v_cvt_f16_f32_e32 v2, v34
	v_cvt_f16_f32_e32 v3, v35
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v39
	v_cvt_f16_f32_e32 v1, v38
	v_cvt_f16_f32_e32 v2, v37
	v_cvt_f16_f32_e32 v3, v36
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v40
	v_cvt_f16_f32_e32 v1, v41
	v_cvt_f16_f32_e32 v2, v42
	v_cvt_f16_f32_e32 v3, v43
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v47
	v_cvt_f16_f32_e32 v1, v46
	v_cvt_f16_f32_e32 v2, v45
	v_cvt_f16_f32_e32 v3, v44
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v4             ;  Reload Reuse
	s_nop 2
	v_accvgpr_read_b32 v31, a15
	v_accvgpr_read_b32 v30, a14
	v_accvgpr_read_b32 v29, a13
	v_accvgpr_read_b32 v28, a12
	v_accvgpr_read_b32 v27, a11
	v_accvgpr_read_b32 v26, a10
	v_accvgpr_read_b32 v25, a9
	v_accvgpr_read_b32 v24, a8
	v_accvgpr_read_b32 v23, a7
	v_accvgpr_read_b32 v22, a6
	v_accvgpr_read_b32 v21, a5
	v_accvgpr_read_b32 v20, a4
	v_accvgpr_read_b32 v19, a3
	v_accvgpr_read_b32 v18, a2
	v_accvgpr_read_b32 v17, a1
	v_accvgpr_read_b32 v16, a0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_16
; %bb.15:                               ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i581.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, s7, v49
	v_lshlrev_b32_e32 v1, 1, v0
	v_add_u32_e32 v49, s2, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
BB0_16:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I606.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_accvgpr_read_b32 v0, a64
	v_cvt_f16_f32_e32 v18, v29
	v_accvgpr_read_b32 v1, a65
	v_accvgpr_read_b32 v2, a66
	v_accvgpr_read_b32 v3, a67
	v_accvgpr_read_b32 v4, a68
	v_accvgpr_read_b32 v5, a69
	v_accvgpr_read_b32 v6, a70
	v_accvgpr_read_b32 v7, a71
	v_accvgpr_read_b32 v8, a72
	v_accvgpr_read_b32 v9, a73
	v_accvgpr_read_b32 v10, a74
	v_accvgpr_read_b32 v11, a75
	v_accvgpr_read_b32 v12, a76
	v_accvgpr_read_b32 v13, a77
	v_accvgpr_read_b32 v14, a78
	v_accvgpr_read_b32 v15, a79
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_18
; %bb.17:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i747.i.i.i
	v_lshlrev_b32_e32 v24, 1, v50
	ds_read2_b64 v[16:19], v24 offset1:1
	ds_read2_b64 v[20:23], v24 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v25, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v49
	v_lshlrev_b32_e32 v17, 1, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[8:11], 0 offen
	v_add_lshl_u32 v25, v16, s7, 1
	ds_read2_b64 v[16:19], v24 offset0:18 offset1:19
	ds_read2_b64 v[20:23], v24 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v16, v49, s7, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v16, s[8:11], 0 offen
BB0_18:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i840.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_cvt_f16_f32_e32 v2, v13
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB0_20
; %bb.19:
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[8:11], 0 offen
	v_add_lshl_u32 v9, v0, s7, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[8:11], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v49, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[8:11], 0 offen
BB0_20:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 26112
		.amdhsa_private_segment_fixed_size 132
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 68
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
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
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end0-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 9920
; NumSgprs: 70
; NumVgprs: 76
; NumAgprs: 128
; TotalNumVgprs: 128
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 26112 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 70
; NumVGPRsForWavesPerEU: 128
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[66:67], s[2:3]
	s_mov_b64 s[64:65], s[0:1]
	s_add_u32 s64, s64, s7
	s_load_dwordx2 s[12:13], s[4:5], 0x0
	s_load_dwordx2 s[16:17], s[4:5], 0x8
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_load_dwordx2 s[2:3], s[4:5], 0x24
	s_load_dword s44, s[4:5], 0x48
	s_load_dword s10, s[4:5], 0x50
	s_load_dword s11, s[4:5], 0x58
	s_load_dwordx2 s[42:43], s[4:5], 0x6c
	s_load_dword s33, s[4:5], 0x84
	s_load_dwordx4 s[20:23], s[4:5], 0x98
	s_load_dwordx4 s[24:27], s[4:5], 0xac
	s_load_dwordx2 s[28:29], s[4:5], 0xbc
	s_load_dwordx2 s[30:31], s[4:5], 0xd4
	s_load_dwordx2 s[34:35], s[4:5], 0xe4
	s_load_dwordx2 s[36:37], s[4:5], 0x114
	s_load_dwordx2 s[38:39], s[4:5], 0x120
	s_load_dwordx2 s[40:41], s[4:5], 0x12c
	s_load_dwordx2 s[0:1], s[4:5], 0x13c
	s_load_dwordx2 s[14:15], s[4:5], 0x148
	s_load_dwordx2 s[18:19], s[4:5], 0x154
	s_load_dword s45, s[4:5], 0x16c
	s_load_dword s60, s[4:5], 0x180
	s_load_dword s7, s[4:5], 0x18c
	s_waitcnt lgkmcnt(0)
	s_load_dword s23, s[4:5], 0x1b0
	s_load_dword s46, s[4:5], 0x1c4
	s_load_dword s47, s[4:5], 0x1d4
	s_load_dwordx4 s[48:51], s[4:5], 0x1e0
	s_load_dwordx4 s[52:55], s[4:5], 0x1f4
	s_load_dwordx4 s[56:59], s[4:5], 0x208
	s_addc_u32 s65, s65, 0
	v_lshrrev_b32_e32 v2, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s4, s55, s6
	s_add_i32 s4, s6, s4
	s_lshr_b32 s4, s4, s59
	s_mul_i32 s5, s4, s51
	s_sub_i32 s5, s6, s5
	s_mul_hi_u32 s6, s4, s54
	s_add_i32 s6, s4, s6
	s_lshr_b32 s6, s6, s58
	s_mul_i32 s50, s6, s50
	s_sub_i32 s4, s4, s50
	s_mul_hi_u32 s50, s6, s53
	s_add_i32 s50, s6, s50
	s_lshr_b32 s50, s50, s57
	s_mul_i32 s49, s50, s49
	s_sub_i32 s6, s6, s49
	s_mul_hi_u32 s49, s50, s52
	s_add_i32 s49, s50, s49
	s_lshr_b32 s49, s49, s56
	s_mul_i32 s48, s49, s48
	s_sub_i32 s48, s50, s48
	s_mul_i32 s6, s6, s47
	s_add_i32 s6, s5, s6
	s_mul_i32 s5, s48, s46
	s_add_i32 s5, s5, s4
	v_mad_i32_i24 v25, v33, -4, v2
	s_mul_i32 s4, s49, s43
	v_add_u32_e32 v1, s4, v25
	v_mul_hi_u32 v3, v1, s10
	v_lshrrev_b32_e32 v32, 4, v0
	s_lshl_b32 s6, s6, 7
	v_lshrrev_b32_e32 v38, 6, v0
	v_add_u32_e32 v3, v1, v3
	v_lshrrev_b32_e32 v4, s11, v3
	v_mul_lo_u32 v3, v4, s44
	s_movk_i32 s43, 0xffe0
	v_mad_i32_i24 v6, v38, -4, v32
	s_mul_i32 s49, s49, s45
	v_sub_u32_e32 v5, v1, v3
	v_mad_i32_i24 v3, v32, -16, v0
	v_lshl_add_u32 v9, v3, 3, s6
	v_mad_i32_i24 v30, v2, s43, v0
	v_mul_hi_u32 v10, v9, s15
	v_add_u32_e32 v2, s49, v6
	v_mul_hi_u32 v12, v2, s39
	v_lshlrev_b32_e32 v31, 2, v33
	v_add_u32_e32 v10, v9, v10
	v_lshrrev_b32_e32 v10, s19, v10
	v_add_u32_e32 v12, v2, v12
	v_mul_hi_u32 v11, v10, s14
	v_lshrrev_b32_e32 v12, s41, v12
	v_mul_hi_u32 v14, v12, s38
	v_mul_lo_u32 v16, v12, s37
	v_add_u32_e32 v11, v10, v11
	v_lshrrev_b32_e32 v11, s18, v11
	v_add_u32_e32 v14, v12, v14
	v_mul_lo_u32 v15, v11, s0
	v_lshrrev_b32_e32 v39, s40, v14
	v_mul_lo_u32 v14, v39, s36
	v_sub_u32_e32 v41, v2, v16
	v_sub_u32_e32 v15, v10, v15
	v_mul_lo_u32 v11, v11, s30
	v_sub_u32_e32 v42, v12, v14
	v_mul_lo_u32 v12, v15, s34
	v_mul_lo_u32 v14, v41, s35
	v_mul_lo_u32 v15, v42, s31
	v_lshl_or_b32 v8, v4, 3, v31
	v_mul_lo_u32 v8, v8, s2
	v_mul_lo_u32 v13, v5, s3
	v_add_u32_e32 v50, v14, v12
	v_lshlrev_b32_e32 v40, 1, v38
	v_mul_lo_u32 v10, v10, s1
	v_add_u32_e32 v51, v15, v11
	v_subrev_u32_e32 v11, s28, v50
	s_lshl_b32 s4, s5, 8
	v_lshl_or_b32 v16, v39, 3, v40
	v_subrev_u32_e32 v12, s25, v51
	v_mul_lo_u32 v11, v11, s22
	v_lshl_add_u32 v7, v30, 3, s4
	v_mul_lo_u32 v14, v16, s20
	v_mul_lo_u32 v12, v12, s21
	s_sub_i32 s27, s27, s29
	v_add3_u32 v7, v7, v8, v13
	v_cmp_le_i32_e32 vcc, s28, v50
	v_cmp_gt_i32_e64 s[0:1], s27, v50
	s_sub_i32 s24, s24, s26
	v_sub_u32_e32 v8, v9, v10
	v_lshlrev_b32_e32 v9, 1, v7
	v_add_u32_e32 v7, s2, v7
	s_and_b64 s[46:47], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v51
	v_cmp_gt_i32_e64 s[0:1], s24, v51
	v_add_u32_e32 v8, v8, v11
	v_lshlrev_b32_e32 v23, 1, v7
	v_add_u32_e32 v7, s2, v7
	s_and_b64 s[0:1], vcc, s[0:1]
	s_brev_b32 s26, -2
	v_add3_u32 v8, v8, v14, v12
	v_lshlrev_b32_e32 v24, 1, v7
	v_add_u32_e32 v44, s2, v7
	v_mov_b32_e32 v7, s26
	s_and_b64 s[0:1], s[46:47], s[0:1]
	s_lshl_b32 s14, s33, 1
	s_mov_b32 s15, 0x20000
	v_cndmask_b32_e64 v7, v7, 0, s[0:1]
	v_add_u32_e32 v49, s20, v8
	v_lshlrev_b32_e32 v43, 1, v44
	s_lshl_b32 s18, s60, 1
	s_mov_b32 s19, s15
	v_lshl_add_u32 v45, v8, 1, v7
	v_lshl_add_u32 v46, v49, 1, v7
	buffer_load_dwordx4 v[7:10], v9, s[12:15], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[11:14], v23, s[12:15], 0 offen
	buffer_load_dwordx4 v[15:18], v24, s[12:15], 0 offen
	buffer_load_dwordx4 v[19:22], v43, s[12:15], 0 offen
	buffer_load_dwordx4 v[26:29], v45, s[16:19], 0 offen
	buffer_load_dwordx4 v[34:37], v46, s[16:19], 0 offen
	s_movk_i32 s29, 0x880
	s_movk_i32 s0, 0x44
	s_movk_i32 s1, 0x440
	v_mul_lo_u32 v6, v6, s1
	v_mul_lo_u32 v3, v3, s0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a16, 0
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v23, v7, v11 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v7, v11, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v24, v15, v19 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v15, v19, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v15, v8, v12 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v8, v12, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v16, v20 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v16, v16, v20, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v19, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v17, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v17, v17, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v20, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	v_mul_lo_u32 v14, v25, s29
	v_mul_lo_u32 v21, v30, s0
	;;#ASMSTART
	
             v_pack_b32_f16 v30, v18, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v18, v18, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	v_or_b32_e32 v14, v14, v31
	v_add_lshl_u32 v25, v14, v21, 1
	ds_write_b64 v25, v[23:24]
	ds_write2_b32 v25, v7, v11 offset0:4 offset1:5
	ds_write2_b32 v25, v15, v12 offset0:8 offset1:9
	ds_write2_b32 v25, v8, v16 offset0:12 offset1:13
	ds_write2_b32 v25, v19, v13 offset0:16 offset1:17
	ds_write2_b32 v25, v9, v17 offset0:20 offset1:21
	ds_write2_b32 v25, v20, v30 offset0:24 offset1:25
	ds_write2_b32 v25, v10, v18 offset0:28 offset1:29
	v_and_b32_e32 v15, 63, v0
	v_and_b32_e32 v16, 32, v0
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v26, v34 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v26, v34, op_sel:[1, 1] 
             
	;;#ASMEND
	v_sub_u32_e32 v15, v15, v16
	v_lshlrev_b32_e32 v34, 5, v33
	v_add_u32_e32 v17, v15, v34
	v_ashrrev_i16_e32 v18, 15, v17
	v_lshrrev_b16_e32 v18, 13, v18
	v_add_u16_e32 v18, v17, v18
	v_ashrrev_i16_e32 v19, 3, v18
	v_and_b32_e32 v18, -8, v18
	v_sub_u16_e32 v18, v17, v18
	v_and_b32_e32 v16, 2, v32
	v_bfe_i32 v19, v19, 0, 16
	v_bfe_i32 v18, v18, 0, 16
	v_mul_u32_u24_e32 v20, s29, v16
	v_mul_i32_i24_e32 v21, s0, v19
	v_lshlrev_b32_e32 v22, 3, v18
	v_add3_u32 v20, v21, v20, v22
	v_add_u32_e32 v22, 4, v1
	v_mul_hi_u32 v23, v22, s10
	v_add_u32_e32 v30, 4, v2
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v27, v35 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v27, v35, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v23, v22, v23
	v_lshrrev_b32_e32 v47, s11, v23
	v_mul_lo_u32 v23, v47, s44
	v_mul_hi_u32 v27, v30, s39
	v_sub_u32_e32 v4, v47, v4
	v_lshl_add_u32 v4, v4, 3, -3
	v_sub_u32_e32 v46, v22, v23
	v_add_u32_e32 v22, v30, v27
	v_lshrrev_b32_e32 v22, s41, v22
	v_mul_hi_u32 v23, v22, s38
	v_sub_u32_e32 v5, v46, v5
	v_mul_lo_u32 v27, v22, s37
	v_mad_i32_i24 v21, v33, -2, v38
	v_add_u32_e32 v23, v22, v23
	v_lshrrev_b32_e32 v43, s40, v23
	v_mul_lo_u32 v4, v4, s2
	v_mul_lo_u32 v5, v5, s3
	v_mul_lo_u32 v23, v43, s36
	v_lshl_add_u32 v35, v21, 5, v15
	v_ashrrev_i32_e32 v15, 31, v35
	v_lshrrev_b32_e32 v15, 29, v15
	v_sub_u32_e32 v45, v30, v27
	v_add_u32_e32 v15, v35, v15
	v_add3_u32 v48, v5, v4, v44
	v_sub_u32_e32 v44, v22, v23
	v_sub_u32_e32 v4, v45, v41
	v_ashrrev_i32_e32 v21, 3, v15
	v_mul_lo_u32 v4, v4, s35
	v_sub_u32_e32 v22, v44, v42
	v_mul_lo_u32 v24, v21, s0
	v_mul_lo_u32 v22, v22, s31
	v_and_b32_e32 v15, -8, v15
	v_sub_u32_e32 v5, v43, v39
	v_sub_u32_e32 v15, v35, v15
	v_lshl_add_u32 v5, v5, 3, -1
	v_mul_u32_u24_e32 v16, s1, v16
	v_lshlrev_b32_e32 v26, 3, v15
	v_mul_lo_u32 v5, v5, s20
	v_mul_lo_u32 v23, v4, s22
	v_add3_u32 v16, v24, v16, v26
	v_mul_lo_u32 v24, v22, s21
	s_movk_i32 s1, 0x4400
	v_add_u32_e32 v5, v5, v23
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v28, v36 
             
	;;#ASMEND
	v_add3_u32 v49, v5, v24, v49
	v_or_b32_e32 v5, v6, v40
	v_add_lshl_u32 v3, v5, v3, 1
	v_add_u32_e32 v32, s1, v3
	v_add_u32_e32 v3, 64, v35
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v28, v36, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v36, v4, v50
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshrrev_b32_e32 v4, 29, v4
	v_add_u32_e32 v4, v3, v4
	v_ashrrev_i32_e32 v5, 3, v4
	v_sub_u32_e32 v5, v5, v21
	v_mul_lo_u32 v5, v5, s0
	v_and_b32_e32 v4, 0xffffff8, v4
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v3, v3, v15
	v_lshl_add_u32 v31, v16, 1, s1
	v_lshl_add_u32 v3, v3, 3, v5
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v29, v37 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v29, v37, op_sel:[1, 1] 
             
	;;#ASMEND
	v_lshl_add_u32 v29, v3, 1, v31
	v_add_u32_e32 v3, 64, v17
	v_add_u32_e32 v5, 0x80, v17
	v_add_u32_e32 v6, 0xc0, v17
	v_lshrrev_b32_e32 v3, 3, v3
	v_lshrrev_b32_e32 v5, 3, v5
	v_lshrrev_b32_e32 v6, 3, v6
	v_sub_u32_e32 v3, v3, v19
	v_sub_u32_e32 v5, v5, v19
	v_sub_u32_e32 v6, v6, v19
	v_mul_lo_u32 v3, v3, s0
	v_mul_lo_u32 v5, v5, s0
	v_mul_lo_u32 v6, v6, s0
	v_and_b32_e32 v4, 7, v17
	v_sub_u32_e32 v4, v4, v18
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a48, 0
	v_lshl_add_u32 v4, v4, 3, v20
	v_add_u32_e32 v39, 8, v2
	v_add_u32_e32 v40, 8, v1
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	s_mov_b32 s43, 0
	s_mov_b32 s4, s39
	v_add_u32_e32 v37, v22, v51
	v_lshlrev_b32_e32 v38, 1, v20
	v_add_lshl_u32 v28, v4, v3, 1
	v_add_lshl_u32 v27, v4, v5, 1
	v_add_lshl_u32 v26, v4, v6, 1
	s_add_i32 s30, s42, -4
	s_sub_i32 s33, 0, s37
	s_sub_i32 s34, 0, s44
	s_movk_i32 s39, 0x1100
	v_mov_b32_e32 v41, v40
	v_mov_b32_e32 v42, v39
	ds_write2_b32 v32, v7, v8 offset1:4
	ds_write2_b32 v32, v9, v10 offset0:8 offset1:12
	ds_write2_b32 v32, v11, v12 offset0:16 offset1:20
	ds_write2_b32 v32, v13, v14 offset0:24 offset1:28
BB1_1:                                  ; %_ZZN2ck22move_tensor_coordinateINS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS2_IJiiiEEELb0EEENS3_INS2_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESB_NS_23Merge_v2_magic_divisionINS2_IJiiEEEEESB_NSA_IS7_EENS3_ISD_Lb0EEESB_SF_EEENS2_IJNS_8SequenceIJLi0EEEENSI_IJLi1EEEENSI_IJLi2EEEENSI_IJLi3EEEENSI_IJLi4ELi6EEEENSI_IJLi7EEEENSI_IJLi5EEEENSI_IJLi8EEEENSI_IJLi9EEEENSI_IJLi10EEEEEEENS2_IJNSI_IJLi1ELi2ELi3EEEENSI_IJLi4ELi5EEEENSI_IJLi6EEEESO_SQ_SR_SS_NSI_IJLi11ELi12EEEENSI_IJLi13EEEENSI_IJLi14EEEEEEENSI_IJLi11ELi12ELi13ELi14EEEEiEENS_16TensorCoordinateILi15EKS11_EENS_20TensorCoordinateStepILi10ELi4ENSI_IJLi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0ELi0EEEEEEEEvRKT_RT0_RKT1_ENKUlS19_E_clINS6_IiLi9EEEEEDaS19_.exit.i.i.i.i.i226.i
                                        ; =>This Inner Loop Header: Depth=1
	v_add_u32_e32 v9, s2, v48
	v_add_u32_e32 v13, s2, v9
	v_cmp_le_i32_e32 vcc, s28, v36
	v_cmp_gt_i32_e64 s[0:1], s27, v36
	v_lshlrev_b32_e32 v1, 1, v48
	v_add_u32_e32 v48, s2, v13
	s_and_b64 s[44:45], vcc, s[0:1]
	v_cmp_le_i32_e32 vcc, s25, v37
	v_cmp_gt_i32_e64 s[0:1], s24, v37
	v_lshlrev_b32_e32 v5, 1, v9
	v_lshlrev_b32_e32 v9, 1, v13
	v_lshlrev_b32_e32 v13, 1, v48
	s_and_b64 s[0:1], vcc, s[0:1]
	buffer_load_dwordx4 v[1:4], v1, s[12:15], 0 offen
	s_and_b64 s[0:1], s[0:1], s[44:45]
	buffer_load_dwordx4 v[5:8], v5, s[12:15], 0 offen
	v_add_u32_e32 v55, s39, v38
	buffer_load_dwordx4 v[9:12], v9, s[12:15], 0 offen
	v_add_u32_e32 v71, s29, v29
	buffer_load_dwordx4 v[21:24], v13, s[12:15], 0 offen
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	v_mov_b32_e32 v13, s26
	ds_read2_b64 v[51:54], v38 offset1:1
	v_cndmask_b32_e64 v17, v13, 0, s[0:1]
	v_lshl_add_u32 v13, v49, 1, v17
	v_add_u32_e32 v49, s20, v49
	v_lshl_add_u32 v17, v49, 1, v17
	buffer_load_dwordx4 v[13:16], v13, s[16:19], 0 offen
	v_add_u32_e32 v63, s29, v31
	buffer_load_dwordx4 v[17:20], v17, s[16:19], 0 offen
	ds_read2_b64 v[55:58], v55 offset1:1
	ds_read2_b64 v[59:62], v31 offset1:1
	ds_read2_b64 v[67:70], v29 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[112:127], v[51:52], v[59:60], a[112:127]
	ds_read2_b64 v[71:74], v71 offset1:1
	ds_read2_b64 v[63:66], v63 offset1:1
	v_mul_hi_u32 v75, s10, v41
	v_mul_hi_u32 v50, s4, v42
	v_add_u32_e32 v30, 4, v30
	v_add_u32_e32 v42, 4, v42
	v_add_u32_e32 v41, 4, v41
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[96:111], v[51:52], v[67:68], a[96:111]
	v_mfma_f32_32x32x8f16 a[112:127], v[53:54], v[61:62], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[53:54], v[69:70], a[96:111]
	ds_read2_b64 v[51:54], v28 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[80:95], v[51:52], v[59:60], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[51:52], v[67:68], a[64:79]
	v_mfma_f32_32x32x8f16 a[96:111], v[55:56], v[71:72], a[96:111]
	v_mfma_f32_32x32x8f16 a[80:95], v[53:54], v[61:62], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[53:54], v[69:70], a[64:79]
	ds_read2_b64 v[51:54], v27 offset1:1
	v_mfma_f32_32x32x8f16 a[112:127], v[55:56], v[63:64], a[112:127]
	v_add_u32_e32 v55, s39, v28
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[51:52], v[59:60], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[51:52], v[67:68], a[16:31]
	v_mfma_f32_32x32x8f16 a[112:127], v[57:58], v[65:66], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[57:58], v[73:74], a[96:111]
	ds_read2_b64 v[55:58], v55 offset1:1
	v_mfma_f32_32x32x8f16 a[32:47], v[53:54], v[61:62], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[53:54], v[69:70], a[16:31]
	ds_read2_b64 v[51:54], v26 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[80:95], v[55:56], v[63:64], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[55:56], v[71:72], a[64:79]
	v_add_u32_e32 v55, s39, v27
	v_mfma_f32_32x32x8f16 a[80:95], v[57:58], v[65:66], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[57:58], v[73:74], a[64:79]
	ds_read2_b64 v[55:58], v55 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[0:15], v[51:52], v[59:60], a[0:15]
	v_mfma_f32_32x32x8f16 a[48:63], v[51:52], v[67:68], a[48:63]
	v_add3_u32 v51, v40, v75, s43
	v_lshrrev_b32_e32 v51, s11, v51
	v_mul_lo_u32 v52, s34, v51
	v_sub_u32_e32 v47, v51, v47
	v_lshl_add_u32 v47, v47, 3, -3
	v_mul_lo_u32 v47, v47, s2
	v_sub_u32_e32 v46, v52, v46
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[55:56], v[63:64], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[55:56], v[71:72], a[16:31]
	v_add_u32_e32 v55, s39, v26
	v_mfma_f32_32x32x8f16 a[0:15], v[53:54], v[61:62], a[0:15]
	v_mfma_f32_32x32x8f16 a[48:63], v[53:54], v[69:70], a[48:63]
	v_add_u32_e32 v53, s43, v40
	v_add_u32_e32 v46, v53, v46
	v_mul_lo_u32 v46, v46, s3
	v_add_u32_e32 v54, v53, v52
	v_add3_u32 v48, v47, v48, v46
	v_add3_u32 v46, v39, v50, s43
	v_lshrrev_b32_e32 v46, s41, v46
	v_mul_lo_u32 v50, s33, v46
	v_mul_lo_u32 v47, v46, s37
	v_sub_u32_e32 v45, v50, v45
	v_mul_hi_u32 v50, v46, s38
	v_mfma_f32_32x32x8f16 a[32:47], v[57:58], v[65:66], a[32:47]
	v_add3_u32 v45, v39, s43, v45
	v_mul_lo_u32 v45, v45, s35
	v_add_u32_e32 v50, v46, v50
	v_lshrrev_b32_e32 v50, s40, v50
	v_mul_lo_u32 v52, v50, s36
	v_sub_u32_e32 v43, v50, v43
	v_lshl_add_u32 v43, v43, 3, -1
	v_add_u32_e32 v36, v45, v36
	v_sub_u32_e32 v46, v46, v52
	v_sub_u32_e32 v44, v46, v44
	v_mul_lo_u32 v44, v44, s31
	v_mul_lo_u32 v45, v45, s22
	v_mul_lo_u32 v43, v43, s20
	v_sub_u32_e32 v47, v30, v47
	v_add_u32_e32 v37, v44, v37
	v_mfma_f32_32x32x8f16 a[16:31], v[57:58], v[73:74], a[16:31]
	ds_read2_b64 v[55:58], v55 offset1:1
	v_mul_lo_u32 v44, v44, s21
	v_add_u32_e32 v45, v45, v49
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_add_i32 s43, s43, 4
	v_add3_u32 v49, v45, v43, v44
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v43, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v44, v9, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v9, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v21, v3, v7 
             
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[0:15], v[55:56], v[63:64], a[0:15]
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v11, v23 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v11, v23, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v22, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v24 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v12, v24, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write_b64 v25, v[43:44]
	ds_write2_b32 v25, v1, v5 offset0:4 offset1:5
	ds_write2_b32 v25, v9, v6 offset0:8 offset1:9
	ds_write2_b32 v25, v2, v10 offset0:12 offset1:13
	ds_write2_b32 v25, v21, v7 offset0:16 offset1:17
	ds_write2_b32 v25, v3, v11 offset0:20 offset1:21
	ds_write2_b32 v25, v22, v8 offset0:24 offset1:25
	ds_write2_b32 v25, v4, v12 offset0:28 offset1:29
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v13, v17 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v13, v17, op_sel:[1, 1] 
             
	;;#ASMEND
	s_cmp_lt_i32 s43, s30
	v_mov_b32_e32 v43, v50
	v_mov_b32_e32 v44, v46
	v_mov_b32_e32 v45, v47
	v_mov_b32_e32 v46, v54
	v_mfma_f32_32x32x8f16 a[48:63], v[55:56], v[71:72], a[48:63]
	v_mov_b32_e32 v47, v51
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v14, v18 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v14, v18, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v15, v19 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v15, v19, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v16, v20 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v16, v20, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b32 v32, v1, v2 offset1:4
	ds_write2_b32 v32, v3, v4 offset0:8 offset1:12
	ds_write2_b32 v32, v5, v6 offset0:16 offset1:20
	ds_write2_b32 v32, v7, v8 offset0:24 offset1:28
	v_mfma_f32_32x32x8f16 a[0:15], v[57:58], v[65:66], a[0:15]
	v_mfma_f32_32x32x8f16 a[48:63], v[57:58], v[73:74], a[48:63]
	s_cbranch_scc1 BB1_1
; %bb.2:                                ; %_ZZN2ck23Merge_v2_magic_divisionINS_5TupleIJNS_17integral_constantIiLi4EEENS2_IiLi2EEEiiiEEEEC1ERKS5_ENKUlT_E_clIS4_EEDaS9_.exit.i.i.i.i.i.i.i.i
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v38 offset1:1
	s_movk_i32 s0, 0x1100
	v_add_u32_e32 v5, s0, v38
	ds_read2_b64 v[5:8], v5 offset1:1
	ds_read2_b64 v[9:12], v31 offset1:1
	ds_read2_b64 v[17:20], v29 offset1:1
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[112:127], v[1:2], v[9:10], a[112:127]
	s_movk_i32 s1, 0x880
	v_add_u32_e32 v13, s1, v31
	v_add_u32_e32 v21, s1, v29
	ds_read2_b64 v[13:16], v13 offset1:1
	ds_read2_b64 v[21:24], v21 offset1:1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[96:111], v[1:2], v[17:18], a[96:111]
	v_mfma_f32_32x32x8f16 a[112:127], v[3:4], v[11:12], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[3:4], v[19:20], a[96:111]
	ds_read2_b64 v[1:4], v28 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[80:95], v[1:2], v[9:10], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[1:2], v[17:18], a[64:79]
	v_mfma_f32_32x32x8f16 a[80:95], v[3:4], v[11:12], a[80:95]
	v_mfma_f32_32x32x8f16 a[64:79], v[3:4], v[19:20], a[64:79]
	ds_read2_b64 v[1:4], v27 offset1:1
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[1:2], v[9:10], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[17:18], a[16:31]
	v_mfma_f32_32x32x8f16 a[112:127], v[5:6], v[13:14], a[112:127]
	v_mfma_f32_32x32x8f16 a[96:111], v[5:6], v[21:22], a[96:111]
	v_add_u32_e32 v5, s0, v28
	ds_read2_b64 v[28:31], v5 offset1:1
	v_mfma_f32_32x32x8f16 a[32:47], v[3:4], v[11:12], a[32:47]
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[19:20], a[16:31]
	ds_read2_b64 v[1:4], v26 offset1:1
	v_add_u32_e32 v5, s0, v27
	ds_read2_b64 v[36:39], v5 offset1:1
	v_add_u32_e32 v5, s0, v26
	ds_read2_b64 v[40:43], v5 offset1:1
	s_movk_i32 s0, 0x80
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[17:18], a[48:63]
	v_mfma_f32_32x32x8f16 a[80:95], v[28:29], v[13:14], a[80:95]
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[19:20], a[48:63]
	v_mfma_f32_32x32x8f16 a[0:15], v[1:2], v[9:10], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[48:63], v[40:41], v[21:22], a[48:63]
	v_mfma_f32_32x32x8f16 a[80:95], v[30:31], v[15:16], a[80:95]
	v_mfma_f32_32x32x8f16 a[0:15], v[3:4], v[11:12], a[0:15]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v3, a80              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:68 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a81              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:72 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a82              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:76 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a83              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[48:63], v[42:43], v[23:24], a[48:63]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:80 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a84              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:84 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a85              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:88 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a86              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:92 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a87              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[64:79], v[28:29], v[21:22], a[64:79]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:96 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a88              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:100 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a89              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:104 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a90              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:108 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a91              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[32:47], v[36:37], v[13:14], a[32:47]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:112 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a92              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:116 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a93              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:120 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a94              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:124 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a95              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[16:31], v[36:37], v[21:22], a[16:31]
	v_mul_i32_i24_e32 v36, 0xffffffe0, v33
	buffer_store_dword v3, off, s[64:67], 0 offset:128 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a48              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a49              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a50              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a51              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[0:15], v[40:41], v[13:14], a[0:15]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a52              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a53              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a54              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a55              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[32:47], v[38:39], v[15:16], a[32:47]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a56              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a57              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a58              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a59              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[96:111], v[7:8], v[23:24], a[96:111]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a60              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a61              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a62              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[64:67], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a63              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[16:31], v[38:39], v[23:24], a[16:31]
	s_nop 0
	buffer_store_dword v3, off, s[64:67], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[48:63], v[7:8], v[15:16], a[112:127]
	v_mfma_f32_32x32x8f16 a[80:95], v[30:31], v[23:24], a[64:79]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v17, a48
	v_accvgpr_read_b32 v18, a49
	v_accvgpr_read_b32 v19, a50
	v_accvgpr_read_b32 v20, a51
	v_accvgpr_read_b32 v21, a52
	v_accvgpr_read_b32 v22, a53
	v_accvgpr_read_b32 v23, a54
	v_accvgpr_read_b32 v24, a55
	v_accvgpr_read_b32 v25, a56
	v_accvgpr_read_b32 v26, a57
	v_accvgpr_read_b32 v27, a58
	v_accvgpr_read_b32 v28, a59
	v_accvgpr_read_b32 v29, a60
	v_mfma_f32_32x32x8f16 a[64:79], v[42:43], v[15:16], a[0:15]
	v_accvgpr_read_b32 v30, a61
	v_accvgpr_read_b32 v31, a62
	v_accvgpr_read_b32 v32, a63
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_4
; %bb.3:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s5, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s7
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v50, v2, v4, v1
	v_add3_u32 v49, s6, v2, v3
BB1_4:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES18_S1D_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v51, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v17
	ds_write_b16 v51, v0 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a96
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a97
	v_accvgpr_read_b32 v3, a98
	v_accvgpr_read_b32 v4, a99
	v_accvgpr_read_b32 v5, a100
	v_accvgpr_read_b32 v6, a101
	v_accvgpr_read_b32 v7, a102
	v_accvgpr_read_b32 v8, a103
	v_accvgpr_read_b32 v9, a104
	v_accvgpr_read_b32 v10, a105
	v_accvgpr_read_b32 v11, a106
	v_accvgpr_read_b32 v12, a107
	v_accvgpr_read_b32 v13, a108
	v_accvgpr_read_b32 v14, a109
	v_accvgpr_read_b32 v15, a110
	v_accvgpr_read_b32 v16, a111
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_6
; %bb.5:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v50
	ds_read2_b32 v[17:18], v0 offset0:2 offset1:3
	ds_read2_b32 v[19:20], v0 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v20, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 12 offen
	ds_read2_b32 v[17:18], v0 offset0:6 offset1:7
	ds_read2_b32 v[19:20], v0 offset0:4 offset1:5
	v_add_u32_e32 v21, 8, v49
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v20, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 12 offen
	ds_read2_b32 v[17:18], v0 offset0:38 offset1:39
	ds_read2_b32 v[19:20], v0 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v20, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 12 offen
	ds_read2_b32 v[17:18], v0 offset0:34 offset1:35
	ds_read2_b32 v[19:20], v0 offset0:32 offset1:33
	v_add_lshl_u32 v0, v49, s7, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v19, v0, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v20, v0, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v17, v0, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v18, v0, s[8:11], 12 offen
BB1_6:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v17, a80
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v18, a81
	v_accvgpr_read_b32 v19, a82
	v_accvgpr_read_b32 v20, a83
	v_accvgpr_read_b32 v21, a84
	v_accvgpr_read_b32 v22, a85
	v_accvgpr_read_b32 v23, a86
	v_accvgpr_read_b32 v24, a87
	v_accvgpr_read_b32 v25, a88
	v_accvgpr_read_b32 v26, a89
	v_accvgpr_read_b32 v27, a90
	v_accvgpr_read_b32 v28, a91
	v_accvgpr_read_b32 v29, a92
	v_accvgpr_read_b32 v30, a93
	v_accvgpr_read_b32 v31, a94
	v_accvgpr_read_b32 v32, a95
	v_cvt_f16_f32_e32 v3, v13
	s_mul_i32 s2, s7, 63
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_8
; %bb.7:                                ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	v_add_u32_e32 v6, s7, v49
	v_add_u32_e32 v49, s2, v6
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_lshlrev_b32_e32 v4, 1, v6
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 12 offen
BB1_8:                                  ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v4, off, s[64:67], 0 offset:68 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v0, v17
	v_cvt_f16_f32_e32 v1, v18
	v_cvt_f16_f32_e32 v2, v19
	v_cvt_f16_f32_e32 v3, v20
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:124 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[64:67], 0 offset:128 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v23
	v_cvt_f16_f32_e32 v2, v22
	v_cvt_f16_f32_e32 v3, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v1, v26
	v_cvt_f16_f32_e32 v2, v27
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v31
	v_cvt_f16_f32_e32 v2, v30
	v_cvt_f16_f32_e32 v3, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v4             ;  Reload Reuse
	s_nop 2
	v_accvgpr_read_b32 v48, a15
	v_accvgpr_read_b32 v47, a14
	v_accvgpr_read_b32 v46, a13
	v_accvgpr_read_b32 v45, a12
	v_accvgpr_read_b32 v44, a11
	v_accvgpr_read_b32 v43, a10
	v_accvgpr_read_b32 v42, a9
	v_accvgpr_read_b32 v41, a8
	v_accvgpr_read_b32 v40, a7
	v_accvgpr_read_b32 v39, a6
	v_accvgpr_read_b32 v38, a5
	v_accvgpr_read_b32 v37, a4
	v_accvgpr_read_b32 v36, a3
	v_accvgpr_read_b32 v35, a2
	v_accvgpr_read_b32 v34, a1
	v_accvgpr_read_b32 v33, a0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_10
; %bb.9:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i161.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s7, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 12 offen
BB1_10:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i254.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v33
	v_cvt_f16_f32_e32 v17, v34
	v_cvt_f16_f32_e32 v18, v35
	v_cvt_f16_f32_e32 v19, v36
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v40
	v_cvt_f16_f32_e32 v17, v39
	v_cvt_f16_f32_e32 v18, v38
	v_cvt_f16_f32_e32 v19, v37
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v41
	v_cvt_f16_f32_e32 v17, v42
	v_cvt_f16_f32_e32 v18, v43
	v_cvt_f16_f32_e32 v19, v44
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v48
	v_cvt_f16_f32_e32 v17, v47
	v_accvgpr_read_b32 v0, a32
	v_cvt_f16_f32_e32 v18, v46
	v_accvgpr_read_b32 v1, a33
	v_accvgpr_read_b32 v2, a34
	v_accvgpr_read_b32 v3, a35
	v_accvgpr_read_b32 v4, a36
	v_accvgpr_read_b32 v5, a37
	v_accvgpr_read_b32 v6, a38
	v_accvgpr_read_b32 v7, a39
	v_accvgpr_read_b32 v8, a40
	v_accvgpr_read_b32 v9, a41
	v_accvgpr_read_b32 v10, a42
	v_accvgpr_read_b32 v11, a43
	v_accvgpr_read_b32 v12, a44
	v_accvgpr_read_b32 v13, a45
	v_accvgpr_read_b32 v14, a46
	v_accvgpr_read_b32 v15, a47
	v_cvt_f16_f32_e32 v19, v45
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_12
; %bb.11:                               ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i318.i.i.i
	v_lshlrev_b32_e32 v20, 1, v50
	ds_read2_b32 v[16:17], v20 offset0:2 offset1:3
	ds_read2_b32 v[18:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:6 offset1:7
	ds_read2_b32 v[18:19], v20 offset0:4 offset1:5
	v_add_u32_e32 v21, 8, v49
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:38 offset1:39
	ds_read2_b32 v[18:19], v20 offset0:36 offset1:37
	v_add_u32_e32 v22, s7, v49
	v_add_u32_e32 v49, s2, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:34 offset1:35
	ds_read2_b32 v[18:19], v20 offset0:32 offset1:33
	v_lshlrev_b32_e32 v20, 1, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 12 offen
BB1_12:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I343.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a16
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a17
	v_accvgpr_read_b32 v18, a18
	v_accvgpr_read_b32 v19, a19
	v_accvgpr_read_b32 v20, a20
	v_accvgpr_read_b32 v21, a21
	v_accvgpr_read_b32 v22, a22
	v_accvgpr_read_b32 v23, a23
	v_accvgpr_read_b32 v24, a24
	v_accvgpr_read_b32 v25, a25
	v_accvgpr_read_b32 v26, a26
	v_accvgpr_read_b32 v27, a27
	v_accvgpr_read_b32 v28, a28
	v_accvgpr_read_b32 v29, a29
	v_accvgpr_read_b32 v30, a30
	v_accvgpr_read_b32 v31, a31
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_14
; %bb.13:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i504.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s7, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 12 offen
BB1_14:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i597.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v32, off, s[64:67], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[64:67], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_cvt_f16_f32_e32 v18, v29
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v32            ;  Reload Reuse
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_16
; %bb.15:                               ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i661.i.i.i
	v_lshlrev_b32_e32 v20, 1, v50
	ds_read2_b32 v[16:17], v20 offset0:2 offset1:3
	ds_read2_b32 v[18:19], v20 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v21, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:6 offset1:7
	ds_read2_b32 v[18:19], v20 offset0:4 offset1:5
	v_add_u32_e32 v21, 8, v49
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v22, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v22, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v22, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v22, s[8:11], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:38 offset1:39
	ds_read2_b32 v[18:19], v20 offset0:36 offset1:37
	v_add_u32_e32 v22, s7, v49
	v_add_u32_e32 v49, s2, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v21, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[8:11], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:34 offset1:35
	ds_read2_b32 v[18:19], v20 offset0:32 offset1:33
	v_lshlrev_b32_e32 v20, 1, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v20, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v19, v20, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v16, v20, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v17, v20, s[8:11], 12 offen
BB1_16:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I686.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a64
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a65
	v_accvgpr_read_b32 v18, a66
	v_accvgpr_read_b32 v19, a67
	v_accvgpr_read_b32 v20, a68
	v_accvgpr_read_b32 v21, a69
	v_accvgpr_read_b32 v22, a70
	v_accvgpr_read_b32 v23, a71
	v_accvgpr_read_b32 v24, a72
	v_accvgpr_read_b32 v25, a73
	v_accvgpr_read_b32 v26, a74
	v_accvgpr_read_b32 v27, a75
	v_accvgpr_read_b32 v28, a76
	v_accvgpr_read_b32 v29, a77
	v_accvgpr_read_b32 v30, a78
	v_accvgpr_read_b32 v31, a79
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_18
; %bb.17:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i847.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s7, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 12 offen
BB1_18:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i940.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v17
	v_cvt_f16_f32_e32 v2, v18
	v_cvt_f16_f32_e32 v3, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v1, v22
	v_cvt_f16_f32_e32 v2, v21
	v_cvt_f16_f32_e32 v3, v20
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v25
	v_cvt_f16_f32_e32 v2, v26
	v_cvt_f16_f32_e32 v3, v27
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v1, v30
	v_cvt_f16_f32_e32 v2, v29
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB1_20
; %bb.19:
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_lshl_b32 s10, s23, 1
	s_mov_b32 s11, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[8:11], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s7, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[8:11], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[8:11], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[8:11], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[8:11], 12 offen
BB1_20:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 26112
		.amdhsa_private_segment_fixed_size 132
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 68
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
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
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end1-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 10880
; NumSgprs: 70
; NumVgprs: 76
; NumAgprs: 128
; TotalNumVgprs: 128
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 26112 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 70
; NumVGPRsForWavesPerEU: 128
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[58:59], s[2:3]
	s_mov_b64 s[56:57], s[0:1]
	s_add_u32 s56, s56, s7
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[12:13], s[4:5], 0x8
	s_load_dword s24, s[4:5], 0x48
	s_load_dword s52, s[4:5], 0x50
	s_load_dword s25, s[4:5], 0x58
	s_load_dword s53, s[4:5], 0x70
	s_load_dword s7, s[4:5], 0x84
	s_load_dwordx4 s[16:19], s[4:5], 0x98
	s_load_dwordx4 s[20:23], s[4:5], 0xac
	s_load_dwordx2 s[10:11], s[4:5], 0xbc
	s_load_dwordx2 s[2:3], s[4:5], 0xd4
	s_load_dwordx2 s[8:9], s[4:5], 0xe4
	s_load_dwordx2 s[14:15], s[4:5], 0x114
	s_load_dwordx2 s[30:31], s[4:5], 0x120
	s_load_dwordx2 s[26:27], s[4:5], 0x12c
	s_load_dwordx2 s[28:29], s[4:5], 0x13c
	s_load_dwordx2 s[36:37], s[4:5], 0x148
	s_load_dwordx2 s[34:35], s[4:5], 0x154
	s_load_dword s33, s[4:5], 0x16c
	s_waitcnt lgkmcnt(0)
	s_load_dword s19, s[4:5], 0x180
	s_load_dword s54, s[4:5], 0x1d4
	s_load_dwordx4 s[40:43], s[4:5], 0x1e0
	s_load_dwordx4 s[44:47], s[4:5], 0x1f4
	s_load_dwordx4 s[48:51], s[4:5], 0x208
	s_addc_u32 s57, s57, 0
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s38, s47, s6
	s_add_i32 s38, s6, s38
	s_lshr_b32 s38, s38, s51
	s_mul_i32 s39, s38, s43
	s_sub_i32 s6, s6, s39
	s_mul_hi_u32 s39, s38, s46
	s_add_i32 s39, s38, s39
	s_lshr_b32 s43, s39, s50
	s_mul_i32 s39, s43, s42
	s_sub_i32 s39, s38, s39
	s_mul_hi_u32 s38, s43, s45
	s_add_i32 s38, s43, s38
	s_lshr_b32 s42, s38, s49
	s_mul_i32 s38, s42, s41
	s_sub_i32 s41, s43, s38
	s_mul_hi_u32 s38, s42, s44
	s_add_i32 s38, s42, s38
	s_lshr_b32 s38, s38, s48
	s_mul_i32 s40, s38, s40
	s_sub_i32 s40, s42, s40
	v_mad_i32_i24 v2, v33, -4, v1
	s_mul_i32 s42, s38, s53
	v_add_u32_e32 v3, s42, v2
	v_mul_hi_u32 v4, v3, s52
	s_mul_i32 s41, s41, s54
	s_add_i32 s41, s6, s41
	s_load_dword s42, s[4:5], 0x1c4
	s_load_dword s6, s[4:5], 0x18c
	v_add_u32_e32 v4, v3, v4
	v_lshrrev_b32_e32 v4, s25, v4
	v_mul_lo_u32 v5, v4, s24
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s40, s40, s42
	s_load_dwordx2 s[42:43], s[4:5], 0x24
	s_load_dwordx2 s[24:25], s[4:5], 0x10
	v_lshlrev_b32_e32 v6, 2, v33
	v_sub_u32_e32 v3, v3, v5
	v_lshl_or_b32 v4, v4, 3, v6
	s_add_i32 s40, s40, s39
	s_lshl_b32 s39, s41, 7
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v4, v4, s42
	v_mul_lo_u32 v3, v3, s43
	s_movk_i32 s41, 0xffe0
	s_lshl_b32 s44, s40, 8
	v_mad_i32_i24 v1, v1, s41, v0
	v_lshl_add_u32 v5, v1, 3, s44
	s_movk_i32 s44, 0x44
	v_mul_lo_u32 v27, v1, s44
	v_lshrrev_b32_e32 v1, 4, v0
	v_add3_u32 v3, v5, v4, v3
	v_mad_i32_i24 v4, v1, -16, v0
	v_lshl_add_u32 v7, v4, 3, s39
	v_mul_hi_u32 v8, v7, s37
	v_lshrrev_b32_e32 v5, 6, v0
	v_mad_i32_i24 v9, v5, -4, v1
	s_mul_i32 s38, s38, s33
	v_add_u32_e32 v8, v7, v8
	s_movk_i32 s43, 0x880
	v_add_u32_e32 v10, s38, v9
	v_lshrrev_b32_e32 v8, s35, v8
	v_mul_lo_u32 v2, v2, s43
	v_mul_hi_u32 v11, v8, s36
	v_mul_hi_u32 v12, v10, s31
	v_mul_lo_u32 v16, v8, s29
	v_or_b32_e32 v28, v2, v6
	v_add_u32_e32 v6, v8, v11
	v_add_u32_e32 v11, v10, v12
	v_lshrrev_b32_e32 v11, s27, v11
	v_mul_hi_u32 v12, v11, s30
	v_lshrrev_b32_e32 v6, s34, v6
	v_mul_lo_u32 v13, v6, s28
	v_mul_lo_u32 v14, v11, s15
	v_add_u32_e32 v12, v11, v12
	v_lshrrev_b32_e32 v12, s26, v12
	v_mul_lo_u32 v15, v12, s14
	v_sub_u32_e32 v8, v8, v13
	v_sub_u32_e32 v10, v10, v14
	v_mul_lo_u32 v8, v8, s8
	v_sub_u32_e32 v11, v11, v15
	v_mul_lo_u32 v10, v10, s9
	v_mul_lo_u32 v6, v6, s2
	s_movk_i32 s2, 0x440
	v_mul_lo_u32 v11, v11, s3
	v_mul_lo_u32 v9, v9, s2
	v_lshlrev_b32_e32 v2, 1, v5
	v_add_u32_e32 v17, v10, v8
	v_lshl_or_b32 v12, v12, 3, v2
	v_add_u32_e32 v18, v11, v6
	v_subrev_u32_e32 v6, s10, v17
	v_mul_lo_u32 v29, v4, s44
	v_or_b32_e32 v30, v9, v2
	v_and_b32_e32 v2, 63, v0
	v_and_b32_e32 v4, 32, v0
	v_subrev_u32_e32 v8, s21, v18
	v_mul_lo_u32 v6, v6, s18
	v_sub_u32_e32 v2, v2, v4
	v_lshlrev_b32_e32 v34, 5, v33
	v_mul_lo_u32 v10, v12, s16
	v_mul_lo_u32 v8, v8, s17
	v_add_u32_e32 v36, v2, v34
	v_ashrrev_i16_e32 v4, 15, v36
	v_sub_u32_e32 v7, v7, v16
	v_lshrrev_b16_e32 v4, 13, v4
	v_add_u32_e32 v6, v7, v6
	v_add_u16_e32 v4, v36, v4
	v_add3_u32 v19, v6, v10, v8
	v_ashrrev_i16_e32 v6, 3, v4
	v_and_b32_e32 v4, -8, v4
	v_sub_u16_e32 v4, v36, v4
	v_and_b32_e32 v1, 2, v1
	v_bfe_i32 v37, v6, 0, 16
	v_bfe_i32 v31, v4, 0, 16
	v_mul_u32_u24_e32 v4, s43, v1
	v_mul_i32_i24_e32 v6, s44, v37
	v_lshlrev_b32_e32 v7, 3, v31
	v_add3_u32 v32, v6, v4, v7
	v_mad_i32_i24 v4, v33, -2, v5
	v_lshl_add_u32 v35, v4, 5, v2
	v_ashrrev_i32_e32 v2, 31, v35
	v_lshrrev_b32_e32 v2, 29, v2
	v_add_u32_e32 v2, v35, v2
	v_ashrrev_i32_e32 v38, 3, v2
	v_mul_lo_u32 v4, v38, s44
	v_and_b32_e32 v2, -8, v2
	s_mov_b32 s3, 0x20000
	v_lshlrev_b32_e32 v9, 1, v3
	v_mad_u32_u24 v41, v1, s2, v4
	s_lshl_b32 s2, s7, 1
	v_add_u32_e32 v10, s42, v3
	v_sub_u32_e32 v39, v35, v2
	v_lshlrev_b32_e32 v11, 1, v10
	buffer_load_dwordx4 v[1:4], v9, s[0:3], 0 offen
	buffer_load_dwordx4 v[5:8], v11, s[0:3], 0 offen
	v_add_u32_e32 v9, s42, v10
	v_lshlrev_b32_e32 v20, 1, v9
	v_add_lshl_u32 v21, v9, s42, 1
	buffer_load_dwordx4 v[9:12], v20, s[0:3], 0 offen
	buffer_load_dwordx4 v[13:16], v21, s[0:3], 0 offen
	s_sub_i32 s0, s23, s11
	v_cmp_le_i32_e32 vcc, s10, v17
	v_cmp_gt_i32_e64 s[0:1], s0, v17
	s_and_b64 s[10:11], vcc, s[0:1]
	s_sub_i32 s0, s20, s22
	v_cmp_le_i32_e32 vcc, s21, v18
	v_cmp_gt_i32_e64 s[0:1], s0, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	v_bfrev_b32_e32 v17, -2
	s_and_b64 s[0:1], s[10:11], s[0:1]
	v_cndmask_b32_e64 v17, v17, 0, s[0:1]
	s_lshl_b32 s14, s19, 1
	s_mov_b32 s15, s3
	v_lshl_add_u32 v25, v19, 1, v17
	v_add_u32_e32 v18, s16, v19
	v_lshl_add_u32 v26, v18, 1, v17
	buffer_load_dwordx4 v[17:20], v25, s[12:15], 0 offen
	buffer_load_dwordx4 v[21:24], v26, s[12:15], 0 offen
	s_mov_b32 s8, 0
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	s_mov_b32 s16, s8
	s_mov_b32 s17, s8
	s_mov_b32 s18, s8
	s_mov_b32 s19, s8
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v3, v7 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v11, v15 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v11, v15, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_lshl_u32 v15, v28, v27, 1
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v16 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v12, v16, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write_b64 v15, v[25:26]
	ds_write2_b32 v15, v1, v5 offset0:4 offset1:5
	ds_write2_b32 v15, v9, v6 offset0:8 offset1:9
	ds_write2_b32 v15, v2, v10 offset0:12 offset1:13
	ds_write2_b32 v15, v13, v7 offset0:16 offset1:17
	ds_write2_b32 v15, v3, v11 offset0:20 offset1:21
	ds_write2_b32 v15, v14, v8 offset0:24 offset1:25
	ds_write2_b32 v15, v4, v12 offset0:28 offset1:29
	s_mov_b32 s20, s8
	s_mov_b32 s21, s8
	s_mov_b32 s22, s8
	s_mov_b32 s23, s8
	v_mov_b32_e32 v25, s8
	v_mov_b32_e32 v26, s9
	v_mov_b32_e32 v27, s10
	v_accvgpr_write_b32 a0, v25
	v_mov_b32_e32 v25, s11
	v_accvgpr_write_b32 a1, v26
	v_accvgpr_write_b32 a2, v27
	v_accvgpr_write_b32 a3, v25
	v_mov_b32_e32 v26, s12
	v_mov_b32_e32 v27, s13
	v_mov_b32_e32 v25, s14
	v_accvgpr_write_b32 a4, v26
	v_accvgpr_write_b32 a5, v27
	v_accvgpr_write_b32 a6, v25
	v_mov_b32_e32 v26, s15
	v_mov_b32_e32 v27, s16
	v_mov_b32_e32 v25, s17
	v_add_lshl_u32 v9, v30, v29, 1
	s_movk_i32 s0, 0x4400
	v_lshlrev_b32_e32 v40, 3, v39
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v17, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v17, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v9, s0, v9
	v_accvgpr_write_b32 a7, v26
	v_accvgpr_write_b32 a8, v27
	v_accvgpr_write_b32 a9, v25
	v_mov_b32_e32 v26, s18
	v_mov_b32_e32 v27, s19
	v_mov_b32_e32 v25, s20
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v18, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v18, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v19, v23 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v19, v23, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v20, v24 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v20, v24, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b32 v9, v1, v2 offset1:4
	ds_write2_b32 v9, v3, v4 offset0:8 offset1:12
	ds_write2_b32 v9, v5, v6 offset0:16 offset1:20
	ds_write2_b32 v9, v7, v8 offset0:24 offset1:28
	v_add_lshl_u32 v9, v41, v40, 1
	v_lshlrev_b32_e32 v1, 1, v32
	v_add_u32_e32 v17, s0, v9
	s_movk_i32 s0, 0x1100
	v_accvgpr_write_b32 a10, v26
	v_accvgpr_write_b32 a11, v27
	v_accvgpr_write_b32 a12, v25
	v_mov_b32_e32 v26, s21
	v_mov_b32_e32 v27, s22
	v_mov_b32_e32 v25, s23
	v_add_u32_e32 v5, s0, v1
	v_add_u32_e32 v13, 0x4c80, v9
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[5:8], v5 offset1:1
	ds_read2_b64 v[9:12], v17 offset1:1
	ds_read2_b64 v[13:16], v13 offset1:1
	v_accvgpr_write_b32 a13, v26
	v_accvgpr_write_b32 a14, v27
	v_accvgpr_write_b32 a15, v25
	v_add_u32_e32 v18, 64, v35
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[9:10], a[0:15]
	v_ashrrev_i32_e32 v19, 31, v18
	s_movk_i32 s1, 0x80
	v_lshrrev_b32_e32 v19, 29, v19
	v_add_u32_e32 v19, v18, v19
	v_ashrrev_i32_e32 v20, 3, v19
	v_sub_u32_e32 v20, v20, v38
	v_mul_lo_u32 v20, v20, s44
	v_and_b32_e32 v19, 0xffffff8, v19
	v_sub_u32_e32 v18, v18, v19
	v_sub_u32_e32 v18, v18, v39
	v_lshl_add_u32 v18, v18, 3, v20
	v_lshl_add_u32 v17, v18, 1, v17
	v_add_u32_e32 v21, s43, v17
	ds_read2_b64 v[17:20], v17 offset1:1
	ds_read2_b64 v[21:24], v21 offset1:1
	v_mov_b32_e32 v49, 0
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[11:12], a[16:31]
	v_cmp_gt_u32_e32 vcc, s1, v0
	v_mov_b32_e32 v50, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[112:127], v[5:6], v[13:14], a[16:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[17:18], a[0:15]
	v_add_u32_e32 v1, 64, v36
	v_lshrrev_b32_e32 v1, 3, v1
	v_sub_u32_e32 v1, v1, v37
	v_mul_lo_u32 v1, v1, s44
	v_and_b32_e32 v2, 7, v36
	v_sub_u32_e32 v2, v2, v31
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[19:20], a[16:31]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[5:6], v[21:22], a[16:31]
	v_lshl_add_u32 v5, v2, 3, v32
	v_add_lshl_u32 v1, v5, v1, 1
	v_add_u32_e32 v6, s0, v1
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[25:28], v6 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[9:10], a[0:15]
	v_add_u32_e32 v1, s1, v36
	v_lshrrev_b32_e32 v1, 3, v1
	v_sub_u32_e32 v1, v1, v37
	v_mul_lo_u32 v1, v1, s44
	v_add_lshl_u32 v1, v5, v1, 1
	v_add_u32_e32 v6, s0, v1
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[19:20], a[48:63]
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[11:12], a[16:31]
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[29:32], v6 offset1:1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[80:95], v[25:26], v[21:22], a[48:63]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[9:10], a[0:15]
	v_mfma_f32_32x32x8f16 a[64:79], v[1:2], v[17:18], a[0:15]
	v_add_u32_e32 v1, 0xc0, v36
	v_lshrrev_b32_e32 v1, 3, v1
	v_sub_u32_e32 v1, v1, v37
	v_mul_lo_u32 v1, v1, s44
	v_add_lshl_u32 v1, v5, v1, 1
	v_add_u32_e32 v5, s0, v1
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[11:12], a[48:63]
	v_mfma_f32_32x32x8f16 a[64:79], v[3:4], v[19:20], a[64:79]
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[36:39], v5 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[96:111], v[1:2], v[9:10], a[0:15]
	v_mfma_f32_32x32x8f16 a[0:15], v[1:2], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[25:26], v[13:14], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[3:4], v[19:20], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[0:15], v[36:37], v[21:22], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[27:28], v[15:16], a[16:31]
	v_mfma_f32_32x32x8f16 a[96:111], v[3:4], v[11:12], a[96:111]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v3, a16              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:68 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a17              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:72 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a18              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:76 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a19              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[0:15], v[38:39], v[23:24], a[0:15]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:80 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a20              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:84 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a21              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:88 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a22              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:92 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a23              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[48:63], v[29:30], v[13:14], a[48:63]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:96 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a24              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:100 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a25              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:104 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a26              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:108 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a27              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[64:79], v[29:30], v[21:22], a[64:79]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:112 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a28              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:116 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a29              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:120 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a30              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:124 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a31              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[96:111], v[36:37], v[13:14], a[96:111]
	v_mul_i32_i24_e32 v36, s41, v33
	buffer_store_dword v3, off, s[56:59], 0 offset:128 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a0               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a1               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a2               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a3               ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[48:63], v[31:32], v[15:16], a[48:63]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a4               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a5               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a6               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a7               ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[16:31], v[7:8], v[23:24], a[32:47]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a8               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a9               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a10              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a11              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[32:47], v[31:32], v[23:24], a[64:79]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a12              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a13              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a14              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a15              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[80:95], v[27:28], v[23:24], a[80:95]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[0:15], v[7:8], v[15:16], a[112:127]
	v_mfma_f32_32x32x8f16 a[64:79], v[38:39], v[15:16], a[96:111]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v32, a15
	v_accvgpr_read_b32 v31, a14
	v_accvgpr_read_b32 v30, a13
	v_accvgpr_read_b32 v29, a12
	v_accvgpr_read_b32 v28, a11
	v_accvgpr_read_b32 v27, a10
	v_accvgpr_read_b32 v26, a9
	v_accvgpr_read_b32 v25, a8
	v_accvgpr_read_b32 v24, a7
	v_accvgpr_read_b32 v23, a6
	v_accvgpr_read_b32 v22, a5
	v_accvgpr_read_b32 v21, a4
	v_accvgpr_read_b32 v20, a3
	v_accvgpr_read_b32 v19, a2
	v_accvgpr_read_b32 v18, a1
	v_accvgpr_read_b32 v17, a0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s40, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s6
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v50, v2, v4, v1
	v_add3_u32 v49, s39, v2, v3
BB2_2:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES18_S1D_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v51, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v17
	ds_write_b16 v51, v0 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	s_load_dword s2, s[4:5], 0x1b0
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a16
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a17
	v_accvgpr_read_b32 v3, a18
	v_accvgpr_read_b32 v4, a19
	v_accvgpr_read_b32 v5, a20
	v_accvgpr_read_b32 v6, a21
	v_accvgpr_read_b32 v7, a22
	v_accvgpr_read_b32 v8, a23
	v_accvgpr_read_b32 v9, a24
	v_accvgpr_read_b32 v10, a25
	v_accvgpr_read_b32 v11, a26
	v_accvgpr_read_b32 v12, a27
	v_accvgpr_read_b32 v13, a28
	v_accvgpr_read_b32 v14, a29
	v_accvgpr_read_b32 v15, a30
	v_accvgpr_read_b32 v16, a31
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_4
; %bb.3:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v50
	ds_read2_b64 v[17:20], v0 offset1:1
	ds_read2_b64 v[21:24], v0 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v49
	buffer_store_dwordx4 v[17:20], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v17, 8, v49
	v_lshlrev_b32_e32 v18, 1, v17
	buffer_store_dwordx4 v[21:24], v18, s[24:27], 0 offen
	v_add_lshl_u32 v25, v17, s6, 1
	ds_read2_b64 v[17:20], v0 offset0:18 offset1:19
	ds_read2_b64 v[21:24], v0 offset0:16 offset1:17
	v_add_lshl_u32 v0, v49, s6, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[17:20], v25, s[24:27], 0 offen
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[21:24], v0, s[24:27], 0 offen
BB2_4:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v17, a80
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v18, a81
	v_accvgpr_read_b32 v19, a82
	v_accvgpr_read_b32 v20, a83
	v_accvgpr_read_b32 v21, a84
	v_accvgpr_read_b32 v22, a85
	v_accvgpr_read_b32 v23, a86
	v_accvgpr_read_b32 v24, a87
	v_accvgpr_read_b32 v25, a88
	v_accvgpr_read_b32 v26, a89
	v_accvgpr_read_b32 v27, a90
	v_accvgpr_read_b32 v28, a91
	v_accvgpr_read_b32 v29, a92
	v_accvgpr_read_b32 v30, a93
	v_accvgpr_read_b32 v31, a94
	v_accvgpr_read_b32 v32, a95
	v_cvt_f16_f32_e32 v3, v13
	s_mul_i32 s3, s6, 63
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_6
; %bb.5:                                ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, s6, v49
	v_lshlrev_b32_e32 v1, 1, v0
	v_add_u32_e32 v49, s3, v0
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
BB2_6:                                  ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v4, off, s[56:59], 0 offset:68 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v0, v17
	v_cvt_f16_f32_e32 v1, v18
	v_cvt_f16_f32_e32 v2, v19
	v_cvt_f16_f32_e32 v3, v20
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:124 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:128 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v23
	v_cvt_f16_f32_e32 v2, v22
	v_cvt_f16_f32_e32 v3, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v1, v26
	v_cvt_f16_f32_e32 v2, v27
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v31
	v_cvt_f16_f32_e32 v2, v30
	v_cvt_f16_f32_e32 v3, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v4             ;  Reload Reuse
	s_nop 2
	v_accvgpr_read_b32 v48, a15
	v_accvgpr_read_b32 v47, a14
	v_accvgpr_read_b32 v46, a13
	v_accvgpr_read_b32 v45, a12
	v_accvgpr_read_b32 v44, a11
	v_accvgpr_read_b32 v43, a10
	v_accvgpr_read_b32 v42, a9
	v_accvgpr_read_b32 v41, a8
	v_accvgpr_read_b32 v40, a7
	v_accvgpr_read_b32 v39, a6
	v_accvgpr_read_b32 v38, a5
	v_accvgpr_read_b32 v37, a4
	v_accvgpr_read_b32 v36, a3
	v_accvgpr_read_b32 v35, a2
	v_accvgpr_read_b32 v34, a1
	v_accvgpr_read_b32 v33, a0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_8
; %bb.7:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i141.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v49, s6, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_8:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i234.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v33
	v_cvt_f16_f32_e32 v17, v34
	v_cvt_f16_f32_e32 v18, v35
	v_cvt_f16_f32_e32 v19, v36
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v40
	v_cvt_f16_f32_e32 v17, v39
	v_cvt_f16_f32_e32 v18, v38
	v_cvt_f16_f32_e32 v19, v37
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v41
	v_cvt_f16_f32_e32 v17, v42
	v_cvt_f16_f32_e32 v18, v43
	v_cvt_f16_f32_e32 v19, v44
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v48
	v_cvt_f16_f32_e32 v17, v47
	v_accvgpr_read_b32 v0, a48
	v_cvt_f16_f32_e32 v18, v46
	v_accvgpr_read_b32 v1, a49
	v_accvgpr_read_b32 v2, a50
	v_accvgpr_read_b32 v3, a51
	v_accvgpr_read_b32 v4, a52
	v_accvgpr_read_b32 v5, a53
	v_accvgpr_read_b32 v6, a54
	v_accvgpr_read_b32 v7, a55
	v_accvgpr_read_b32 v8, a56
	v_accvgpr_read_b32 v9, a57
	v_accvgpr_read_b32 v10, a58
	v_accvgpr_read_b32 v11, a59
	v_accvgpr_read_b32 v12, a60
	v_accvgpr_read_b32 v13, a61
	v_accvgpr_read_b32 v14, a62
	v_accvgpr_read_b32 v15, a63
	v_cvt_f16_f32_e32 v19, v45
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_10
; %bb.9:                                ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i278.i.i.i
	v_lshlrev_b32_e32 v24, 1, v50
	ds_read2_b64 v[16:19], v24 offset1:1
	ds_read2_b64 v[20:23], v24 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v49
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v49
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read2_b64 v[16:19], v24 offset0:18 offset1:19
	ds_read2_b64 v[20:23], v24 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s6, v49
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v49, s3, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
BB2_10:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I303.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a32
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a33
	v_accvgpr_read_b32 v18, a34
	v_accvgpr_read_b32 v19, a35
	v_accvgpr_read_b32 v20, a36
	v_accvgpr_read_b32 v21, a37
	v_accvgpr_read_b32 v22, a38
	v_accvgpr_read_b32 v23, a39
	v_accvgpr_read_b32 v24, a40
	v_accvgpr_read_b32 v25, a41
	v_accvgpr_read_b32 v26, a42
	v_accvgpr_read_b32 v27, a43
	v_accvgpr_read_b32 v28, a44
	v_accvgpr_read_b32 v29, a45
	v_accvgpr_read_b32 v30, a46
	v_accvgpr_read_b32 v31, a47
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_12
; %bb.11:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i444.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v49, s6, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_12:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i537.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v32, off, s[56:59], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_cvt_f16_f32_e32 v18, v29
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v32            ;  Reload Reuse
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_14
; %bb.13:                               ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i581.i.i.i
	v_lshlrev_b32_e32 v24, 1, v50
	ds_read2_b64 v[16:19], v24 offset1:1
	ds_read2_b64 v[20:23], v24 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v25, 1, v49
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, 8, v49
	v_lshlrev_b32_e32 v17, 1, v16
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
	v_add_lshl_u32 v25, v16, s6, 1
	ds_read2_b64 v[16:19], v24 offset0:18 offset1:19
	ds_read2_b64 v[20:23], v24 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[16:19], v25, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v16, s6, v49
	v_lshlrev_b32_e32 v17, 1, v16
	v_add_u32_e32 v49, s3, v16
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[20:23], v17, s[24:27], 0 offen
BB2_14:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I606.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a64
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a65
	v_accvgpr_read_b32 v18, a66
	v_accvgpr_read_b32 v19, a67
	v_accvgpr_read_b32 v20, a68
	v_accvgpr_read_b32 v21, a69
	v_accvgpr_read_b32 v22, a70
	v_accvgpr_read_b32 v23, a71
	v_accvgpr_read_b32 v24, a72
	v_accvgpr_read_b32 v25, a73
	v_accvgpr_read_b32 v26, a74
	v_accvgpr_read_b32 v27, a75
	v_accvgpr_read_b32 v28, a76
	v_accvgpr_read_b32 v29, a77
	v_accvgpr_read_b32 v30, a78
	v_accvgpr_read_b32 v31, a79
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_16
; %bb.15:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE0ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i747.i.i.i
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v49, s6, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_16:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i840.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v17
	v_cvt_f16_f32_e32 v2, v18
	v_cvt_f16_f32_e32 v3, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v1, v22
	v_cvt_f16_f32_e32 v2, v21
	v_cvt_f16_f32_e32 v3, v20
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v25
	v_cvt_f16_f32_e32 v2, v26
	v_cvt_f16_f32_e32 v3, v27
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v1, v30
	v_cvt_f16_f32_e32 v2, v29
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB2_18
; %bb.17:
	v_lshlrev_b32_e32 v8, 1, v50
	ds_read2_b64 v[0:3], v8 offset1:1
	ds_read2_b64 v[4:7], v8 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v9, 1, v49
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_u32_e32 v0, 8, v49
	v_lshlrev_b32_e32 v1, 1, v0
	buffer_store_dwordx4 v[4:7], v1, s[24:27], 0 offen
	v_add_lshl_u32 v9, v0, s6, 1
	ds_read2_b64 v[0:3], v8 offset0:18 offset1:19
	ds_read2_b64 v[4:7], v8 offset0:16 offset1:17
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[0:3], v9, s[24:27], 0 offen
	s_nop 0
	v_add_lshl_u32 v0, v49, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[4:7], v0, s[24:27], 0 offen
BB2_18:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 26112
		.amdhsa_private_segment_fixed_size 132
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 60
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
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
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end2-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 7704
; NumSgprs: 62
; NumVgprs: 52
; NumAgprs: 128
; TotalNumVgprs: 128
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 26112 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 128
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_ ; -- Begin function _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.globl	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
	.p2align	8
	.type	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_,@function
_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_: ; @_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
; %bb.0:
	s_mov_b64 s[58:59], s[2:3]
	s_mov_b64 s[56:57], s[0:1]
	s_add_u32 s56, s56, s7
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_load_dwordx2 s[12:13], s[4:5], 0x8
	s_load_dword s24, s[4:5], 0x48
	s_load_dword s52, s[4:5], 0x50
	s_load_dword s25, s[4:5], 0x58
	s_load_dword s53, s[4:5], 0x70
	s_load_dword s7, s[4:5], 0x84
	s_load_dwordx4 s[16:19], s[4:5], 0x98
	s_load_dwordx4 s[20:23], s[4:5], 0xac
	s_load_dwordx2 s[10:11], s[4:5], 0xbc
	s_load_dwordx2 s[2:3], s[4:5], 0xd4
	s_load_dwordx2 s[8:9], s[4:5], 0xe4
	s_load_dwordx2 s[14:15], s[4:5], 0x114
	s_load_dwordx2 s[30:31], s[4:5], 0x120
	s_load_dwordx2 s[26:27], s[4:5], 0x12c
	s_load_dwordx2 s[28:29], s[4:5], 0x13c
	s_load_dwordx2 s[36:37], s[4:5], 0x148
	s_load_dwordx2 s[34:35], s[4:5], 0x154
	s_load_dword s33, s[4:5], 0x16c
	s_waitcnt lgkmcnt(0)
	s_load_dword s19, s[4:5], 0x180
	s_load_dword s54, s[4:5], 0x1d4
	s_load_dwordx4 s[40:43], s[4:5], 0x1e0
	s_load_dwordx4 s[44:47], s[4:5], 0x1f4
	s_load_dwordx4 s[48:51], s[4:5], 0x208
	s_addc_u32 s57, s57, 0
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshrrev_b32_e32 v33, 7, v0
	s_waitcnt lgkmcnt(0)
	s_mul_hi_u32 s38, s47, s6
	s_add_i32 s38, s6, s38
	s_lshr_b32 s38, s38, s51
	s_mul_i32 s39, s38, s43
	s_sub_i32 s6, s6, s39
	s_mul_hi_u32 s39, s38, s46
	s_add_i32 s39, s38, s39
	s_lshr_b32 s43, s39, s50
	s_mul_i32 s39, s43, s42
	s_sub_i32 s39, s38, s39
	s_mul_hi_u32 s38, s43, s45
	s_add_i32 s38, s43, s38
	s_lshr_b32 s42, s38, s49
	s_mul_i32 s38, s42, s41
	s_sub_i32 s41, s43, s38
	s_mul_hi_u32 s38, s42, s44
	s_add_i32 s38, s42, s38
	s_lshr_b32 s38, s38, s48
	s_mul_i32 s40, s38, s40
	s_sub_i32 s40, s42, s40
	v_mad_i32_i24 v2, v33, -4, v1
	s_mul_i32 s42, s38, s53
	v_add_u32_e32 v3, s42, v2
	v_mul_hi_u32 v4, v3, s52
	s_mul_i32 s41, s41, s54
	s_add_i32 s41, s6, s41
	s_load_dword s42, s[4:5], 0x1c4
	s_load_dword s6, s[4:5], 0x18c
	v_add_u32_e32 v4, v3, v4
	v_lshrrev_b32_e32 v4, s25, v4
	v_mul_lo_u32 v5, v4, s24
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s40, s40, s42
	s_load_dwordx2 s[42:43], s[4:5], 0x24
	s_load_dwordx2 s[24:25], s[4:5], 0x10
	v_lshlrev_b32_e32 v6, 2, v33
	v_sub_u32_e32 v3, v3, v5
	v_lshl_or_b32 v4, v4, 3, v6
	s_add_i32 s40, s40, s39
	s_lshl_b32 s39, s41, 7
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v4, v4, s42
	v_mul_lo_u32 v3, v3, s43
	s_movk_i32 s41, 0xffe0
	s_lshl_b32 s44, s40, 8
	v_mad_i32_i24 v1, v1, s41, v0
	v_lshl_add_u32 v5, v1, 3, s44
	s_movk_i32 s44, 0x44
	v_mul_lo_u32 v27, v1, s44
	v_lshrrev_b32_e32 v1, 4, v0
	v_add3_u32 v3, v5, v4, v3
	v_mad_i32_i24 v4, v1, -16, v0
	v_lshl_add_u32 v7, v4, 3, s39
	v_mul_hi_u32 v8, v7, s37
	v_lshrrev_b32_e32 v5, 6, v0
	v_mad_i32_i24 v9, v5, -4, v1
	s_mul_i32 s38, s38, s33
	v_add_u32_e32 v8, v7, v8
	s_movk_i32 s43, 0x880
	v_add_u32_e32 v10, s38, v9
	v_lshrrev_b32_e32 v8, s35, v8
	v_mul_lo_u32 v2, v2, s43
	v_mul_hi_u32 v11, v8, s36
	v_mul_hi_u32 v12, v10, s31
	v_mul_lo_u32 v16, v8, s29
	v_or_b32_e32 v28, v2, v6
	v_add_u32_e32 v6, v8, v11
	v_add_u32_e32 v11, v10, v12
	v_lshrrev_b32_e32 v11, s27, v11
	v_mul_hi_u32 v12, v11, s30
	v_lshrrev_b32_e32 v6, s34, v6
	v_mul_lo_u32 v13, v6, s28
	v_mul_lo_u32 v14, v11, s15
	v_add_u32_e32 v12, v11, v12
	v_lshrrev_b32_e32 v12, s26, v12
	v_mul_lo_u32 v15, v12, s14
	v_sub_u32_e32 v8, v8, v13
	v_sub_u32_e32 v10, v10, v14
	v_mul_lo_u32 v8, v8, s8
	v_sub_u32_e32 v11, v11, v15
	v_mul_lo_u32 v10, v10, s9
	v_mul_lo_u32 v6, v6, s2
	s_movk_i32 s2, 0x440
	v_mul_lo_u32 v11, v11, s3
	v_mul_lo_u32 v9, v9, s2
	v_lshlrev_b32_e32 v2, 1, v5
	v_add_u32_e32 v17, v10, v8
	v_lshl_or_b32 v12, v12, 3, v2
	v_add_u32_e32 v18, v11, v6
	v_subrev_u32_e32 v6, s10, v17
	v_mul_lo_u32 v29, v4, s44
	v_or_b32_e32 v30, v9, v2
	v_and_b32_e32 v2, 63, v0
	v_and_b32_e32 v4, 32, v0
	v_subrev_u32_e32 v8, s21, v18
	v_mul_lo_u32 v6, v6, s18
	v_sub_u32_e32 v2, v2, v4
	v_lshlrev_b32_e32 v34, 5, v33
	v_mul_lo_u32 v10, v12, s16
	v_mul_lo_u32 v8, v8, s17
	v_add_u32_e32 v36, v2, v34
	v_ashrrev_i16_e32 v4, 15, v36
	v_sub_u32_e32 v7, v7, v16
	v_lshrrev_b16_e32 v4, 13, v4
	v_add_u32_e32 v6, v7, v6
	v_add_u16_e32 v4, v36, v4
	v_add3_u32 v19, v6, v10, v8
	v_ashrrev_i16_e32 v6, 3, v4
	v_and_b32_e32 v4, -8, v4
	v_sub_u16_e32 v4, v36, v4
	v_and_b32_e32 v1, 2, v1
	v_bfe_i32 v37, v6, 0, 16
	v_bfe_i32 v31, v4, 0, 16
	v_mul_u32_u24_e32 v4, s43, v1
	v_mul_i32_i24_e32 v6, s44, v37
	v_lshlrev_b32_e32 v7, 3, v31
	v_add3_u32 v32, v6, v4, v7
	v_mad_i32_i24 v4, v33, -2, v5
	v_lshl_add_u32 v35, v4, 5, v2
	v_ashrrev_i32_e32 v2, 31, v35
	v_lshrrev_b32_e32 v2, 29, v2
	v_add_u32_e32 v2, v35, v2
	v_ashrrev_i32_e32 v38, 3, v2
	v_mul_lo_u32 v4, v38, s44
	v_and_b32_e32 v2, -8, v2
	s_mov_b32 s3, 0x20000
	v_lshlrev_b32_e32 v9, 1, v3
	v_mad_u32_u24 v41, v1, s2, v4
	s_lshl_b32 s2, s7, 1
	v_add_u32_e32 v10, s42, v3
	v_sub_u32_e32 v39, v35, v2
	v_lshlrev_b32_e32 v11, 1, v10
	buffer_load_dwordx4 v[1:4], v9, s[0:3], 0 offen
	buffer_load_dwordx4 v[5:8], v11, s[0:3], 0 offen
	v_add_u32_e32 v9, s42, v10
	v_lshlrev_b32_e32 v20, 1, v9
	v_add_lshl_u32 v21, v9, s42, 1
	buffer_load_dwordx4 v[9:12], v20, s[0:3], 0 offen
	buffer_load_dwordx4 v[13:16], v21, s[0:3], 0 offen
	s_sub_i32 s0, s23, s11
	v_cmp_le_i32_e32 vcc, s10, v17
	v_cmp_gt_i32_e64 s[0:1], s0, v17
	s_and_b64 s[10:11], vcc, s[0:1]
	s_sub_i32 s0, s20, s22
	v_cmp_le_i32_e32 vcc, s21, v18
	v_cmp_gt_i32_e64 s[0:1], s0, v18
	s_and_b64 s[0:1], vcc, s[0:1]
	v_bfrev_b32_e32 v17, -2
	s_and_b64 s[0:1], s[10:11], s[0:1]
	v_cndmask_b32_e64 v17, v17, 0, s[0:1]
	s_lshl_b32 s14, s19, 1
	s_mov_b32 s15, s3
	v_lshl_add_u32 v25, v19, 1, v17
	v_add_u32_e32 v18, s16, v19
	v_lshl_add_u32 v26, v18, 1, v17
	buffer_load_dwordx4 v[17:20], v25, s[12:15], 0 offen
	buffer_load_dwordx4 v[21:24], v26, s[12:15], 0 offen
	s_mov_b32 s8, 0
	s_mov_b32 s9, s8
	s_mov_b32 s10, s8
	s_mov_b32 s11, s8
	s_mov_b32 s12, s8
	s_mov_b32 s13, s8
	s_mov_b32 s14, s8
	s_mov_b32 s15, s8
	s_mov_b32 s16, s8
	s_mov_b32 s17, s8
	s_mov_b32 s18, s8
	s_mov_b32 s19, s8
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	
             v_pack_b32_f16 v25, v1, v5 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v1, v5, op_sel:[1, 1] 
             
	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	
             v_pack_b32_f16 v26, v9, v13 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v9, v13, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v9, v2, v6 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v2, v6, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v10, v14 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v10, v10, v14, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v13, v3, v7 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v3, v7, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v11, v15 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v11, v11, v15, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_lshl_u32 v15, v28, v27, 1
	;;#ASMSTART
	
             v_pack_b32_f16 v14, v4, v8 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v4, v8, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v12, v16 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v12, v12, v16, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write_b64 v15, v[25:26]
	ds_write2_b32 v15, v1, v5 offset0:4 offset1:5
	ds_write2_b32 v15, v9, v6 offset0:8 offset1:9
	ds_write2_b32 v15, v2, v10 offset0:12 offset1:13
	ds_write2_b32 v15, v13, v7 offset0:16 offset1:17
	ds_write2_b32 v15, v3, v11 offset0:20 offset1:21
	ds_write2_b32 v15, v14, v8 offset0:24 offset1:25
	ds_write2_b32 v15, v4, v12 offset0:28 offset1:29
	s_mov_b32 s20, s8
	s_mov_b32 s21, s8
	s_mov_b32 s22, s8
	s_mov_b32 s23, s8
	v_mov_b32_e32 v25, s8
	v_mov_b32_e32 v26, s9
	v_mov_b32_e32 v27, s10
	v_accvgpr_write_b32 a0, v25
	v_mov_b32_e32 v25, s11
	v_accvgpr_write_b32 a1, v26
	v_accvgpr_write_b32 a2, v27
	v_accvgpr_write_b32 a3, v25
	v_mov_b32_e32 v26, s12
	v_mov_b32_e32 v27, s13
	v_mov_b32_e32 v25, s14
	v_accvgpr_write_b32 a4, v26
	v_accvgpr_write_b32 a5, v27
	v_accvgpr_write_b32 a6, v25
	v_mov_b32_e32 v26, s15
	v_mov_b32_e32 v27, s16
	v_mov_b32_e32 v25, s17
	v_add_lshl_u32 v9, v30, v29, 1
	s_movk_i32 s0, 0x4400
	v_lshlrev_b32_e32 v40, 3, v39
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	
             v_pack_b32_f16 v1, v17, v21 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v2, v17, v21, op_sel:[1, 1] 
             
	;;#ASMEND
	v_add_u32_e32 v9, s0, v9
	v_accvgpr_write_b32 a7, v26
	v_accvgpr_write_b32 a8, v27
	v_accvgpr_write_b32 a9, v25
	v_mov_b32_e32 v26, s18
	v_mov_b32_e32 v27, s19
	v_mov_b32_e32 v25, s20
	;;#ASMSTART
	
             v_pack_b32_f16 v3, v18, v22 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v4, v18, v22, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v5, v19, v23 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v6, v19, v23, op_sel:[1, 1] 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v7, v20, v24 
             
	;;#ASMEND
	;;#ASMSTART
	
             v_pack_b32_f16 v8, v20, v24, op_sel:[1, 1] 
             
	;;#ASMEND
	ds_write2_b32 v9, v1, v2 offset1:4
	ds_write2_b32 v9, v3, v4 offset0:8 offset1:12
	ds_write2_b32 v9, v5, v6 offset0:16 offset1:20
	ds_write2_b32 v9, v7, v8 offset0:24 offset1:28
	v_add_lshl_u32 v9, v41, v40, 1
	v_lshlrev_b32_e32 v1, 1, v32
	v_add_u32_e32 v17, s0, v9
	s_movk_i32 s0, 0x1100
	v_accvgpr_write_b32 a10, v26
	v_accvgpr_write_b32 a11, v27
	v_accvgpr_write_b32 a12, v25
	v_mov_b32_e32 v26, s21
	v_mov_b32_e32 v27, s22
	v_mov_b32_e32 v25, s23
	v_add_u32_e32 v5, s0, v1
	v_add_u32_e32 v13, 0x4c80, v9
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[5:8], v5 offset1:1
	ds_read2_b64 v[9:12], v17 offset1:1
	ds_read2_b64 v[13:16], v13 offset1:1
	v_accvgpr_write_b32 a13, v26
	v_accvgpr_write_b32 a14, v27
	v_accvgpr_write_b32 a15, v25
	v_add_u32_e32 v18, 64, v35
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[9:10], a[0:15]
	v_ashrrev_i32_e32 v19, 31, v18
	s_movk_i32 s1, 0x80
	v_lshrrev_b32_e32 v19, 29, v19
	v_add_u32_e32 v19, v18, v19
	v_ashrrev_i32_e32 v20, 3, v19
	v_sub_u32_e32 v20, v20, v38
	v_mul_lo_u32 v20, v20, s44
	v_and_b32_e32 v19, 0xffffff8, v19
	v_sub_u32_e32 v18, v18, v19
	v_sub_u32_e32 v18, v18, v39
	v_lshl_add_u32 v18, v18, 3, v20
	v_lshl_add_u32 v17, v18, 1, v17
	v_add_u32_e32 v21, s43, v17
	ds_read2_b64 v[17:20], v17 offset1:1
	ds_read2_b64 v[21:24], v21 offset1:1
	v_mov_b32_e32 v49, 0
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[11:12], a[16:31]
	v_cmp_gt_u32_e32 vcc, s1, v0
	v_mov_b32_e32 v50, 0
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[112:127], v[5:6], v[13:14], a[16:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[17:18], a[0:15]
	v_add_u32_e32 v1, 64, v36
	v_lshrrev_b32_e32 v1, 3, v1
	v_sub_u32_e32 v1, v1, v37
	v_mul_lo_u32 v1, v1, s44
	v_and_b32_e32 v2, 7, v36
	v_sub_u32_e32 v2, v2, v31
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[19:20], a[16:31]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[32:47], v[5:6], v[21:22], a[16:31]
	v_lshl_add_u32 v5, v2, 3, v32
	v_add_lshl_u32 v1, v5, v1, 1
	v_add_u32_e32 v6, s0, v1
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[25:28], v6 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[1:2], v[9:10], a[0:15]
	v_add_u32_e32 v1, s1, v36
	v_lshrrev_b32_e32 v1, 3, v1
	v_sub_u32_e32 v1, v1, v37
	v_mul_lo_u32 v1, v1, s44
	v_add_lshl_u32 v1, v5, v1, 1
	v_add_u32_e32 v6, s0, v1
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[19:20], a[48:63]
	v_mfma_f32_32x32x8f16 a[16:31], v[3:4], v[11:12], a[16:31]
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[29:32], v6 offset1:1
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_32x32x8f16 a[80:95], v[25:26], v[21:22], a[48:63]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[48:63], v[1:2], v[9:10], a[0:15]
	v_mfma_f32_32x32x8f16 a[64:79], v[1:2], v[17:18], a[0:15]
	v_add_u32_e32 v1, 0xc0, v36
	v_lshrrev_b32_e32 v1, 3, v1
	v_sub_u32_e32 v1, v1, v37
	v_mul_lo_u32 v1, v1, s44
	v_add_lshl_u32 v1, v5, v1, 1
	v_add_u32_e32 v5, s0, v1
	v_mfma_f32_32x32x8f16 a[48:63], v[3:4], v[11:12], a[48:63]
	v_mfma_f32_32x32x8f16 a[64:79], v[3:4], v[19:20], a[64:79]
	ds_read2_b64 v[1:4], v1 offset1:1
	ds_read2_b64 v[36:39], v5 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8f16 a[96:111], v[1:2], v[9:10], a[0:15]
	v_mfma_f32_32x32x8f16 a[0:15], v[1:2], v[17:18], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[25:26], v[13:14], a[16:31]
	v_mfma_f32_32x32x8f16 a[0:15], v[3:4], v[19:20], a[0:15]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8f16 a[0:15], v[36:37], v[21:22], a[0:15]
	v_mfma_f32_32x32x8f16 a[16:31], v[27:28], v[15:16], a[16:31]
	v_mfma_f32_32x32x8f16 a[96:111], v[3:4], v[11:12], a[96:111]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v3, a16              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:68 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a17              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:72 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a18              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:76 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a19              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[0:15], v[38:39], v[23:24], a[0:15]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:80 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a20              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:84 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a21              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:88 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a22              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:92 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a23              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[48:63], v[29:30], v[13:14], a[48:63]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:96 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a24              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:100 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a25              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:104 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a26              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:108 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a27              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[64:79], v[29:30], v[21:22], a[64:79]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:112 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a28              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:116 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a29              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:120 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a30              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:124 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a31              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[96:111], v[36:37], v[13:14], a[96:111]
	v_mul_i32_i24_e32 v36, s41, v33
	buffer_store_dword v3, off, s[56:59], 0 offset:128 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a0               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:4 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a1               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:8 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a2               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:12 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a3               ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[48:63], v[31:32], v[15:16], a[48:63]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:16 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a4               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:20 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a5               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:24 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a6               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:28 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a7               ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[16:31], v[7:8], v[23:24], a[32:47]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:32 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a8               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:36 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a9               ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:40 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a10              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:44 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a11              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[32:47], v[31:32], v[23:24], a[64:79]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:48 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a12              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:52 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a13              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:56 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a14              ;  Reload Reuse
	s_nop 1
	buffer_store_dword v3, off, s[56:59], 0 offset:60 ; 4-byte Folded Spill
	v_accvgpr_read_b32 v3, a15              ;  Reload Reuse
	v_mfma_f32_32x32x8f16 a[80:95], v[27:28], v[23:24], a[80:95]
	s_nop 0
	buffer_store_dword v3, off, s[56:59], 0 offset:64 ; 4-byte Folded Spill
	v_mfma_f32_32x32x8f16 a[0:15], v[7:8], v[15:16], a[112:127]
	v_mfma_f32_32x32x8f16 a[64:79], v[38:39], v[15:16], a[96:111]
	s_nop 7
	s_nop 7
	s_nop 0
	v_accvgpr_read_b32 v32, a15
	v_accvgpr_read_b32 v31, a14
	v_accvgpr_read_b32 v30, a13
	v_accvgpr_read_b32 v29, a12
	v_accvgpr_read_b32 v28, a11
	v_accvgpr_read_b32 v27, a10
	v_accvgpr_read_b32 v26, a9
	v_accvgpr_read_b32 v25, a8
	v_accvgpr_read_b32 v24, a7
	v_accvgpr_read_b32 v23, a6
	v_accvgpr_read_b32 v22, a5
	v_accvgpr_read_b32 v21, a4
	v_accvgpr_read_b32 v20, a3
	v_accvgpr_read_b32 v19, a2
	v_accvgpr_read_b32 v18, a1
	v_accvgpr_read_b32 v17, a0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_2
; %bb.1:
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_i32_i24_e32 v2, -4, v1
	v_add_u32_e32 v1, v36, v1
	v_lshlrev_b32_e32 v3, 1, v1
	v_add_u32_e32 v4, s40, v33
	v_lshl_add_u32 v3, v4, 8, v3
	v_mul_lo_u32 v3, v3, s6
	v_add_lshl_u32 v2, v2, v0, 4
	v_lshlrev_b32_e32 v4, 12, v33
	v_lshlrev_b32_e32 v1, 7, v1
	v_add3_u32 v50, v2, v4, v1
	v_add3_u32 v49, s39, v2, v3
BB3_2:                                  ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EEC2ERSO_RKNSA_IJiiiiEEES18_S1D_RKS3_.exit.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v0, v0, 4, v34
	v_lshlrev_b32_e32 v33, 5, v33
	v_lshrrev_b32_e32 v34, 6, v35
	v_add3_u32 v0, v0, v36, v33
	v_sub_u32_e32 v0, v0, v34
	v_lshlrev_b32_e32 v0, 6, v0
	v_cvt_f16_f32_e32 v17, v17
	v_add_lshl_u32 v51, v0, v35, 1
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v18, v19
	v_cvt_f16_f32_e32 v19, v20
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v17
	ds_write_b16 v51, v0 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v17, v23
	v_cvt_f16_f32_e32 v18, v22
	v_cvt_f16_f32_e32 v19, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v17, v26
	v_cvt_f16_f32_e32 v18, v27
	s_load_dword s2, s[4:5], 0x1b0
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v17, v31
	v_accvgpr_read_b32 v1, a16
	v_cvt_f16_f32_e32 v18, v30
	v_accvgpr_read_b32 v2, a17
	v_accvgpr_read_b32 v3, a18
	v_accvgpr_read_b32 v4, a19
	v_accvgpr_read_b32 v5, a20
	v_accvgpr_read_b32 v6, a21
	v_accvgpr_read_b32 v7, a22
	v_accvgpr_read_b32 v8, a23
	v_accvgpr_read_b32 v9, a24
	v_accvgpr_read_b32 v10, a25
	v_accvgpr_read_b32 v11, a26
	v_accvgpr_read_b32 v12, a27
	v_accvgpr_read_b32 v13, a28
	v_accvgpr_read_b32 v14, a29
	v_accvgpr_read_b32 v15, a30
	v_accvgpr_read_b32 v16, a31
	v_cvt_f16_f32_e32 v19, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_4
; %bb.3:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i.i.i.i
	v_lshlrev_b32_e32 v0, 1, v50
	ds_read2_b32 v[17:18], v0 offset0:2 offset1:3
	ds_read2_b32 v[19:20], v0 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v49
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v20, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 12 offen
	ds_read2_b32 v[17:18], v0 offset0:6 offset1:7
	ds_read2_b32 v[19:20], v0 offset0:4 offset1:5
	v_add_u32_e32 v21, 8, v49
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v20, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 12 offen
	ds_read2_b32 v[17:18], v0 offset0:38 offset1:39
	ds_read2_b32 v[19:20], v0 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v20, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 12 offen
	ds_read2_b32 v[17:18], v0 offset0:34 offset1:35
	ds_read2_b32 v[19:20], v0 offset0:32 offset1:33
	v_add_lshl_u32 v0, v49, s6, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v19, v0, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v20, v0, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v17, v0, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v18, v0, s[24:27], 12 offen
BB3_4:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v1
	v_cvt_f16_f32_e32 v1, v2
	v_cvt_f16_f32_e32 v2, v3
	v_cvt_f16_f32_e32 v3, v4
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v7
	v_cvt_f16_f32_e32 v2, v6
	v_cvt_f16_f32_e32 v3, v5
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v9
	v_cvt_f16_f32_e32 v1, v10
	v_cvt_f16_f32_e32 v2, v11
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v15
	v_accvgpr_read_b32 v17, a80
	v_cvt_f16_f32_e32 v2, v14
	v_accvgpr_read_b32 v18, a81
	v_accvgpr_read_b32 v19, a82
	v_accvgpr_read_b32 v20, a83
	v_accvgpr_read_b32 v21, a84
	v_accvgpr_read_b32 v22, a85
	v_accvgpr_read_b32 v23, a86
	v_accvgpr_read_b32 v24, a87
	v_accvgpr_read_b32 v25, a88
	v_accvgpr_read_b32 v26, a89
	v_accvgpr_read_b32 v27, a90
	v_accvgpr_read_b32 v28, a91
	v_accvgpr_read_b32 v29, a92
	v_accvgpr_read_b32 v30, a93
	v_accvgpr_read_b32 v31, a94
	v_accvgpr_read_b32 v32, a95
	v_cvt_f16_f32_e32 v3, v13
	s_mul_i32 s3, s6, 63
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_6
; %bb.5:                                ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	v_add_u32_e32 v6, s6, v49
	v_add_u32_e32 v49, s3, v6
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_lshlrev_b32_e32 v4, 1, v6
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 12 offen
BB3_6:                                  ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v4, off, s[56:59], 0 offset:68 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v0, v17
	v_cvt_f16_f32_e32 v1, v18
	v_cvt_f16_f32_e32 v2, v19
	v_cvt_f16_f32_e32 v3, v20
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:84 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:88 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:92 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:96 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:100 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:104 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v4              ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:108 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:116 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:124 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v4             ;  Reload Reuse
	buffer_load_dword v4, off, s[56:59], 0 offset:128 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v23
	v_cvt_f16_f32_e32 v2, v22
	v_cvt_f16_f32_e32 v3, v21
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v25
	v_cvt_f16_f32_e32 v1, v26
	v_cvt_f16_f32_e32 v2, v27
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v32
	v_cvt_f16_f32_e32 v1, v31
	v_cvt_f16_f32_e32 v2, v30
	v_cvt_f16_f32_e32 v3, v29
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v4             ;  Reload Reuse
	s_nop 2
	v_accvgpr_read_b32 v48, a15
	v_accvgpr_read_b32 v47, a14
	v_accvgpr_read_b32 v46, a13
	v_accvgpr_read_b32 v45, a12
	v_accvgpr_read_b32 v44, a11
	v_accvgpr_read_b32 v43, a10
	v_accvgpr_read_b32 v42, a9
	v_accvgpr_read_b32 v41, a8
	v_accvgpr_read_b32 v40, a7
	v_accvgpr_read_b32 v39, a6
	v_accvgpr_read_b32 v38, a5
	v_accvgpr_read_b32 v37, a4
	v_accvgpr_read_b32 v36, a3
	v_accvgpr_read_b32 v35, a2
	v_accvgpr_read_b32 v34, a1
	v_accvgpr_read_b32 v33, a0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_8
; %bb.7:                                ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i161.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s6, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 12 offen
BB3_8:                                  ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i254.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v16, v33
	v_cvt_f16_f32_e32 v17, v34
	v_cvt_f16_f32_e32 v18, v35
	v_cvt_f16_f32_e32 v19, v36
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v40
	v_cvt_f16_f32_e32 v17, v39
	v_cvt_f16_f32_e32 v18, v38
	v_cvt_f16_f32_e32 v19, v37
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v41
	v_cvt_f16_f32_e32 v17, v42
	v_cvt_f16_f32_e32 v18, v43
	v_cvt_f16_f32_e32 v19, v44
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v48
	v_cvt_f16_f32_e32 v17, v47
	v_accvgpr_read_b32 v0, a48
	v_cvt_f16_f32_e32 v18, v46
	v_accvgpr_read_b32 v1, a49
	v_accvgpr_read_b32 v2, a50
	v_accvgpr_read_b32 v3, a51
	v_accvgpr_read_b32 v4, a52
	v_accvgpr_read_b32 v5, a53
	v_accvgpr_read_b32 v6, a54
	v_accvgpr_read_b32 v7, a55
	v_accvgpr_read_b32 v8, a56
	v_accvgpr_read_b32 v9, a57
	v_accvgpr_read_b32 v10, a58
	v_accvgpr_read_b32 v11, a59
	v_accvgpr_read_b32 v12, a60
	v_accvgpr_read_b32 v13, a61
	v_accvgpr_read_b32 v14, a62
	v_accvgpr_read_b32 v15, a63
	v_cvt_f16_f32_e32 v19, v45
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_10
; %bb.9:                                ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i318.i.i.i
	v_lshlrev_b32_e32 v20, 1, v50
	ds_read2_b32 v[16:17], v20 offset0:2 offset1:3
	ds_read2_b32 v[18:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v49
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:6 offset1:7
	ds_read2_b32 v[18:19], v20 offset0:4 offset1:5
	v_add_u32_e32 v21, 8, v49
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:38 offset1:39
	ds_read2_b32 v[18:19], v20 offset0:36 offset1:37
	v_add_u32_e32 v22, s6, v49
	v_add_u32_e32 v49, s3, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:34 offset1:35
	ds_read2_b32 v[18:19], v20 offset0:32 offset1:33
	v_lshlrev_b32_e32 v20, 1, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 12 offen
BB3_10:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I343.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a32
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a33
	v_accvgpr_read_b32 v18, a34
	v_accvgpr_read_b32 v19, a35
	v_accvgpr_read_b32 v20, a36
	v_accvgpr_read_b32 v21, a37
	v_accvgpr_read_b32 v22, a38
	v_accvgpr_read_b32 v23, a39
	v_accvgpr_read_b32 v24, a40
	v_accvgpr_read_b32 v25, a41
	v_accvgpr_read_b32 v26, a42
	v_accvgpr_read_b32 v27, a43
	v_accvgpr_read_b32 v28, a44
	v_accvgpr_read_b32 v29, a45
	v_accvgpr_read_b32 v30, a46
	v_accvgpr_read_b32 v31, a47
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_12
; %bb.11:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i504.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s6, 1
	v_add_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 12 offen
BB3_12:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i597.i.i.i
	s_or_b64 exec, exec, s[0:1]
	buffer_load_dword v32, off, s[56:59], 0 offset:4 ; 4-byte Folded Reload
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v19, v19
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a0, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a1, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a2, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a3, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a4, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a5, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a6, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a7, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a8, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a9, v32             ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a10, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a11, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a12, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a13, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a14, v32            ;  Reload Reuse
	buffer_load_dword v32, off, s[56:59], 0 offset:64 ; 4-byte Folded Reload
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v16
	ds_write_b16 v51, v17 offset:128
	ds_write_b16 v51, v18 offset:256
	ds_write_b16 v51, v19 offset:384
	v_cvt_f16_f32_e32 v16, v23
	v_cvt_f16_f32_e32 v17, v22
	v_cvt_f16_f32_e32 v18, v21
	v_cvt_f16_f32_e32 v19, v20
	ds_write_b16 v51, v16 offset:1408
	ds_write_b16 v51, v17 offset:1280
	ds_write_b16 v51, v18 offset:1152
	ds_write_b16 v51, v19 offset:1024
	v_cvt_f16_f32_e32 v16, v24
	v_cvt_f16_f32_e32 v17, v25
	v_cvt_f16_f32_e32 v18, v26
	v_cvt_f16_f32_e32 v19, v27
	ds_write_b16 v51, v16 offset:2048
	ds_write_b16 v51, v17 offset:2176
	ds_write_b16 v51, v18 offset:2304
	ds_write_b16 v51, v19 offset:2432
	v_cvt_f16_f32_e32 v16, v31
	v_cvt_f16_f32_e32 v17, v30
	v_cvt_f16_f32_e32 v18, v29
	v_cvt_f16_f32_e32 v19, v28
	ds_write_b16 v51, v16 offset:3456
	ds_write_b16 v51, v17 offset:3328
	ds_write_b16 v51, v18 offset:3200
	ds_write_b16 v51, v19 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_accvgpr_write_b32 a15, v32            ;  Reload Reuse
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_14
; %bb.13:                               ; %_ZNK2ck10static_forILi0ELi2ELi1EEclIZZNS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS6_IJiiiEEELb0EEENS7_INS6_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESF_NS_23Merge_v2_magic_divisionINS6_IJiiEEEEESF_NSE_ISB_EENS7_ISH_Lb0EEESF_SJ_EEENS6_IJNS_8SequenceIJLi0EEEENSM_IJLi1EEEENSM_IJLi2EEEENSM_IJLi3EEEENSM_IJLi4ELi6EEEENSM_IJLi7EEEENSM_IJLi5EEEENSM_IJLi8EEEENSM_IJLi9EEEENSM_IJLi10EEEEEEENS6_IJNSM_IJLi1ELi2ELi3EEEENSM_IJLi4ELi5EEEENSM_IJLi6EEEESS_SU_SV_SW_NSM_IJLi11ELi12EEEENSM_IJLi13EEEENSM_IJLi14EEEEEEENSM_IJLi11ELi12ELi13ELi14EEEEiEENS5_INS6_IJNS7_INS6_IJiiiiEEELb0EEESF_NS_3PadIiiiLb0EEES1A_SF_SF_NS_5EmbedISH_SH_Lb0EEES1C_SF_SD_SF_SF_SF_SF_SF_NSG_IS8_EES1D_SJ_SK_SF_SJ_EEENS6_IJSN_SO_SP_SQ_NSM_IJLi4EEEEST_S10_SS_SU_SV_SW_NSM_IJLi11EEEENSM_IJLi12EEEES12_S13_NSM_IJLi15ELi18ELi20EEEENSM_IJLi17ELi19ELi21EEEENSM_IJLi16EEEENSM_IJLi22EEEENSM_IJLi23EEEENSM_IJLi24EEEEEEENS6_IJNSM_IJLi1ELi2ELi3ELi4EEEEST_.i661.i.i.i
	v_lshlrev_b32_e32 v20, 1, v50
	ds_read2_b32 v[16:17], v20 offset0:2 offset1:3
	ds_read2_b32 v[18:19], v20 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v21, 1, v49
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:6 offset1:7
	ds_read2_b32 v[18:19], v20 offset0:4 offset1:5
	v_add_u32_e32 v21, 8, v49
	v_lshlrev_b32_e32 v22, 1, v21
	v_add_lshl_u32 v21, v21, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v22, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v22, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v22, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v22, s[24:27], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:38 offset1:39
	ds_read2_b32 v[18:19], v20 offset0:36 offset1:37
	v_add_u32_e32 v22, s6, v49
	v_add_u32_e32 v49, s3, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v21, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v21, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v21, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v21, s[24:27], 12 offen
	ds_read2_b32 v[16:17], v20 offset0:34 offset1:35
	ds_read2_b32 v[18:19], v20 offset0:32 offset1:33
	v_lshlrev_b32_e32 v20, 1, v22
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v18, v20, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v19, v20, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v16, v20, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v17, v20, s[24:27], 12 offen
BB3_14:                                 ; %_ZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_I686.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v1, v1
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v3, v3
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v7
	v_cvt_f16_f32_e32 v1, v6
	v_cvt_f16_f32_e32 v2, v5
	v_cvt_f16_f32_e32 v3, v4
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v1, v9
	v_cvt_f16_f32_e32 v2, v10
	v_cvt_f16_f32_e32 v3, v11
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v15
	v_cvt_f16_f32_e32 v1, v14
	v_accvgpr_read_b32 v16, a64
	v_cvt_f16_f32_e32 v2, v13
	v_accvgpr_read_b32 v17, a65
	v_accvgpr_read_b32 v18, a66
	v_accvgpr_read_b32 v19, a67
	v_accvgpr_read_b32 v20, a68
	v_accvgpr_read_b32 v21, a69
	v_accvgpr_read_b32 v22, a70
	v_accvgpr_read_b32 v23, a71
	v_accvgpr_read_b32 v24, a72
	v_accvgpr_read_b32 v25, a73
	v_accvgpr_read_b32 v26, a74
	v_accvgpr_read_b32 v27, a75
	v_accvgpr_read_b32 v28, a76
	v_accvgpr_read_b32 v29, a77
	v_accvgpr_read_b32 v30, a78
	v_accvgpr_read_b32 v31, a79
	v_cvt_f16_f32_e32 v3, v12
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_16
; %bb.15:                               ; %_ZN2ck33BlockwiseTensorSliceTransfer_v6r1ILi256ENS_16tensor_operation12element_wise11PassThroughELNS_25InMemoryDataOperationEnumE1ENS_8SequenceIJLi1ELi64ELi1ELi64EEEENS5_IJLi1ELi32ELi1ELi4EEEENS5_IJLi0ELi1ELi2ELi3EEEEDF16_DF16_KNS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINSA_IJNS_17integral_constantIiLi1EEENSC_IiLi64EEESD_SE_EEELb0EEEEEENSA_IJNS5_IJLi0EEEEEEENSA_IJNS5_IJLi1ELi2ELi3ELi4EEEEEEESK_NSC_IiLi4096EEEEERKNS9_INSA_IJNSB_INSA_IJiiEEELb0EEENSB_INSA_IJiNSC_IiLi256EEEEEELb0EEENSB_INSA_IJiNSC_IiLi128EEEEEELb0EEEEEENSA_IJSI_NS5_IJLi1EEEENS5_IJLi2EEEEEEENSA_IJNS5_IJLi1ELi2EEEENS5_IJLi3ELi4EEEENS5_IJLi5ELi6EEEEEEENS5_IJLi3ELi4ELi5ELi6EEEEiEES8_Li3ELi8ELb1ELb0EE3RunINS_13DynamicBufferILNS_16AddressSpaceEnumE2EDF16_SM_Lb1EEENS1B_ILS1C_1EDF16_iLb1EEEEEvRSO_RKT_S18_RT0_.exit.i.i.i.i847.i.i.i
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s6, 1
	v_subrev_u32_e32 v49, 64, v49
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 12 offen
BB3_16:                                 ; %_ZZZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_.i.i.i940.i.i.i
	s_or_b64 exec, exec, s[0:1]
	v_cvt_f16_f32_e32 v0, v16
	v_cvt_f16_f32_e32 v1, v17
	v_cvt_f16_f32_e32 v2, v18
	v_cvt_f16_f32_e32 v3, v19
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	ds_write_b16 v51, v0
	ds_write_b16 v51, v1 offset:128
	ds_write_b16 v51, v2 offset:256
	ds_write_b16 v51, v3 offset:384
	v_cvt_f16_f32_e32 v0, v23
	v_cvt_f16_f32_e32 v1, v22
	v_cvt_f16_f32_e32 v2, v21
	v_cvt_f16_f32_e32 v3, v20
	ds_write_b16 v51, v0 offset:1408
	ds_write_b16 v51, v1 offset:1280
	ds_write_b16 v51, v2 offset:1152
	ds_write_b16 v51, v3 offset:1024
	v_cvt_f16_f32_e32 v0, v24
	v_cvt_f16_f32_e32 v1, v25
	v_cvt_f16_f32_e32 v2, v26
	v_cvt_f16_f32_e32 v3, v27
	ds_write_b16 v51, v0 offset:2048
	ds_write_b16 v51, v1 offset:2176
	ds_write_b16 v51, v2 offset:2304
	ds_write_b16 v51, v3 offset:2432
	v_cvt_f16_f32_e32 v0, v31
	v_cvt_f16_f32_e32 v1, v30
	v_cvt_f16_f32_e32 v2, v29
	v_cvt_f16_f32_e32 v3, v28
	ds_write_b16 v51, v0 offset:3456
	ds_write_b16 v51, v1 offset:3328
	ds_write_b16 v51, v2 offset:3200
	ds_write_b16 v51, v3 offset:3072
	;;#ASMSTART
	    s_waitcnt lgkmcnt(0) 
     s_barrier     
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz BB3_18
; %bb.17:
	v_lshlrev_b32_e32 v4, 1, v50
	ds_read2_b32 v[0:1], v4 offset0:2 offset1:3
	ds_read2_b32 v[2:3], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s26, s2, 1
	s_mov_b32 s27, 0x20000
	v_lshlrev_b32_e32 v5, 1, v49
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:6 offset1:7
	ds_read2_b32 v[2:3], v4 offset0:4 offset1:5
	v_add_u32_e32 v5, 8, v49
	v_lshlrev_b32_e32 v6, 1, v5
	v_add_lshl_u32 v5, v5, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v6, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v6, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v6, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v6, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:38 offset1:39
	ds_read2_b32 v[2:3], v4 offset0:36 offset1:37
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v5, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v5, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v5, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v5, s[24:27], 12 offen
	ds_read2_b32 v[0:1], v4 offset0:34 offset1:35
	ds_read2_b32 v[2:3], v4 offset0:32 offset1:33
	v_add_lshl_u32 v4, v49, s6, 1
	s_waitcnt lgkmcnt(0)
	buffer_atomic_pk_add_f16 v2, v4, s[24:27], 0 offen
	buffer_atomic_pk_add_f16 v3, v4, s[24:27], 4 offen
	buffer_atomic_pk_add_f16 v0, v4, s[24:27], 8 offen
	buffer_atomic_pk_add_f16 v1, v4, s[24:27], 12 offen
BB3_18:                                 ; %_ZN2ck43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS3_IJiiiEEELb0EEENS4_INS3_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESC_NS_23Merge_v2_magic_divisionINS3_IJiiEEEEESC_NSB_IS8_EENS4_ISE_Lb0EEESC_SG_EEENS3_IJNS_8SequenceIJLi0EEEENSJ_IJLi1EEEENSJ_IJLi2EEEENSJ_IJLi3EEEENSJ_IJLi4ELi6EEEENSJ_IJLi7EEEENSJ_IJLi5EEEENSJ_IJLi8EEEENSJ_IJLi9EEEENSJ_IJLi10EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3EEEENSJ_IJLi4ELi5EEEENSJ_IJLi6EEEESP_SR_SS_ST_NSJ_IJLi11ELi12EEEENSJ_IJLi13EEEENSJ_IJLi14EEEEEEENSJ_IJLi11ELi12ELi13ELi14EEEEiEENS2_INS3_IJNS4_INS3_IJiiiiEEELb0EEESC_NS_3PadIiiiLb0EEES17_SC_SC_NS_5EmbedISE_SE_Lb0EEES19_SC_SA_SC_SC_SC_SC_SC_NSD_IS5_EES1A_SG_SH_SC_SG_EEENS3_IJSK_SL_SM_SN_NSJ_IJLi4EEEESQ_SX_SP_SR_SS_ST_NSJ_IJLi11EEEENSJ_IJLi12EEEESZ_S10_NSJ_IJLi15ELi18ELi20EEEENSJ_IJLi17ELi19ELi21EEEENSJ_IJLi16EEEENSJ_IJLi22EEEENSJ_IJLi23EEEENSJ_IJLi24EEEEEEENS3_IJNSJ_IJLi1ELi2ELi3ELi4EEEESQ_SX_SP_SR_SS_NSJ_IJLi10ELi11EEEENSJ_IJ
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6
	.amdhsa_kernel _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
		.amdhsa_group_segment_fixed_size 26112
		.amdhsa_private_segment_fixed_size 132
		.amdhsa_kernarg_size 600
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 60
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
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
	.size	_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_, .Lfunc_end3-_ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 8700
; NumSgprs: 62
; NumVgprs: 52
; NumAgprs: 128
; TotalNumVgprs: 128
; ScratchSize: 132
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 26112 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 128
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.ident	"AMD clang version 14.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-5.0.0 22051 235b6880e2e515507478181ec11a20c1ec87945b)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 26112
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 132
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     128
    .vgpr_spill_count: 32
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 26112
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 132
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb1EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     128
    .vgpr_spill_count: 32
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 26112
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 132
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     128
    .vgpr_spill_count: 32
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           112
        .value_kind:     by_value
      - .offset:         136
        .size:           252
        .value_kind:     by_value
      - .offset:         388
        .size:           48
        .value_kind:     by_value
      - .offset:         436
        .size:           1
        .value_kind:     by_value
      - .offset:         437
        .size:           1
        .value_kind:     by_value
      - .offset:         438
        .size:           1
        .value_kind:     by_value
      - .offset:         440
        .size:           104
        .value_kind:     by_value
      - .offset:         544
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         552
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         560
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .address_space:  global
        .offset:         568
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         576
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         584
        .size:           8
        .value_kind:     hidden_none
      - .address_space:  global
        .offset:         592
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
    .group_segment_fixed_size: 26112
    .kernarg_segment_align: 8
    .kernarg_segment_size: 600
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_
    .private_segment_fixed_size: 132
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         _ZN2ck25kernel_gemm_xdlops_v2r4r2INS_43GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_v2r4r2ILi256EDF16_fDF16_LNS_25InMemoryDataOperationEnumE1ENS_16TensorDescriptorINS_5TupleIJNS_7UnMergeINS4_IJiiiEEELb0EEENS5_INS4_IJiNS_17integral_constantIiLi8EEEEEELb0EEENS_11PassThroughIiEESD_NS_23Merge_v2_magic_divisionINS4_IJiiEEEEESD_NSC_IS9_EENS5_ISF_Lb0EEESD_SH_EEENS4_IJNS_8SequenceIJLi0EEEENSK_IJLi1EEEENSK_IJLi2EEEENSK_IJLi3EEEENSK_IJLi4ELi6EEEENSK_IJLi7EEEENSK_IJLi5EEEENSK_IJLi8EEEENSK_IJLi9EEEENSK_IJLi10EEEEEEENS4_IJNSK_IJLi1ELi2ELi3EEEENSK_IJLi4ELi5EEEENSK_IJLi6EEEESQ_SS_ST_SU_NSK_IJLi11ELi12EEEENSK_IJLi13EEEENSK_IJLi14EEEEEEENSK_IJLi11ELi12ELi13ELi14EEEEiEENS3_INS4_IJNS5_INS4_IJiiiiEEELb0EEESD_NS_3PadIiiiLb0EEES18_SD_SD_NS_5EmbedISF_SF_Lb0EEES1A_SD_SB_SD_SD_SD_SD_SD_NSE_IS6_EES1B_SH_SI_SD_SH_EEENS4_IJSL_SM_SN_SO_NSK_IJLi4EEEESR_SY_SQ_SS_ST_SU_NSK_IJLi11EEEENSK_IJLi12EEEES10_S11_NSK_IJLi15ELi18ELi20EEEENSK_IJLi17ELi19ELi21EEEENSK_IJLi16EEEENSK_IJLi22EEEENSK_IJLi23EEEENSK_IJLi24EEEEEEENS4_IJNSK_IJLi1ELi2ELi3ELi4EEEESR_SY_SQ_SS_ST_NSK_IJLi10ELi11EEEENSK_IJLi12ELi13EEEES11_NSK_IJLi15ELi16EEEENSK_IJLi17EEEENSK_IJLi18EEEENSK_IJLi19EEEENSK_IJLi20EEEENSK_IJLi21EEEES1J_S1K_S1L_NSK_IJLi25ELi26EEEENSK_IJLi27EEEENSK_IJLi28EEEEEEENSK_IJLi25ELi26ELi27ELi28EEEEiEENS3_INS4_IJSI_EEENS4_IJSL_EEENS4_IJNSK_IJLi1ELi2EEEEEEES24_iEENS_16tensor_operation12element_wise11PassThroughES29_S29_Li256ELi128ELi4ELi32ELi32ELi8ELi4ELi2ENSK_IJLi1ELi4ELi32ELi2EEEENSK_IJLi0ELi3ELi1ELi2EEEENSK_IJLi0ELi2ELi1ELi3EEEELi2ELi8ELi4ELb0ELb1ENSK_IJLi1ELi4ELi16ELi4EEEES2B_S2C_Li2ELi8ELi2ELb0ELb1ELi1ELi1ELi8ENSK_IJLi1ELi32ELi1ELi4EEEELb1ELb1EEEDF16_DF16_S14_S21_NS3_INS4_IJSI_NS5_INS4_IJiNS8_IiLi256EEEEEELb0EEENS5_INS4_IJiNS8_IiLi128EEEEEELb0EEEEEENS4_IJSL_SM_SN_EEENS4_IJS24_NSK_IJLi3ELi4EEEENSK_IJLi5ELi6EEEEEEENSK_IJLi3ELi4ELi5ELi6EEEEiEES29_S29_S29_NS_13TensorAdaptorINS4_IJSD_SI_SI_NSE_INS4_IJiiiiiEEEEEEEENS4_IJSL_SM_SN_NSK_IJLi3ELi4ELi5ELi6ELi7EEEEEEENS4_IJSO_SP_NSK_IJLi5ELi7EEEESS_EEENSK_IJLi0ELi1ELi2EEEESS_EELb0EEEvPKT0_S35_PT1_T2_T3_T4_T5_T6_T7_T8_.kd
    .vgpr_count:     128
    .vgpr_spill_count: 32
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
