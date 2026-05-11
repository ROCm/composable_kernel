	s_addc_u32 s5, s13, s5                                     // 000000002650: 8205050D
	s_add_i32 s17, s17, 32                                     // 000000002654: 8111A011
	s_add_i32 s10, s10, s15                                    // 000000002658: 810A0F0A
	s_add_u32 s0, s0, 64                                       // 00000000265C: 8000C000
	v_mul_lo_u32 v13, v13, s8                                  // 000000002660: D285000D 0000110D
	s_addc_u32 s1, s1, 0                                       // 000000002668: 82018001
	s_mov_b32 s7, s3                                           // 00000000266C: BE870003
	v_add_lshl_u32 v13, v13, v4, 1                             // 000000002670: D1FE000D 0206090D
	s_cmp_lt_i32 s17, s9                                       // 000000002678: BF040911
	s_waitcnt vmcnt(6)                                         // 00000000267C: BF8C0F76
	v_perm_b32 v15, v15, v14, s16                              // 000000002680: D1ED000F 00421D0F
	s_waitcnt vmcnt(3)                                         // 000000002688: BF8C0F73
	v_perm_b32 v17, v18, v17, s16                              // 00000000268C: D1ED0011 00422312
	s_waitcnt vmcnt(2)                                         // 000000002694: BF8C0F72
	v_perm_b32 v16, v19, v16, s16                              // 000000002698: D1ED0010 00422113
	s_waitcnt vmcnt(0)                                         // 0000000026A0: BF8C0F70
	v_perm_b32 v14, v21, v20, s16                              // 0000000026A4: D1ED000E 00422915
	ds_write_b128 v12, v[14:17]                                // 0000000026AC: D9BE0000 00000E0C
	s_waitcnt lgkmcnt(0)                                       // 0000000026B4: BF8CC07F
	s_barrier                                                  // 0000000026B8: BF8A0000
	ds_read_u16 v14, v28                                       // 0000000026BC: D8780000 0E00001C
	ds_read_u16 v15, v27                                       // 0000000026C4: D8780000 0F00001B
	ds_read_u16 v16, v24                                       // 0000000026CC: D8780000 10000018
	ds_read_u16 v17, v25                                       // 0000000026D4: D8780000 11000019
	ds_read_u16 v18, v29 offset:128                            // 0000000026DC: D8780080 1200001D
	ds_read_u16 v19, v23                                       // 0000000026E4: D8780000 13000017
	ds_read_u16 v20, v26 offset:128                            // 0000000026EC: D8780080 1400001A
	ds_read_u16 v21, v22 offset:256                            // 0000000026F4: D8780100 15000016
	s_waitcnt lgkmcnt(0)                                       // 0000000026FC: BF8CC07F
	s_barrier                                                  // 000000002700: BF8A0000
	buffer_store_short v14, v13, s[4:7], 0 offen               // 000000002704: E0681000 80010E0D
	buffer_store_short v15, v13, s[4:7], 0 offen offset:2      // 00000000270C: E0681002 80010F0D
	buffer_store_short v16, v13, s[4:7], 0 offen offset:4      // 000000002714: E0681004 8001100D
	buffer_store_short v17, v13, s[4:7], 0 offen offset:6      // 00000000271C: E0681006 8001110D
	buffer_store_short v18, v13, s[4:7], 0 offen offset:8      // 000000002724: E0681008 8001120D
	buffer_store_short v19, v13, s[4:7], 0 offen offset:10     // 00000000272C: E068100A 8001130D
	buffer_store_short v20, v13, s[4:7], 0 offen offset:12     // 000000002734: E068100C 8001140D
	buffer_store_short v21, v13, s[4:7], 0 offen offset:14     // 00000000273C: E068100E 8001150D
	s_waitcnt lgkmcnt(0)                                       // 000000002744: BF8CC07F
	s_barrier                                                  // 000000002748: BF8A0000
	s_cbranch_scc1 65391                                       // 00000000274C: BF85FF6F <_ZN7ck_tile6kentryILi256E25ProductionTransposeKernelIDF16_Lb1EEJPKDF16_PDF16_iiEEEvDpT1_+0x10c>
	s_endpgm                                                   // 000000002750: BF810000
