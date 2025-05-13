# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-
"""
generate kernel instances to speed up compilation
"""

DATA_TYPE_MAP = {'fp32'  : 'float',
                 'fp16'  : 'ck_tile::half_t',
                 'bf16'  : 'ck_tile::bf16_t',
                 'int8'  : 'ck_tile::int8_t',
                 'fp8'   : 'ck_tile::fp8_t',
                 'bf8'   : 'ck_tile::bf8_t',
                 'int4'  : 'ck_tile::pk_int4_t'
                }

LAYOUT_MAP = {'r' : 'ck_tile::tensor_layout::gemm::RowMajor',
              'c' : 'ck_tile::tensor_layout::gemm::ColumnMajor'} 

DEFAULT_EPILOGUE = """
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
                                ck_tile::DefaultGemm2DEpilogueProblem<AccDataType, 
                                                                      CDataType, 
                                                                      CLayout, 
                                                                      pad_m,
                                                                      pad_n,
                                                                      WarpTileM,
                                                                      WarpTileN,
                                                                      WarpTileK,
                                                                      UniversalGemmProblem::TransposeC>>;
"""

CSHUFFLE_EPILOGUE = """
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                            ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             CDataType,
                                                             CLayout,
                                                             GemmPipelineProblem::kBlockSize,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             WarpM,
                                                             WarpN,
                                                             WarpTileM,
                                                             WarpTileN,
                                                             WarpTileK,
                                                             UniversalGemmProblem::TransposeC>>;
"""

HOT_LOOP_FALSE = """
            if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                Run(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                Run(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else
            {
                throw std::runtime_error("Num K loop must be larger than number of prefetech stages.");
            }  
"""

RUN_MEM = """
            if(tail_num == ck_tile::TailNumber::One)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
            }
            else if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }

            if constexpr(BaseGemmPipeline::PrefetchStages > 2)
            {
                if(tail_num == ck_tile::TailNumber::Two)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
                }
        
                if(tail_num == ck_tile::TailNumber::Three)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
                }
                if(tail_num == ck_tile::TailNumber::Four)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
                }
                if(tail_num == ck_tile::TailNumber::Five)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
                }
                if(tail_num == ck_tile::TailNumber::Six)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
                }
                if(tail_num == ck_tile::TailNumber::Seven)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
                }
                throw std::runtime_error("The tile number is wrong! It should not exceed the prefetch stage numbers");
            }
"""

RUN_COMPV3 = """
            if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
            }
            else
            {
                throw std::runtime_error("The tail number is wrong. It should be Full, Odd, or Even.");
            }
"""

RUN_COMPV4 = """
            if(tail_num == ck_tile::TailNumber::Three)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
            else
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
"""

PIPELINE_MAP = {'mem' : ['ck_tile::BaseGemmPipelineAgBgCrMem', 'ck_tile::GemmPipelineAgBgCrMem'],
                'compv3' : ['ck_tile::BaseGemmPipelineAgBgCrCompV3', 'ck_tile::GemmPipelineAgBgCrCompV3'],
                'compv4' : ['ck_tile::BaseGemmPipelineAgBgCrCompV4', 'ck_tile::GemmPipelineAgBgCrCompV4']}

SCHEDULER_MAP = {'interwave' : 'ck_tile::GemmPipelineScheduler::Interwave',
                 'intrawave' : 'ck_tile::GemmPipelineScheduler::Intrawave'}

EPILOGUE_MAP = {'default' :DEFAULT_EPILOGUE,
                'cshuffle' : CSHUFFLE_EPILOGUE}      

HOT_LOOP_TRUE = {'mem' : RUN_MEM,
                 'compv3' : RUN_COMPV3,
                 'compv4' : RUN_COMPV4}  

BOOL_MAP = lambda b_: {True: 'true', False: 'false'}[bool(b_)]