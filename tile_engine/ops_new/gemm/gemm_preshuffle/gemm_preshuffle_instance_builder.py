# class GemmPreshuffleKernelBuilder(BaseGemmKernelBuilder):
#     def _get_kernel_prefix(self):
#         return "gemm_preshuffle"

#     def _get_pipeline_maps(self):
#         # Return preshuffle-specific pipeline maps

#     def _generate_kernel_instance(self, tile_config, trait_combo, k_block_per_cu, permute_n, is_header=True):
#         # Preshuffle-specific kernel generation logic

#     def _get_file_names(self):
#         return "gemm_preshuffle_kernel_count.txt", "gemm_preshuffle_kernel_list.txt"

#     def _get_cmake_function_name(self):
#         return "create_individual_gemm_preshuffle_target"

#     def _validate_layout_constraints(self, layout_parts):
#         # Only allow rcr layout for preshuffle

#     def _get_additional_validation_constraints(self, tile_config):
#         # Preshuffle-specific validation (permute_n constraint)
