# class GemmKernelBuilder:
#     def __init__(self, working_path, gpu_target, datatype, layout, config_json=None)

#     # Common methods
#     def write_kernel_list(self)
#     def _get_tile_configs(self, fast_mode=False)
#     def _generate_values(self, min_val, max_val, step)
#     def _generate_trait_combinations(self)
#     def _validate_tile_config(self, ..., fast_mode=False)
#     def run(self, num_workers=None)
#     def generate_individual(self, num_workers=None)
#     def _generate_cmake_individual_targets(self, kernel_list)

#     # Abstract methods (to be implemented by derived classes)
#     def _get_kernel_prefix(self):
#         """Return kernel name prefix (e.g., 'gemm', 'gemm_preshuffle')"""
#         pass

#     def _get_pipeline_maps(self):
#         """Return pipeline implementation and base pipeline maps"""
#         pass

#     def _generate_kernel_instance(self, tile_config, trait_combo, k_block_per_cu, **kwargs):
#         """Generate kernel instance code"""
#         pass

#     def _get_file_names(self):
#         """Return file names for kernel list and count files"""
#         pass

#     def _get_cmake_function_name(self):
#         """Return CMake function name for individual targets"""
#         pass

#     def _validate_layout_constraints(self, layout_parts):
#         """Validate layout-specific constraints"""
#         pass

#     def _get_additional_validation_constraints(self, tile_config):
#         """Additional validation constraints specific to kernel type"""
#         return True
