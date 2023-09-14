import enum
import os.path
import shutil
import functools
import operator
import collections
import subprocess
import re
import gemm_op
from gemm_op import *
import user

# function to substitute values into template
def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text

# setting up the template with all the user input
class EmitGemmInstance:
    def __init__(self):
        self.gemm_op_template =     """

DeviceGemmMultipleD_Xdl_CShuffle<${layout_a}, ${layout_b}, ${layout_ds}, ${layout_e}, ${type_a}, ${type_b}, ${type_acc}, ${type_cshuffle}, ${type_ds}, ${type_e}, ${elementwise_op_a}, ${elementwise_op_b}, ${elementwise_op_cde}, ${Gemm_spec}, ${num_gemmk_prefetch_stage}, ${block_size}, ${mperblock}, ${nperblock}, ${kperblock}, ${ak1}, ${bk1}, ${mperXDL}, ${nperXDL}, ${mXdlperwave}, ${nXdlperwave}, ${ABT_thread_cluster_lengths_K0_M_K1}, ${ABT_thread_cluster_arrange_order}, ${ABT_src_access_order}, ${ABT_src_vec_dim}, ${ABT_src_scalar_per_vec}, ${ABT_dst_scalar_per_vec_k1}, ${ABT_lds_add_extra_m}, ${BBT_thread_cluster_lengths_K0_N_K1}, ${BBT_thread_cluster_arrange_order}, ${BBT_src_access_order}, ${BBT_src_vec_dim}, ${BBT_src_scalar_per_vec}, ${BBT_dst_scalar_per_vec_k1}, ${BBT_lds_add_extra_n}, ${CS_m_Xdl_per_wave_per_shuffle}, ${CS_n_Xdl_per_wave_per_shuffle}, ${CTT_cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl}, ${CTT_scalar_per_vector_n_wave_n_per_Xdl}>,

"""

    # function that takes in operation from gemm_op and gets tuning parameters
    def emit(self,operation):
        #name = (str(operation.tile_desc.block_size) + "_" + str(operation.tile_desc.m_per_block) + "_" + str(operation.tile_desc.n_per_block) + "_" + str(operation.tile_desc.ak1))
        values = {
            #'name' : name,
            'layout_a' : operation.A.layout,
            'layout_b' : operation.B.layout,
            'layout_ds' : operation.Ds.layout,
            'layout_e' : operation.E.layout,
            'type_a' : operation.A.element,
            'type_b' : operation.B.element,
            'type_acc' : operation.acc,
            'type_cshuffle' : operation.cs_type,
            'type_ds' : operation.Ds.element,
            'type_e' : operation.E.element,
            'elementwise_op_a' : operation.a_elem_op,
            'elementwise_op_b' : operation.b_elem_op,
            'elementwise_op_cde' : operation.cde_elem_op,
            'Gemm_spec' : operation.gemm_specialization,
            'num_gemmk_prefetch_stage' : str(operation.tile_desc.num_gemmk_prefetch_stage),
            'block_size' : str(operation.tile_desc.block_size),
            'mperblock' : str(operation.tile_desc.m_per_block),
            'nperblock' : str(operation.tile_desc.n_per_block),
            'kperblock' : str(operation.tile_desc.k_per_block),
            'ak1' : str(operation.tile_desc.ak1),
            'bk1' : str(operation.tile_desc.bk1),
            'mperXDL' : str(operation.tile_desc.m_per_XDL),
            'nperXDL' : str(operation.tile_desc.n_per_XDL),
            'mXdlperwave' : str(operation.tile_desc.m_Xdl_per_wave),
            'nXdlperwave' : str(operation.tile_desc.n_Xdl_per_wave),
            'ABT_thread_cluster_lengths_K0_M_K1' : operation.a_block_transfer.thread_cluster_length,
            'ABT_thread_cluster_arrange_order' : operation.a_block_transfer.thread_cluster_arrange_order,
            'ABT_src_access_order' : operation.a_block_transfer.src_access_order,
            'ABT_src_vec_dim' : str(operation.a_block_transfer.src_vec_dim),
            'ABT_src_scalar_per_vec' : str(operation.a_block_transfer.src_scalar_per_vector),
            'ABT_dst_scalar_per_vec_k1' : str(operation.a_block_transfer.dst_scalar_per_vector_k1),
            'ABT_lds_add_extra_m' : str(operation.a_block_transfer.lds_add_extra_dim),
            'BBT_thread_cluster_lengths_K0_N_K1' : operation.b_block_transfer.thread_cluster_length,
            'BBT_thread_cluster_arrange_order' : operation.b_block_transfer.thread_cluster_arrange_order,
            'BBT_src_access_order' : operation.b_block_transfer.src_access_order,
            'BBT_src_vec_dim' : str(operation.b_block_transfer.src_vec_dim),
            'BBT_src_scalar_per_vec' : str(operation.b_block_transfer.src_scalar_per_vector),
            'BBT_dst_scalar_per_vec_k1' : str(operation.b_block_transfer.dst_scalar_per_vector_k1),
            'BBT_lds_add_extra_n' : str(operation.b_block_transfer.lds_add_extra_dim),
            'CS_m_Xdl_per_wave_per_shuffle' : str(operation.cshuffle.m_Xdl_per_wave_per_shuffle),
            'CS_n_Xdl_per_wave_per_shuffle' : str(operation.cshuffle.n_Xdl_per_wave_per_shuffle),
            'CTT_cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl' : operation.c_block_transfer.cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl,
            'CTT_scalar_per_vector_n_wave_n_per_Xdl' : str(operation.c_block_transfer.scalar_per_vector_n_wave_n_per_Xdl),
        }

        # name = (str(operation.tile_desc.block_size) + "_" + str(operation.tile_desc.m_per_block) + "_" + str(operation.tile_desc.n_per_block)
        # + "_" + str(operation.tile_desc.k_per_block) + "_" + str(operation.tile_desc.ak1))

        # defining the template to use and substituting the values
        template = self.gemm_op_template
        instances = SubstituteTemplate(template, values)
        print(instances)
        
        # cf = open("instances.cpp",'w')
        # cf.write(SubstituteTemplate(template, values))
        # cf.close()

        # cf = open("%s.cpp" % name,'w')
        # cf.write(SubstituteTemplate(template, values))
        # cf.close()
        