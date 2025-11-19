.. _ck_tile-mem:

ck_tile - AgBgCrMem - [SCHEDULER]
--------------------------------------------

**loop: buffer_load(i+prefetch)**

.. code-block::
    
    static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
        a_block_tiles.at(number<prefetch_idx>{}) =
            load_tile_with_elementwise(a_copy_dram_window, a_element_func);

        move_tile_window(a_copy_dram_window, a_dram_tile_window_step);

        b_block_tiles.at(number<prefetch_idx>{}) =
            load_tile_with_elementwise(b_copy_dram_window, b_element_func);

        move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
    });

**full class definition**

.. literalinclude:: ../../../include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp