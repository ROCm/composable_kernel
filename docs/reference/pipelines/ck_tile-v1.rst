.. _ck_tile-v1:

ck_tile - AGmemBGmemCReg - V1 - [SCHEDULER]
--------------------------------------------

**loop: buffer_load(i+1)**

.. code-block::

    elementwise_As_res = load_tile_with_elementwise(as_copy_dram_window, a_element_func);
    elementwise_Bs_res = load_tile_with_elementwise(bs_copy_dram_window, b_element_func);    

The ``load_tile_with_elementwise`` function calls the ``load`` method on ``tile_window_with_static_distribution`` instances.  

- The ``tile_window_with_static_distribution::load`` method calls ``make_static_distributed_tensor`` to create distributed tensors for storing loaded data.
- The ``static_distributed_tensor`` class uses ``make_naive_tensor_descriptor_packed`` from ``ck_tile/core/tensor/tensor_descriptor.hpp`` in its tensor slicing operations (via ``set_y_sliced_thread_data``).


**full class definition**

.. literalinclude:: ../../../include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp