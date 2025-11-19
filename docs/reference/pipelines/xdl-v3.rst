.. _xdl-v3:

Xdl - v3 - Intrawave
----------------------

**loop: lds_write(i+1)**

**buffer_load(i+2)**

.. code-block::
    
    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

    static_for<0, KRepeat, 1>{}([&](auto k0) {
        static_for<0, MRepeat, 1>{}([&](auto m0) {
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                vector_type<ComputeDataTypeBuf, KPack> a_thread_vec;
                vector_type<ComputeDataTypeBuf, KPack> b_thread_vec;

                static_for<0, KPack, 1>{}([&](auto ik) {
                    a_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                            make_tuple(m0, I0, k0, ik))>{}];
                    b_thread_vec.template AsType<ComputeDataTypeBuf>()(ik) =
                        b_thread_buf[Number<b_thread_desc_.CalculateOffset(
                            make_tuple(n0, I0, k0, ik))>{}];
                });

                using mfma_input_type =
                    typename vector_type<ComputeDataTypeBuf,
                                            xdlops_gemm.K1PerXdlops>::type;

                constexpr index_t c_offset =
                    c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

**full class definition**

.. literalinclude:: ../../../include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp