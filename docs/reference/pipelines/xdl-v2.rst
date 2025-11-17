.. _xdl-v2:

Xdl - v2 - Intrawave
----------------------

**buffer_load(1:prefetch)**

.. code-block::

    // Global prefetch [2, PrefetchStages]
    static_for<1, PrefetchStages, 1>{}([&](auto iprefetch) {
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, iprefetch);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, iprefetch);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
    });

    // main body
    if constexpr(HasMainLoop)
    {
        index_t i = 0;
        do
        {
            static_for<0, PrefetchStages, 1>{}([&](auto iprefetch) {
                // -------------------------------------------------------------------------------------------
                block_sync_lds();
                static_for<0, KRepeat, 1>{}([&](auto k0) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                            make_tuple(m0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                            a_block_buf,
                                            a_thread_desc_,
                                            make_tuple(m0, I0, k0, I0),
                                            a_thread_buf);
                    });
                    static_for<0, NRepeat, 1>{}([&](auto n0) {
                        b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                            make_tuple(n0, I0, I0, Number<k0 * KPerInnerLoop>{}),
                                            b_block_buf,
                                            b_thread_desc_,
                                            make_tuple(n0, I0, k0, I0),
                                            b_thread_buf);
                    });
                    __builtin_amdgcn_sched_barrier(0);
                    // NOTE: Synchronize threads in a workgroup at the start of each MAC
                    // cluster, but except the first, as we can shorten non-MAC cluster a bit
                    // and there's no observable negative impact. The desired effect is waves in
                    // a workgroup executing MAC in sync. This avoids some out-of-sync waves
                    // hijacking MAC resource from other workgroups and reducing the chance of
                    // latency hiding by waiting for the rest of the workgroup at the eventual
                    // sync point.
                    if constexpr(k0.value != 0 || KRepeat == 1)
                    {
                        __builtin_amdgcn_s_barrier();
                        __builtin_amdgcn_sched_barrier(0);
                    }    

**full class definition**

.. literalinclude:: ../../../include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp