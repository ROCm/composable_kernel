.. _xdl-v1:

Xdl - v1 - Intrawave
----------------------

**loop: buffer_load(i+1)**

.. code-block::

    static_for<0, KRepeat, 1>{}([&](auto k) {
        static_for<0, MRepeat, 1>{}([&](auto m0) {
            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k,
                                make_tuple(m0, I0, I0, Number<k * AMmaKStride>{}),
                                a_block_buf,
                                a_thread_desc_,
                                make_tuple(m0, I0, k, I0),
                                a_thread_buf);
            static_for<0, NRepeat, 1>{}([&](auto n0) {
                b_thread_copy_.Run(b_block_desc_n0_n1_n2_k,
                                    make_tuple(n0, I0, I0, Number<k * BMmaKStride>{}),
                                    b_block_buf,
                                    b_thread_desc_,
                                    make_tuple(n0, I0, k, I0),
                                    b_thread_buf);
            });
        });
    });

The ``a_thread_copy_.Run`` function is a member of the ``ThreadwiseTensorSliceTransfer_v4`` class.  The ``ThreadwiseTensorSliceTransfer_v4`` class uses,

- ``make_tensor_coordinate`` from ``tensor_descriptor.hpp`` 
- ``make_naive_tensor_descriptor_packed`` from ``tensor_descriptor_helper.hpp``

**full class definition**

.. literalinclude:: ../../../include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp