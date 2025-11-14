Pipeline policies
====================

Xdl pipelines
--------------

GEMM Pipelines defined in ``include/ck/tensor_operation/gpu/block`` directory.

:ref:`BlockwiseGemmXdlops_pipeline_v1 <xdl-v1>`

.. code-block::

    buffer_load(0)
    lds_write(0)
    loop:
       buffer_load(i+1)
       gemm(i)
       lds_write(i+1)

**TODO**

- BlockwiseGemmXdlops_pipeline_v2
- BlockwiseGemmXdlops_pipeline_v3

CK_TILE GEMM pipelines
-------------------------

Pipelines defined in ``include/ck_tile/ops/gemm/pipeline`` directory.

:ref:`GemmPipelineAGmemBGmemCRegV1 <ck_tile-v1>`

.. code-block::

    buffer_load(0)
    lds_write(0) 
    loop:
        buffer_load(i+1)  
        gemm(i)
        lds_write(i+1)

**TODO**

- GemmPipelineAgBgCrMem
- GemmPipelineAgBgCrCompV3