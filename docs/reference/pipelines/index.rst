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

:ref:`BlockwiseGemmXdlops_pipeline_v2 <xdl-v2>`

.. code-block::

    buffer_load(0)
    lds_write(0)
    buffer_load(1:prefetch)
        lds_read(i)
        gemm(i)
        lds_write(i+1)
        buffer_load(i+prefetch)    

**TODO**

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

Documentation - Xdl Pipelines
-------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Xdl Pipelines
   
   xdl-v1
   xdl-v2

Documentation - CK_TILE Pipelines
-------------------------------

.. toctree::
   :maxdepth: 2
   :caption: CK Tile Pipelines
   
   ck_tile-v1