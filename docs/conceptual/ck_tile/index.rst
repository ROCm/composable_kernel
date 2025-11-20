.. _ck_tile_conceptual:

CK Tile Conceptual Documentation
================================

[TO DO: Describe what CK Tile is and why it's more advantageous than old CK, less overhead, etc. We can put this also in the old CK doc somewhere. Maybe on the landing page. And then it gets removed once Old CK is removed.]


Welcome to the conceptual documentation for CK Tile, the core abstraction layer of Composable Kernel that enables efficient GPU programming through compile-time coordinate transformations and tile-based data distribution.

Overview
--------

CK Tile provides a mathematical framework for expressing complex GPU computations through:

- Automatic Memory Coalescing: Ensures optimal memory access patterns without manual optimization
- Thread Cooperation: Coordinates work distribution across GPU's hierarchical thread model
- Zero-Overhead Abstractions: Compile-time optimizations ensure no runtime performance penalty
- Portable Performance: Same code achieves high performance across different GPU architectures

Why CK Tile?
------------

Traditional GPU programming requires manual management of:

- Thread-to-data mapping calculations
- Memory coalescing patterns
- Bank conflict avoidance
- Boundary condition handling

CK Tile automates all of these concerns through a unified abstraction that maps logical problem coordinates to physical GPU resources.

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: CK Tile Concepts

   introduction_motivation
   buffer_views
   tensor_views
   tile_distribution
   coordinate_systems
   terminology
   adaptors
   transforms
   descriptors
   tile_window
   load_store_traits
   space_filling_curve
   static_distributed_tensor
   stream_k
   convolution_example
   coordinate_movement
   lds_index_swapping
   swizzling_example
   tensor_coordinates
   sweep_tile
   encoding_internals
   thread_mapping
   hardware/index

Learning Path
-------------

1. Start Here: :ref:`ck_tile_introduction`
   
   Understand the problems CK Tile solves and why it's important for efficient GPU programming.

2. Foundation: :ref:`ck_tile_buffer_views`
   
   Learn how CK Tile provides structured access to raw GPU memory across different address spaces.

3. Multi-Dimensional Views: :ref:`ck_tile_tensor_views`
   
   Understand how to work with multi-dimensional data structures and memory layouts.

4. Core API: :ref:`ck_tile_distribution`
   
   Learn the tile distribution system that automatically maps work to GPU threads.

5. Mathematical Framework: :ref:`ck_tile_coordinate_systems`
   
   Deep dive into the coordinate transformation system that powers CK Tile's abstractions.

6. Reference: :ref:`ck_tile_terminology`
   
   Comprehensive glossary of all terms and concepts used in CK Tile.

Key Concepts at a Glance
------------------------

Coordinate Spaces
~~~~~~~~~~~~~~~~~

- P-space: Processing element coordinates (thread, warp, block)
- Y-space: Local tile access patterns
- X-space: Physical tensor coordinates
- D-space: Linearized memory addresses

Core Components
~~~~~~~~~~~~~~~

- ``BufferView``: Type-safe access to GPU memory
- ``TileDistribution``: Automatic work distribution
- ``TileWindow``: Efficient data loading/storing
- ``Encoding``: Compile-time distribution specification

Quick Example
-------------

.. code-block:: cpp

   // Define how to distribute a 256x256 tile across threads
   using Encoding = tile_distribution_encoding<
       sequence<>,                              // No replication
       tuple<sequence<4,2,8,4>,                // M dimension hierarchy
             sequence<4,2,8,4>>,               // N dimension hierarchy
       tuple<sequence<1,2>, sequence<1,2>>,    // Thread mapping
       tuple<sequence<1,1>, sequence<2,2>>,    // Minor indices
       sequence<1,1,2,2>,                      // Y-space mapping
       sequence<0,3,0,3>                       // Y-space minor
   >;
   
   // Create distribution and load data
   auto distribution = make_static_tile_distribution(Encoding{});
   auto window = make_tile_window(tensor_view, tile_size, origin, distribution);
   auto tile = window.load();
   
   // Process tile efficiently
   sweep_tile(tile, [](auto idx) { /* computation */ });

Performance Impact
------------------

CK Tile enables kernels to achieve:

- >90% memory bandwidth utilization through coalescing
- Minimal register pressure via efficient data distribution
- Zero bank conflicts in shared memory access
- Portable performance across GPU generations

Next Steps
----------

Ready to dive deeper? Start with :ref:`ck_tile_introduction` to understand the motivation and core concepts behind CK Tile.

For practical examples, see the `example/ck_tile` directory in the Composable Kernel repository.
