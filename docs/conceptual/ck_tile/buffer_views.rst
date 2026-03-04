.. _ck_tile_buffer_views:

**********************************
Buffer Views - Raw Memory Access
**********************************

Overview
--------

At the foundation of the CK Tile system lies BufferView, a compile-time abstraction that provides structured access to raw memory regions within GPU kernels. This serves as the bridge between the hardware's physical memory model and the higher-level abstractions that enable efficient GPU programming. BufferView encapsulates the complexity of GPU memory hierarchies while exposing a unified interface that works seamlessly across different memory address spaces including global memory shared across the entire device, local data share (LDS) memory shared within a workgroup, or the ultra-fast register files private to each thread.

BufferView serves as the foundation for :ref:`ck_tile_tensor_views`, which add multi-dimensional structure on top of raw memory access. Understanding BufferView is essential before moving on to more complex abstractions like :ref:`ck_tile_tile_distribution` and :ref:`ck_tile_tile_window`.

By providing compile-time knowledge of buffer properties through template metaprogramming, BufferView enables the compiler to generate optimal machine code for each specific use case. This zero-overhead abstraction ensures that the convenience of a high-level interface comes with no runtime performance penalty.

One of BufferView's most important features is its advanced handling of out-of-bounds memory access. Unlike CPU programming where such accesses typically result in segmentation faults or undefined behavior, GPU programming must gracefully handle cases where threads attempt to access memory beyond allocated boundaries. BufferView provides configurable strategies for these scenarios, where developers can choose between returning either numerical zero values or custom sentinel values for invalid accesses. This flexibility is important for algorithms that naturally extend beyond data boundaries, such as convolutions with padding or matrix operations with non-aligned dimensions.

The abstraction extends beyond simple memory access to encompass both scalar and vector data types. GPUs achieve their highest efficiency when loading or storing multiple data elements in a single instruction. BufferView seamlessly supports these vectorized operations, automatically selecting the appropriate hardware instructions based on the data type and access pattern. This capability transforms what would be multiple memory transactions into single, efficient operations that fully utilize the available memory bandwidth.

BufferView also incorporates AMD GPU-specific optimizations that leverage unique hardware features. The AMD buffer addressing mode, for instance, provides hardware-accelerated bounds checking that ensures memory safety without the performance overhead of software-based checks. Similarly, BufferView exposes atomic operations that are crucial for parallel algorithms requiring thread-safe updates to shared data structures. These hardware-specific optimizations are abstracted behind a portable interface, ensuring that code remains maintainable while achieving optimal performance.

Memory coherence and caching policies represent another layer of complexity that BufferView manages transparently. Different GPU memory spaces have different coherence guarantees and caching behaviors. Global memory accesses can be cached in L1 and L2 caches with various coherence protocols, while LDS memory provides workgroup-level coherence with specialized banking structures (see :ref:`ck_tile_lds_bank_conflicts` for details on avoiding bank conflicts). BufferView encapsulates these details, automatically applying the appropriate memory ordering constraints and cache control directives based on the target address space and operation type.

Address Space Usage Patterns
----------------------------

.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
      .. mermaid::
      
         flowchart TB
             subgraph CF ["Compute Flow"]
                 direction LR
                 GM1["Global Memory<br/>Input Data"] --> LDS["LDS<br/>Tile Cache"]
                 LDS --> VGPR["VGPR<br/>Working Set"]
                 VGPR --> Compute["Compute<br/>Operations"]
                 Compute --> VGPR
                 VGPR --> LDS2["LDS<br/>Reduction"]
                 LDS2 --> GM2["Global Memory<br/>Output Data"]
             end
   
             subgraph UP ["Usage Pattern"]
                 direction LR
                 P1["1. Load tile from Global → LDS"]
                 P2["2. Load working set LDS → VGPR"]
                 P3["3. Compute in VGPR"]
                 P4["4. Store results VGPR → LDS"]
                 P5["5. Reduce in LDS"]
                 P6["6. Write final LDS → Global"]
   
                 P1 --> P2 --> P3 --> P4 --> P5 --> P6
             end
   
             CF ~~~ UP
   
             style GM1 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
             style LDS fill:#fed7aa,stroke:#f59e0b,stroke-width:2px
             style VGPR fill:#d1fae5,stroke:#10b981,stroke-width:2px
             style Compute fill:#e0e7ff,stroke:#4338ca,stroke-width:2px
      
      
   
   
   

.. image:: diagrams/buffer_views_1.svg
   :alt: Diagram
   :align: center
   
C++ Implementation
------------------

**File**: ``include/ck_tile/core/tensor/buffer_view.hpp``

Basic Creation
~~~~~~~~~~~~~~

By encoding critical properties such as buffer size and address space as template parameters, BufferView transforms what would traditionally be runtime decisions into compile-time constants. This design philosophy enables the compiler to perform aggressive optimizations, including constant propagation, loop unrolling, and instruction selection, that would be impossible with runtime parameters.

The use of compile-time constants extends beyond mere optimization. When the buffer size is encoded in the type system using constructs like ``number<8>{}``, the compiler can statically verify that array accesses are within bounds, eliminate unnecessary bounds checks, and even restructure algorithms to better match the known data dimensions. This compile-time knowledge propagates through the entire computation, enabling optimizations at every level of the abstraction hierarchy.

The address space template parameter represents another crucial design decision. By making the memory space part of the type system, BufferView ensures that operations appropriate for one memory space cannot be accidentally applied to another. This type safety prevents common errors such as attempting atomic operations on register memory or using global memory synchronization primitives on local memory. The compiler enforces these constraints at compile time, transforming potential runtime errors into compile-time diagnostics.

.. code-block:: cpp

   #include <ck_tile/core/tensor/buffer_view.hpp>
   #include <ck_tile/core/numeric/integral_constant.hpp>

   // Create buffer view in C++
   __device__ void example_buffer_creation()
   {
       // Static array in global memory
       float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
       constexpr index_t buffer_size = 8;

   // Create buffer view for global memory
   // Template parameters: <AddressSpace>
   auto buffer_view = make_buffer_view<address_space_enum::global>(
       data,        // pointer to data
       buffer_size  // number of elements
   );
   
   // Implementation detail: The actual C++ template is:
   // template <address_space_enum BufferAddressSpace,
   //           typename T,
   //           typename BufferSizeType,
   //           bool InvalidElementUseNumericalZeroValue = true,
   //           amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default>
   // struct buffer_view

       // Alternative: Create with explicit type
       using buffer_t = buffer_view<float*, address_space_enum::global>;
       buffer_t explicit_buffer{data, number<buffer_size>{}};

       // Access properties at compile time
       constexpr auto size = buffer_view.get_buffer_size();
       constexpr auto space = buffer_view.get_address_space();

       // The buffer_view type encodes:
       // - Data type (float)
       // - Address space (global memory)
       // - Size (known at compile time for optimization)
       static_assert(size == 8, "Buffer size should be 8");
       static_assert(space == address_space_enum::global, "Should be global memory");
   }

Out-of-Bounds Handling
~~~~~~~~~~~~~~~~~~~~~~

Traditional approaches to bounds checking often involve conditional branches that can severely impact performance on GPU architectures, where divergent execution paths within a warp lead to serialization. BufferView's approach sidesteps this problem through two carefully designed modes that maintain performance while providing predictable behavior.

The Zero Value Mode leverages the mathematical property that zero often serves as a neutral element in computations. When an access falls outside the valid buffer range, this mode returns numerical zero without branching. This approach proves particularly effective for algorithms like convolution, where out-of-bounds accesses naturally correspond to zero-padding. The branchless implementation ensures that all threads in a warp follow the same execution path, maintaining the SIMD efficiency that is crucial for GPU performance.

The Custom Value Mode extends this concept by letting developers specify arbitrary sentinel values for invalid accesses. This flexibility accommodates algorithms that require specific values for boundary conditions, such as using negative infinity for maximum operations or special markers for missing data. The implementation maintains the same branchless characteristics, using conditional move instructions or predicated execution to avoid divergent control flow.

.. code-block:: cpp

   // Basic buffer view creation with automatic zero for invalid elements
   void basic_creation_example() {
       // Create data array
       constexpr size_t buffer_size = 8;
       float data[buffer_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
       
       // Create global memory buffer view
       auto buffer_view = make_buffer_view<address_space_enum::global>(data, buffer_size);
   }

   // Custom invalid value mode
   void custom_invalid_value_example() {
       constexpr size_t buffer_size = 8;
       float data[buffer_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
       float custom_invalid = 13.0f;
       
       // Create buffer view with custom invalid value
       auto buffer_view = make_buffer_view<address_space_enum::global>(
           data, buffer_size, custom_invalid);
   }

Get Operations
--------------

Scalar Access
~~~~~~~~~~~~~

The get operations in BufferView form the cornerstone of memory access patterns in CK Tile. These operations embody a advanced understanding of GPU memory systems and the patterns that lead to optimal performance. The scalar access interface incorporates multiple layers of optimization and safety mechanisms that work together to provide both performance and correctness.

The parameter structure of scalar access operations reflects careful design choices aimed at maximizing flexibility while maintaining efficiency. The base index parameter ``i`` represents the primary offset into the buffer, expressed in terms of elements of type T rather than raw bytes. This type-aware indexing prevents common errors related to pointer arithmetic and ensures that vector types are handled correctly. The additional ``linear_offset`` parameter provides fine-grained control over the final access location, enabling complex access patterns without requiring expensive index calculations in the kernel code.

The ``is_valid_element`` parameter provides a solution to conditional memory access. Rather than using traditional if-statements that would cause warp divergence, this boolean parameter enables predicated execution where the memory access occurs unconditionally but the result is conditionally used. This approach maintains uniform control flow across all threads in a warp, preserving the SIMD execution model that is fundamental to GPU performance.

The invalid value modes provide a mechanism for handling the boundary conditions that arise in parallel algorithms. When ``InvalidElementUseNumericalZeroValue`` is set to true, the system returns zero for any invalid access, whether due to the ``is_valid_element`` flag or out-of-bounds indexing. This mode is important for algorithms where zero serves as a natural extension value, such as in image processing with zero-padding or sparse matrix operations where missing elements are implicitly zero.

The custom invalid value mode, activated when ``InvalidElementUseNumericalZeroValue`` is false, offers additional flexibility for algorithms with specific boundary requirements. This mode returns a user-specified value for invalid accesses, accommodating use cases such as sentinel values in sorting algorithms, infinity values in optimization problems, or special markers in data processing pipelines. The implementation ensures that this flexibility comes without performance penalty, using the same branchless execution strategies as the zero mode.

Out-of-bounds handling leverages AMD GPU hardware capabilities to provide safety with minimal impact to performance. When AMD buffer addressing is enabled, the hardware automatically clamps memory accesses to valid ranges, preventing the segmentation faults that would occur on CPU systems. This hardware-assisted bounds checking operates at wire speed, adding no overhead to the memory access path while ensuring that kernels cannot corrupt memory outside their allocated regions.

Vector Access
~~~~~~~~~~~~~

Vector memory operations represent one of the most critical optimizations available in modern GPU programming, and BufferView's vector access interface exposes this capability. By using template parameters to specify vector types through constructs like ``ext_vector_t<float, N>``, the interface enables compile-time selection of optimal load and store instructions that can transfer multiple data elements in a single memory transaction. This vectorization is crucial for :ref:`ck_tile_load_store_traits`, which automatically selects optimal access patterns.

The significance of vector operations extends beyond bandwidth improvements. GPUs are designed with wide memory buses that can transfer 128, 256, or even 512 bits per transaction. When scalar operations access only 32 bits at a time, they utilize only a fraction of this available bandwidth. Vector operations align with these wide buses, enabling full bandwidth utilization and reducing the total number of memory transactions required.

The implementation of vector access maintains the same parameter structure as scalar operations, providing consistency across the API while automatically handling the complexities of multi-element transfers. The system manages alignment requirements, ensures that vector loads and stores use the optimal hardware instructions, and handles cases where vector operations extend beyond buffer boundaries. This transparent handling of edge cases allows developers to use vector operations confidently without manual boundary checks or special-case code for partial vectors.

Scalar vs Vectorized Memory Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
      .. mermaid::
      
         graph LR
             subgraph "Scalar Access (4 instructions)"
                 S1["Load float[0]"] --> R1["Register 1"]
                 S2["Load float[1]"] --> R2["Register 2"]
                 S3["Load float[2]"] --> R3["Register 3"]
                 S4["Load float[3]"] --> R4["Register 4"]
             end
   
             subgraph "Vectorized Access (1 instruction)"
                 V1["Load float4[0]"] --> VR["Vector Register<br/>(4 floats)"]
             end
   
             subgraph "Performance Impact"
                 Perf["4x fewer instructions<br/>Better memory bandwidth<br/>Reduced latency"]
             end
   
             R1 & R2 & R3 & R4 --> Perf
             VR --> Perf
   
             style S1 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
             style S2 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
             style S3 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
             style S4 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
             style V1 fill:#d1fae5,stroke:#10b981,stroke-width:2px
             style Perf fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
      
      
   
   
   

.. image:: diagrams/buffer_views_2.svg
   :alt: Diagram
   :align: center

Understanding BufferView Indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
      .. mermaid::
      
         flowchart LR
             subgraph "Input Parameters"
                 Offset["Offset<br/>(e.g., 5)"]
                 ValidFlag["Valid Flag<br/>(optional)"]
             end
   
             subgraph "Processing"
                 BoundsCheck{{"Bounds Check<br/>offset < buffer_size?"}}
                 FlagCheck{{"Flag Check<br/>valid_flag == True?"}}
                 Access["Access Memory<br/>buffer[offset]"]
             end
   
             subgraph "Output"
                 ValidResult["Valid Result<br/>Return value"]
                 Invalid["Invalid Result<br/>Return 0 or default"]
             end
   
             Offset --> BoundsCheck
             ValidFlag --> FlagCheck
   
             BoundsCheck -->|Yes| FlagCheck
             BoundsCheck -->|No| Invalid
   
             FlagCheck -->|Yes| Access
             FlagCheck -->|No| Invalid
   
             Access --> ValidResult
   
             style Offset fill:#e0e7ff,stroke:#4338ca,stroke-width:2px
             style ValidFlag fill:#e0e7ff,stroke:#4338ca,stroke-width:2px
             style ValidResult fill:#d1fae5,stroke:#10b981,stroke-width:2px
             style Invalid fill:#fee2e2,stroke:#ef4444,stroke-width:2px
      
      
   
   
   

.. image:: diagrams/buffer_views_3.svg
   :alt: Diagram
   :align: center

C++ Get Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   __device__ void example_get_operations()
   {
       // Create buffer view
       float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
       auto buffer_view = make_buffer_view<address_space_enum::global>(data, 8);

       // Simple get - compile-time bounds checking when possible
       auto value_buf = buffer_view.template get<float>(0,1,true); //get the buffer from the buffer view
       float value = value_buf.get(0); //get the value from the buffer

       // Get with valid flag - branchless conditional access
       bool valid_flag = false;
       value_buf = buffer_view.template get<float>(0,1,valid_flag);
       value = value_buf.get(0);
       // Returns 0 valid_flag is false

       // vectorized get
       using float2 = ext_vector_t<float, 2>;
       auto vector_buf = buffer_view.template get<float2>(0, 0, true);
       // Loads 2 floats in a single instruction
       float val1 = vector_buf.get(0);
       float val2 = vector_buf.get(1);
   }

Custom Value Return Mode for OOB & Invalid Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   void scalar_get_operations_example() {

       // Create data array
       constexpr size_t buffer_size = 8;
       float data[buffer_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
       float custom_invalid = 13.0f;
       
       // Create global memory buffer view with zero invalid value mode (default)
       auto buffer_view = make_buffer_view<address_space_enum::global>(data, buffer_size, custom_invalid);
       
       // Invalid element access with is_valid_element=false
       // Returns custom_invalid due to custom invalid value mode
       auto invalid_value = buffer_view.template get<float>(0, 0, false);
       printf("Invalid element: %.1f\n", invalid_value.get(0));
       
       // Out of bounds access - AMD buffer addressing handles bounds checking
       // Will return custom_invalid when accessing beyond buffer_size
       auto oob_value = buffer_view.template get<float>(0, 100, true);
       printf("Out of bounds: %.1f\n", oob_value.get(0));
   }

.. note::

   Partial Out Of Bound (OOB) access during vector reads will return 'junk' values for the OOB access. Zero or custom invalid value is only returned for complete invalid/OOB access, in other words, it is only returned when the first address of the vector is invalid.

Update Operations
-----------------

.. code-block:: cpp

   void scalar_set_operations_example() {
           
       // Create data array
       constexpr size_t buffer_size = 8;
       float data[buffer_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
       
       // Create global memory buffer view
       auto buffer_view = make_buffer_view<address_space_enum::global>(data, buffer_size);
       
       // Basic set: set<T>(i, linear_offset, is_valid_element, value)
       // Sets element at position i + linear_offset = 0 + 2 = 2
       buffer_view.template set<float>(0, 2, true, 99.0f);
       
       // Invalid write with is_valid_element=false (ignored)
       buffer_view.template set<float>(0, 3, false, 777.0f);
       
       // Out of bounds write - handled safely by AMD buffer addressing
       buffer_view.template set<float>(0, 100, true, 555.0f);

       // Vector set
       using float2 = ext_vector_t<float, 2>;
       float2 pair_values{100.0f, 200.0f};
       buffer_view.template set<float2>(0, 5, true, pair_values);
   }

Atomic Operations
-----------------

Atomic vs Non-Atomic Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
.. 
   Original mermaid diagram (edit here, then run update_diagrams.py)
   
      .. mermaid::
      
         graph TB
             subgraph "Non-Atomic Operation (Race Condition)"
                 NA1["Thread 1: Read value (10)"] --> NA2["Thread 1: Add 5 (15)"]
                 NA3["Thread 2: Read value (10)"] --> NA4["Thread 2: Add 3 (13)"]
                 NA2 --> NA5["Thread 1: Write 15"]
                 NA4 --> NA6["Thread 2: Write 13"]
                 NA5 & NA6 --> NA7["Final value: 13 ❌<br/>(Lost update from Thread 1)"]
             end
   
             subgraph "Atomic Operation (Thread-Safe)"
                 A1["Thread 1: atomic_add(5)"] --> A2["Hardware ensures<br/>serialization"]
                 A3["Thread 2: atomic_add(3)"] --> A2
                 A2 --> A4["Final value: 18 ✓<br/>(Both updates applied)"]
             end
   
             style NA7 fill:#fee2e2,stroke:#ef4444,stroke-width:2px
             style A4 fill:#d1fae5,stroke:#10b981,stroke-width:2px
      
      
   
   
   

.. image:: diagrams/buffer_views_4.svg
   :alt: Diagram
   :align: center

C++ Atomic Operations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   __device__ void example_atomic_operations()
   {
       // Shared memory for workgroup-level reductions
       __shared__ float shared_sum[256];
       auto shared_buffer_view = make_buffer_view<address_space_enum::lds>(
           shared_sum, 256
       );

       // Initialize shared memory
       if (threadIdx.x < 256) {
           shared_buffer_view.template set<float>(threadIdx.x, 0.0f, true);
       }
       __syncthreads();

       // Each thread atomically adds to shared memory
       auto my_value = static_cast<float>(threadIdx.x);
       shared_buffer_view.template update<memory_operation_enum::atomic_add, float>(0,0,true,my_value);
       
       // Atomic max for finding maximum value
       shared_buffer_view.template update<memory_operation_enum::atomic_max, float>(0,1,true,my_value);
       
       __syncthreads();
   }

Summary
-------

BufferView abstracts GPU memory hierarchies behind a concise interface. The approach is intended to keep overhead small while enabling optimizations that are otherwise awkward in low-level code.

BufferView offers a unified interface across global, shared, and register memory. Using the same API for each space can lower cognitive overhead, reduce certain classes of mistakes, and support code reuse via template parameters.

Address spaces are encoded in types so that common errors are reported at compile time. Consistent with CK Tile’s zero-overhead design aim,  compile-time checks are favored over runtime guards. The C++ type system enforces memory-space constraints and can make valid cases more amenable to compiler optimization.

BufferView supports configurable handling of invalid values, optional runtime bounds checks, and conditional access patterns. It also provides atomic operations for thread-safe updates. These features are intended to cover common edge cases without adding unnecessary overhead.

By hiding the complexity of different memory spaces while exposing the operations needed for high-performance GPU computing, BufferView establishes a pattern that the rest of CK Tile follows: compile-time abstractions that enhance rather than compromise performance. The :ref:`ck_tile_tensor_views` and :ref:`ck_tile_tile_distribution` add capability while maintaining the efficiency established at the base. For hardware-specific details about memory hierarchies, see :ref:`ck_tile_gpu_basics`.

Next Steps
----------

Continue to :ref:`ck_tile_tensor_views` to learn how to build structured tensor views on top of buffer views.
