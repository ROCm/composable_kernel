===============
Wrapper
===============

-------------------------------------
Description
-------------------------------------

.. note::

    The wrapper is under development and its functionality is limited.


CK provides a lightweight wrapper for more complex operations implemented in 
the library. It allows indexing of nested layouts using a simple interface 
(avoiding complex descriptor transformations). 

Example:

.. code-block:: c

    const auto shape_4x2x4         = ck::make_tuple(4, ck::make_tuple(2, 4));
    const auto strides_s2x1x8      = ck::make_tuple(2, ck::make_tuple(1, 8));
    const auto layout = ck::wrapper::make_layout(shape_4x2x4, strides_s2x1x8);

    std::cout << "dims:4,(2,4) strides:2,(1,8)" << std::endl;
    for(ck::index_t h = 0; h < ck::wrapper::size<0>(layout); h++)
    {
        for(ck::index_t w = 0; w < ck::wrapper::size<1>(layout); w++)
        {
            std::cout << layout(ck::make_tuple(h, w)) << " ";
        }
        std::cout << std::endl;
    }

Output::

    dims:4,(2,4) strides:2,(1,8)
    0 1 8 9 16 17 24 25 
    2 3 10 11 18 19 26 27 
    4 5 12 13 20 21 28 29 
    6 7 14 15 22 23 30 31 

-------------------------------------
Layout
-------------------------------------

.. doxygenstruct:: ck::wrapper::Layout

-------------------------------------
Layout helpers
-------------------------------------

.. doxygenfile:: layout_utils.hpp
