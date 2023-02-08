==========================
Supported Primitives Guide
==========================

------------
Introduction
------------

This document contains details of supported primitives in Composable Kernel (CK). In contrast to the API Reference
Guide, the Supported Primitives Guide is an introduction to the math which underpins the algorithms implemented in CK.

Softmax
^^^^^^^

For vectors :math:`x^{(1)}, x^{(2)}, \ldots, x^{(T)}` of size :math:`B` we can decompose the softmax of concatenated
:math:`x = [ x^{(1)}\ | \ \ldots \ | \ x^{(T)} ]` as,

.. math::
   :nowrap:

   \begin{align}
      m(x) & = m( [ x^{(1)}\ | \ \ldots \ | \ x^{(T)} ] ) = \max( m(x^{(1)}),\ldots, m(x^{(T)}) )  \\
      f(x) & = [\exp( m(x^{(1)}) - m(x) ) f( x^{(1)} )\ | \ \ldots \ | \ \exp( m(x^{(T)}) - m(x) ) f( x^{(T)} )] \\
      l(x) & = \exp( m(x^{(1)}) - m(x) )\ l(x^{(1)}) + \ldots + \exp( m(x^{(T)}) - m(x) )\ l(x^{(1)}) \\
      \operatorname{softmax}(x) &= f(x)\ / \ l(x)
   \end{align}

where :math:`f(x^{(j)}) = \exp( x^{(j)} - m(x^{(j)}) )` is of size :math:`B` and
:math:`l(x^{(j)}) = f(x_1^{(j)})+ \ldots+ f(x_B^{(j)})` is a scalar.