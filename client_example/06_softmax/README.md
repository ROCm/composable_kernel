# Client Example: 4D Softmax

## Theory

This client example demonstrates **softmax computation over 4D tensors**. Softmax is a key operation in deep learning, especially in attention mechanisms and classification, converting logits into normalized probabilities.

**Mathematical Formulation:**
Given input $X$ and axis $a$:
$$
\text{softmax}(X)_i = \frac{\exp(X_i)}{\sum_j \exp(X_j)}
$$

**Algorithmic Background:**
- Softmax is implemented using a numerically stable algorithm:
  1. Subtract the maximum value for numerical stability.
  2. Exponentiate and sum.
  3. Normalize by the sum.
- Efficient parallel softmax requires careful reduction and memory access patterns.
- This example demonstrates softmax over a 4D tensor, as used in attention and vision models.

## How to Run

### Prerequisites
```bash
cd composable_kernel/build
make -j install
```

### Build and Execute
```bash
cd composable_kernel/client_example/06_softmax
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run
./softmax4d
```

## Source Code Structure

### Directory Layout
```
client_example/06_softmax/
├── softmax4d.cpp         # Main client example: sets up, runs, and verifies 4D softmax
├── CMakeLists.txt        # Build configuration for the example
```

### Key Functions

- **main()** (in `softmax4d.cpp`):  
  Sets up input tensors, configures softmax parameters, launches the softmax kernel, and verifies the result.
- **Softmax kernel invocation**:  
  Uses the Composable Kernel device API to launch the softmax operation.

This client example provides a demonstration of efficient, numerically stable softmax for 4D tensors in deep learning models.
