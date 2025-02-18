# Grouped Gemm API Higlights

This folder contains example for Grouped GEMM using ck_tile tile-programming implementation.

# Quick Tour for New Users

The Grouped GEMM routines are batched versions of GEMM, executing multiple GEMM operations within a single call. Each GEMM operation performs a matrix-matrix multiplication with general matrices. Unlike regular batched GEMM, Grouped GEMM allows each matrix pair to have different sizes and configurations, making it more flexible for diverse workloads.

### Parsing Arguments
At the beginning, we parse the arguments `group_count`, `repeat`, and `warmup`, which are responsible for the following: 
- `group_count`: number of GEMM operations in the grouped execution, 
- `repeat` – Number of times to repeat the computation for benchmarking,
- `warmup` – Number of iterations before the actual benchmarking (used for cache optimization).

```cpp
// Example
const int group_count = arg_parser.get_int("group_count");
const int repeat      = arg_parser.get_int("repeat");
const int warmup      = arg_parser.get_int("warmup");
```
In the next step, the input parameters `Ms`, `Ns`, `Ks`, as well as the corresponding `stride_As`, `stride_Bs`, and `stride_Cs` are provided. We use std::vector for this because the input size is based on `group_count`.

```cpp
// Example
std::vector<ck_tile::index_t> Ms        = arg_parser.get_int_vec("Ms");
std::vector<ck_tile::index_t> Ns        = arg_parser.get_int_vec("Ns");
std::vector<ck_tile::index_t> Ks        = arg_parser.get_int_vec("Ks");
std::vector<ck_tile::index_t> stride_As = arg_parser.get_int_vec("stride_As");
std::vector<ck_tile::index_t> stride_Bs = arg_parser.get_int_vec("stride_Bs");
std::vector<ck_tile::index_t> stride_Cs = arg_parser.get_int_vec("stride_Cs");
```
Where:
- `Ms` – Row dimensions for each GEMM.
- `Ns` – Column dimensions for each GEMM.
- `Ks` – Inner dimensions for each GEMM (shared between A and B).
- `stride_As` – Stride values for matrix A (how elements are laid out in memory).
- `stride_Bs` – Stride values for matrix B.
- `stride_Cs` – Stride values for matrix C (output matrix).

### HostTensor and Device Memory Buffers (for CPU and GPU) 
Each vector contains one value per GEMM operation, meaning different matrix sizes and strides can be used for different grouped GEMM computations.
The next step is to properly load the input values, for each input matrix `A`, `B`, and `C`, you need to create both `HostTensor` and `DeviceMemory`. Where: 
- HostTensor – Represents the matrix data on the host (CPU). This will store the input matrices before they are transferred to the device for computation.
- DeviceMemory – Represents the matrix data on the device (GPU). This will store the data on the GPU for computation during the Grouped GEMM operation.

#### HostTensor Buffers (for CPU)
In the first step, let's start by creating `HostTensor` for `A`, `B`, `C`. HostTensor allocates memory on the host (CPU) to store the matrices and initializes them with the appropriate dimensions and values. Below is an example code showing how to create HostTensors for those tensors:
```cpp
// Example
std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;
```
Where:
- `a_m_k_tensors`: Vector of `HostTensor` objects for matrix `A` (with dimensions `M × K`). Each tensor will store the data of one grouped GEMM operation.
- `b_k_n_tensors`: Vector of `HostTensor` objects for matrix `B` (with dimensions `K × N`).
- `c_m_n_tensors`: Vector of `HostTensor` objects for matrix `C` (the output matrix with dimensions `M × N`).

Notice that the std::vector container is used for this purpose throughout. As mentioned above, the number of HostTensors is equal to `group_count`.

#### Device Memory Buffers (for GPU)
`DeviceMemory for A, B, C`: Allocate memory on the device (GPU)  memory management system and transfer the matrices from HostTensor to DeviceMemory for actual computation.
```cpp
// Example
std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;
``` 
Where:
- `a_m_k_dev_buf`: For storing matrix A on the GPU.
- `b_k_n_dev_buf`: For storing matrix B on the GPU.
- `c_m_n_dev_buf`: For storing the result matrix C on the GPU.

## Prepare data
In the next step, you need to fill the input tensors with pseudo-random values in the range of -5 to 5 (to this goal we use FillUniformDistribution) and apply the stride for each tensor. Additionally, you will need to create descriptors for each input tensor.

To get stride A, B, and C, `get_default_stride` function can be applied. This is template function that calculates the default stride for a 2D array layout based on whether it is row-major or column-major. Template parameter that determines whether the storage order is row-major (true) or column-major (false). The function takes four params `row`, `col`, `stride` and `bool_constant<is_row_major>`. If stride is explicitly provided (`stride != 0`), it is returned as-is. Otherwise, if `stride == 0` (i.e., not provided), the function computes the default stride. The Row-major order (`is_row_major == true`), the stride is set to the number of columns (col). The column-major order (`is_row_major == false`), the stride is set to the number of rows (row). This function is useful when working with dynamically allocated 2D arrays, where the user may not specify the stride explicitly. It ensures a natural default stride based on the chosen storage order.

```cpp
// Example, API
template <bool is_row_major>
auto get_default_stride(std::size_t row, std::size_t col, std::size_t stride, bool_constant<is_row_major>) {
  // code
}
```
Where: 
- `is_row_major`: A bool template parameter that determines whether the storage order is row-major (true) or column-major (false).
- `row`: The number of rows in the matrix.
- `col`: The number of columns in the matrix.
- `stride`: The current stride (the distance between consecutive elements in memory).
- `bool_constant<is_row_major>`: A tag type that helps in differentiating behavior at compile-time.

In the next step, we need to create host descriptors for each input tensor A, B, and C. For this purpose, we use the `f_host_tensor_descriptor` (code belowe) function, which takes four parameters: row, col, stride, and layout. This function returns a HostTensorDescriptor based on the specified layout.

```cpp
// Example for tensor A
ck_tile::HostTensor<ADataType>(f_host_tensor_descriptor(M, K, stride_As[i], a_layout)))
```

After creating the host_tensors, it's time to create deviceMem for each tensor `A`, `B`, and `C`, and then transfer the data to the device. To create a DeviceMem object, we need to provide the buffer size in bytes. For this purpose, we use the `get_element_space_size_in_bytes()` function. The tensors created this way are suitable for storing data. To transfer data from the host to the device, we use the `ToDevice()` function, passing as a parameter the data we previously generated, e.g., `a_m_k_tensors[i].data()`.

"In the final step of data preparation before execution, we need to retrieve pointers to the buffers of `A`, `B`, and `C` stored on the device using `->GetDeviceBuffer()` and pack them into a shared container, for example: `gemm_descs.push_back({p_a, p_b, p_c, M, N, K, stride_As[i], stride_Bs[i], stride_Cs[i]})`, where `gemm_descs` is `std::vector<grouped_gemm_kargs> gemm_descs`. The container should include values such as:
```cpp
struct GroupedGemmHostArgs
{
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
};
```
The prepared data—warmup, repeat, group_count, and gemm_descs—can be passed to the invoke_gemm function. This is a templated function that additionally takes three template parameters: ALayout, BLayout, and CLayout. The API of this function is as follows:

```cpp
// Example, API
template <typename ALayout, typename BLayout, typename CLayout>
float invoke_gemm(int n_warmup,
                  int n_repeat,
                  int group_count,
                  const std::vector<grouped_gemm_kargs>& args)
```
This function returns the execution time in milliseconds. In this function, we allocate the workspace size required for computation, as follows:
```cpp
// Example
ck_tile::DeviceMem gemm_workspace;
gemm_workspace.Realloc(GetWorkspaceSize(args));
```
In the final step, we pass the arguments to group_gemm and launch the kernel, using grouped_gemm function.
```cpp
// API
template <typename ALayout, typename BLayout, typename CLayout>
float grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                   const ck_tile::stream_config& s,
                   void* p_workspace_)
```
## Build
```
# in the root of ck_tile
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the gemm calculation
make tile_example_grouped_gemm -j
```
This will result in an executable `build/bin/tile_example_grouped_gemm`

## example
```
args:
 -Ms          M dimensions - (Default: empty).
 -Ns          N dimensions - (Default: empty).
 -Ks          K dimensions - (Default: empty).
 -stride_As   Tensor A strides - (Default: empty).
 -stride_Bs   Tensor B strides - (Default: empty).
 -stride_Cs   Tensor C strides - (Default: empty).
 -a_layout    A tensor data layout - (Default: Row).
 -b_layout    B tensor data layout - (Default: Col).
 -c_layout    C tensor data layout - (Default: Row).
 -validate    0. No validation, 1. Validation on CPU. (Default: 1).
 -warmup      Number of iterations before benchmark the kernel. (Default: 10).
 -repeat      Number of iterations to benchmark the kernel. (Default: 100).
 -group_count Group count. (Default: 16).
```
