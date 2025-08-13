#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h> // 确保包含CUDA运行时头
#include <cuda_fp16.h>

// 设备端模板加法函数，支持标量和 CUDA 向量类型
template <typename T>
__device__ T add(const T &a, const T &b)
{
    if constexpr (std::is_arithmetic_v<T>) {
        return a + b;
    }
    else if constexpr (std::is_same_v<T, float2>) {
        return make_float2(a.x + b.x, a.y + b.y);
    }
    else if constexpr (std::is_same_v<T, float4>) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    else if constexpr (std::is_same_v<T, half>) {
        return __hadd(a,b);
    }
    else {
        static_assert(sizeof(T) == 0, "Unsupported type for add()");
    }
}

// Kernel
template<typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n, size_t step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + step;
    for (size_t i = idx; i < n; i += step) {
        c[i] = add(a[i], b[i]);
    }
}

// Wrapper调用
template<typename T>
void vector_add(T* c, const T* a, const T* b, size_t n, const dim3& grid, const dim3& block) {
    size_t step = grid.x * block.x;
    add_kernel<T><<<grid, block>>>(c, a, b, n, step);
}

// Error checking macro
#define CUDA_CHECK(call)                                             \
{                                                                    \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << "\n";      \
        exit(1);                                                     \
    }                                                                \
}

// 主程序示范
int main()
{
    const size_t SIZE = 1 << 30; // 元素总数 half个数
    size_t size_bytes_half = SIZE * sizeof(__half);

    std::vector<__half> h_a(SIZE);
    std::vector<__half> h_b(SIZE);
    std::vector<__half> h_c(SIZE);

    // 初始化，注意要从float转换为half
    for (size_t i = 0; i < SIZE; ++i) {
        h_a[i] = __float2half(1.0f);
        h_b[i] = __float2half(2.0f);
        h_c[i] = __float2half(0.0f);
    }

    __half *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size_bytes_half));
    CUDA_CHECK(cudaMalloc(&d_b, size_bytes_half));
    CUDA_CHECK(cudaMalloc(&d_c, size_bytes_half));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes_half, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes_half, cudaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim((SIZE + block_dim.x - 1) / block_dim.x);

    vector_add<__half>(d_c, d_a, d_b, SIZE, grid_dim, block_dim);

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes_half, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    std::cout << "执行完毕" << std::endl;

    // 输出最后一个half转换为float验证
    std::cout << "c[最后一个half] = " << __half2float(h_c[SIZE - 1]) << std::endl; // 应该是3.0f
    return 0;
}
