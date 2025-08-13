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
    else if constexpr (std::is_same_v<T, half2>) {
        return __hadd2(a,b);
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
    const size_t SIZE = 1 << 24; // 元素总数 (float个数)
    size_t half2_size = SIZE / 2;


    std::vector<half2> h_a2(half2_size, __float2half2_rn(1.0f));
    std::vector<half2> h_b2(half2_size, __float2half2_rn(2.0f));
    std::vector<half2> h_c2(half2_size, __float2half2_rn(0.0f));

    half2 *d_a2, *d_b2, *d_c2;
    CUDA_CHECK(cudaMalloc(&d_a2, half2_size * sizeof(half2)));
    CUDA_CHECK(cudaMalloc(&d_b2, half2_size * sizeof(half2)));
    CUDA_CHECK(cudaMalloc(&d_c2, half2_size * sizeof(half2)));

    CUDA_CHECK(cudaMemcpy(d_a2, h_a2.data(), half2_size * sizeof(half2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), half2_size * sizeof(half2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c2, h_c2.data(), half2_size * sizeof(half2), cudaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim((half2_size + block_dim.x - 1) / block_dim.x);

    // 调用改为half2类型，数量为VECTOR_SIZE
    vector_add<half2>(d_c2, d_a2, d_b2, half2_size, grid_dim, block_dim);

    CUDA_CHECK(cudaMemcpy(h_c2.data(), d_c2, half2_size * sizeof(half2), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_c2));

    std::cout << "执行完毕" << std::endl;
    return 0;
}
