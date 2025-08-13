#include <stdio.h>
#include <iostream>
#include <vector>
#include <cuda_runtime.h> // 确保包含CUDA运行时头

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
    const size_t SIZE = 1 << 40; // 元素总数 (float个数)
    const size_t VECTOR_SIZE = SIZE / 4; // float4数量

    size_t size_bytes_float = SIZE * sizeof(float);
    size_t size_bytes_float4 = VECTOR_SIZE * sizeof(float4);

    // 使用float4数组来分配和初始化数据
    std::vector<float4> h_a(VECTOR_SIZE, make_float4(1.f, 1.f, 1.f, 1.f));
    std::vector<float4> h_b(VECTOR_SIZE, make_float4(2.f, 2.f, 2.f, 2.f));
    std::vector<float4> h_c(VECTOR_SIZE, make_float4(0.f, 0.f, 0.f, 0.f));

    float4 *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size_bytes_float4));
    CUDA_CHECK(cudaMalloc(&d_b, size_bytes_float4));
    CUDA_CHECK(cudaMalloc(&d_c, size_bytes_float4));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes_float4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes_float4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes_float4, cudaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim(256);

    // 调用改为float4类型，数量为VECTOR_SIZE
    vector_add<float4>(d_c, d_a, d_b, VECTOR_SIZE, grid_dim, block_dim);

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes_float4, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // 输出最后一个float4的最后一个分量验证
    std::cout << "执行完毕" << std::endl;
    std::cout << "c[最后一个float4].w = " << h_c[VECTOR_SIZE - 1].w << std::endl; // 应该是3.0

    return 0;
}
