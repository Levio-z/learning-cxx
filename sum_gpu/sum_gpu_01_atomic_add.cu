#include <iostream>
#include <cuda_runtime.h>
// 用于验证结果，可根据需要替换为其他数学库
#include <cstdlib> 

// 核函数声明（模板函数定义一般放头文件，这里为演示直接写在一起）
template <typename T>
__global__ void sum_kernel(T *result, const T *input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 注意：这里有“竞争条件”问题！多个线程同时操作 *result，
        // 实际场景应改用原子操作（如 atomicAdd）或归约（reduction）优化
        atomicAdd(result,input[idx]); 
    }
}

int main() {
    // ========== 1. 定义数据规模和变量 ==========
    const size_t n = 1<<20; 
    // 假设用 float 类型演示，可换成 double 等支持类型
    using DataType = float; 

    // ========== 2. 主机端（CPU）内存分配与初始化 ==========
    DataType *h_input = new DataType[n];
    DataType h_result = 0; 
    for (size_t i = 0; i < n; ++i) {
        // 简单赋随机值，范围 [0, 1)
        h_input[i] = static_cast<DataType>(rand()) / RAND_MAX; 
    }

    // ========== 3. 设备端（GPU）内存分配 ==========
    DataType *d_input, *d_result;
    cudaMalloc((void **)&d_input, n * sizeof(DataType));
    cudaMalloc((void **)&d_result, sizeof(DataType));

    // ========== 4. 数据拷贝（主机 -> 设备） ==========
    cudaMemcpy(d_input, h_input, n * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(DataType), cudaMemcpyHostToDevice);

    // ========== 5. 定义核函数执行配置 ==========
    // 每个 block 的线程数，根据 GPU 架构合理调整（如 256、512 等）
    const int blockSize = 256; 
    // 计算需要的 block 数量，向上取整
    const int gridSize = (n + blockSize - 1) / blockSize; 

    // ========== 6. 调用核函数 ==========
    sum_kernel<<<gridSize, blockSize>>>(d_result, d_input, n);
    // 检查核函数调用错误
    cudaError_t err = cudaGetLastError(); 
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // ========== 7. 数据拷贝（设备 -> 主机），获取结果 ==========
    cudaMemcpy(&h_result, d_result, sizeof(DataType), cudaMemcpyDeviceToHost);

    // ========== 8. 验证结果（可选，这里用 CPU 再算一遍对比） ==========
    DataType cpu_result = 0;
    for (size_t i = 0; i < n; ++i) {
        cpu_result += h_input[i];
    }
    std::cout << "GPU result: " << h_result << std::endl;
    std::cout << "CPU result: " << cpu_result << std::endl;

    // ========== 9. 释放内存 ==========
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}