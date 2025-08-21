#include <iostream>
#include <cuda_runtime.h>
// 用于生成随机数初始化数据
#include <cstdlib>  
// 用于计时（可选，可了解核函数执行时间）
#include <chrono>  

template <typename T>
__global__ void reduce_warp_shfl_register_kernel(T* output, const T* input, size_t n) {
    size_t tid = threadIdx.x;
    size_t i = tid;

    T sum = 0;
    // 线程束内跨线程加载数据累加
    for (; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // 基于 warp shuffle 指令的归约操作
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // 每 32 个线程（一个 warp）的首线程将结果原子加到全局输出
    if (tid % 32 == 0) {
        atomicAdd(output, sum);
    }
}


int main() {
    // ========== 1. 配置参数与数据类型 ==========
    const size_t n = 1<<20;  // 数组元素数量，可按需调整
    // 这里用 float 演示，也可换 double 等支持类型
    using DataType = float;  

    // ========== 2. 主机端（CPU）内存分配与初始化 ==========
    DataType *h_input = new DataType[n];
    DataType h_output = 0;  // 主机端结果初始值
    for (size_t i = 0; i < n; ++i) {
        // 简单生成 [0, 1) 随机数初始化
        h_input[i] = static_cast<DataType>(rand()) / RAND_MAX; 
    }

    // ========== 3. 设备端（GPU）内存分配 ==========
    DataType *d_input, *d_output;
    cudaMalloc((void **)&d_input, n * sizeof(DataType));
    cudaMalloc((void **)&d_output, sizeof(DataType));

    // ========== 4. 数据拷贝（主机 -> 设备） ==========
    cudaMemcpy(d_input, h_input, n * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(DataType), cudaMemcpyHostToDevice);

    // ========== 5. 配置核函数执行参数 ==========
    const int blockSize = 256;  // 每个 block 的线程数，需是 32 的倍数（warp 大小）更高效
    // 根据数据量计算需要的 block 数量，向上取整
    const int gridSize = (n + blockSize - 1) / blockSize; 

    // ========== 6. 调用核函数并计时（可选） ==========
    auto start = std::chrono::high_resolution_clock::now();
    reduce_warp_shfl_register_kernel<<<gridSize, blockSize>>>(d_output, d_input, n);
    // 检查核函数调用错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // 等待核函数执行完毕（必要，确保结果正确获取）
    cudaDeviceSynchronize(); 
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Kernel execution time: " << duration.count() << " ms" << std::endl;

    // ========== 7. 数据拷贝（设备 -> 主机），获取结果 ==========
    cudaMemcpy(&h_output, d_output, sizeof(DataType), cudaMemcpyDeviceToHost);

    // ========== 8. 验证结果（CPU 端重新计算对比） ==========
    DataType cpu_result = 0;
    for (size_t i = 0; i < n; ++i) {
        cpu_result += h_input[i];
    }
    std::cout << "GPU result (via warp reduce): " << h_output << std::endl;
    std::cout << "CPU result (for verification): " << cpu_result << std::endl;

    // ========== 9. 释放内存 ==========
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}