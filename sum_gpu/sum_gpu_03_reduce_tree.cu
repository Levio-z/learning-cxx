#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>  
#include <chrono>  

template <typename T>
__global__ void reduce_smem_tree_kernel(T *output, const T *input, size_t n) {
    // 共享内存声明（外部共享内存，编译时由核函数参数指定大小）
    extern __shared__ T smem[];  
    size_t tid = threadIdx.x;             
    size_t idx = blockIdx.x * blockDim.x + tid;  

    // 1. 加载全局内存到共享内存
    smem[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();  // 确保所有线程完成加载

    // 2. 共享内存树状归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();  // 同步确保阶段内结果正确
    }

    // 3. 单个 Block 归约完成后，用原子操作合并到全局 output
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

int main() {
    // ========== 1. 配置参数与数据类型 ==========
    const size_t n = 1 << 20;  
    using DataType = float;  

    // ========== 2. 主机端（CPU）内存分配与初始化 ==========
    DataType *h_input = new DataType[n];
    DataType h_output = 0;  
    for (size_t i = 0; i < n; ++i) {
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
    const int blockSize = 256;  // 需是合理的 Block 大小（通常 128/256/512/1024）
    const int gridSize = (n + blockSize - 1) / blockSize; 
    size_t smem_size = blockSize * sizeof(DataType);  // 共享内存大小 = Block 线程数 × 数据类型大小

    // ========== 6. 调用核函数并计时（可选） ==========
    auto start = std::chrono::high_resolution_clock::now();
    reduce_smem_tree_kernel<<<gridSize, blockSize, smem_size>>>(d_output, d_input, n);

    // 检查核函数启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();  // 等待核函数执行完毕
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
    std::cout << "GPU result (via shared mem tree reduce): " << h_output << std::endl;
    std::cout << "CPU result (for verification): " << cpu_result << std::endl;

    // ========== 9. 释放内存 ==========
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}