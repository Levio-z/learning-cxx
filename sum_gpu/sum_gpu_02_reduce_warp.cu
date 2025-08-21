#include <iostream>
#include <cuda_runtime.h>
// 用于生成随机数初始化数据
#include <cstdlib>  
// 用于计时（可选，可了解核函数执行时间）
#include <chrono>  

template <typename T>
__global__ void reduce_warp_global_kernel(T *output, const T *input, size_t n) {
    // 1. 线程索引与 Warp 内车道 ID
    size_t tid = threadIdx.x;               // 线程在 Block 内的局部 ID
    unsigned int lane_id = tid % 32;       // 线程在 Warp 内的车道号（0~31）
    size_t idx = blockIdx.x * blockDim.x + tid;  // 线程的全局索引（跨 Block）

    // 2. 仅 Warp 内 lane_id=0 的线程参与归约核心逻辑
    if (lane_id == 0) {  
        T warp_sum = 0;  // 存储当前 Warp 归约后的部分和
        // step = 总线程数（Grid 维度 × Block 维度），即线程跨步间隔
        const size_t step = blockDim.x * gridDim.x;  
        // 计算需要处理的“完整 Warp 批次”：向上取整，确保覆盖所有可能的 j < n
        const size_t total_elements = ((n - idx) + step - 1) / step;  

        // 3. 遍历并累加当前 Warp 负责的所有数据
        for (size_t linear_idx = 0; linear_idx < total_elements * 32; ++linear_idx) {
            // segment = 当前处理的“逻辑批次”（每个批次对应 step 跨步）
            const size_t segment = linear_idx / 32;  
            // lane = 模拟 Warp 内其他线程的车道号（0~31）
            const size_t lane = linear_idx % 32;  
            // j = 全局索引：基址 idx + 批次偏移(segment*step) + 车道偏移(lane)
            const size_t j = idx + segment * step + lane;  

            if (j < n) {  // 防止越界（当 n 不是 step 整数倍时）
                warp_sum += input[j];  // 累加数据
            }
        }

        // 4. 原子操作合并结果到全局 output
        atomicAdd(output, warp_sum);  
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
    reduce_warp_global_kernel<<<gridSize, blockSize>>>(d_output, d_input, n);
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