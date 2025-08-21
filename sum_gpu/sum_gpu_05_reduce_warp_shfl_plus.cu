#include <iostream>
#include <cuda_runtime.h>
// 用于生成随机数初始化数据
#include <cstdlib>  
// 用于计时（可选，可了解核函数执行时间）
#include <chrono>  

template<typename T>
__device__ T wrap_reduce(T val){
#pragma unroll
     // 基于 warp shuffle 指令的归约操作
     for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }    
    return val;
}

template <typename T>
__global__ void reduce_warp_shfl_register_kernel(T* output, const T* input, size_t n) {
    extern __shared__ T smem[];
    // 当前线程在一个 block 内的一维索引。
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;  

    T sum = 0;

    // 线程束内跨线程加载数据累加
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // 所有线程按照线程束规约
    T wrap_sum = wrap_reduce(sum);

    // 线程束规约结果移动到共享内存
    if (tid % 32 == 0){
        smem[tid /32] = wrap_sum;
    }
    
    __syncthreads();
    // 一个 block 最多 32 个 warp → 最多 32 个部分和 → 最多需要 32 个线程来做跨 warp 的归约。
    if (tid < 32){
        // 一个 block 里可能有很多 warp,如果 blockDim.x = 128，那么就有 128 / 32 = 4 个 warp。
        T block_sum  = ( tid < (blockDim.x + 31) /32 )? smem[tid] : T(0);
        // 对wrap内规约
        block_sum = wrap_reduce(block_sum);
        // 每个块规约
        if (tid==0){
            atomicAdd(output,block_sum);
        }
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
    size_t smem_size = ((blockSize + 31) / 32) * sizeof(DataType);  // 共享内存大小 = Block 线程数 × 数据类型大小

    // ========== 6. 调用核函数并计时（可选） ==========
    auto start = std::chrono::high_resolution_clock::now();
    reduce_warp_shfl_register_kernel<<<gridSize, blockSize, smem_size>>>(d_output, d_input, n);
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