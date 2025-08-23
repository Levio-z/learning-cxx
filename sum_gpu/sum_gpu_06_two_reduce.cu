#include <iostream>
#include <vector>
#include <cassert>

// 错误检查宏
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err_) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 线程束内归约函数
template <typename T>
__device__ T warp_reduce(T val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 第一阶段核函数：每个线程块计算部分和
template <typename T>
__global__ void reduce_first_pass(T* output, const T* input, size_t n) {
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;

    // 加载数据并进行线程级归约
    T sum = 0;
    // 线程束内跨线程加载数据累加
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // 线程束内归约
    T warp_sum = warp_reduce(sum);

    // 线程束规约结果移动到共享内存
    if (tid % 32 == 0){
        smem[tid /32] = warp_sum;
    }
    

    __syncthreads();

    // 一个 block 最多 32 个 warp → 最多 32 个部分和 → 最多需要 32 个线程来做跨 warp 的归约。
    if (tid < 32){
        // 一个 block 里可能有很多 warp,如果 blockDim.x = 128，那么就有 128 / 32 = 4 个 warp。
        T block_sum  = ( tid < (blockDim.x + 31) /32 )? smem[tid] : T(0);
        // 对wrap内规约
        block_sum = warp_reduce(block_sum);
        // 每个块规约
        if (tid==0){
            atomicAdd(output,block_sum);
        }
    }
}

// 第二阶段核函数：合并所有线程块的结果
template <typename T>
__global__ void reduce_second_pass(T* output, const T* intermediate, size_t n) {
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;

    // 加载数据并进行线程级归约
    T sum = 0;
    // 线程束内跨线程加载数据累加
    for (size_t i = tid; i < n; i += blockDim.x) {
        sum += intermediate[i];
    }

    // 线程束内归约
    T warp_sum = warp_reduce(sum);

    // 线程束规约结果移动到共享内存
    if (tid % 32 == 0){
        smem[tid /32] = warp_sum;
    }
    

    __syncthreads();

    // 一个 block 最多 32 个 warp → 最多 32 个部分和 → 最多需要 32 个线程来做跨 warp 的归约。
    if (tid < 32){
        // 一个 block 里可能有很多 warp,如果 blockDim.x = 128，那么就有 128 / 32 = 4 个 warp。
        T block_sum  = ( tid < (blockDim.x + 31) /32 )? smem[tid] : T(0);
        // 对wrap内规约
        block_sum = warp_reduce(block_sum);
        // 每个块规约
        if (tid==0){
            atomicAdd(output,block_sum);
        }
    }
}

// 两阶段归约主机端函数
template <typename T>
void reduce_two_pass(T* d_result, const T* d_input, size_t n) {
    // 配置线程块和网格大小
    const dim3 block(256);  // 每个线程块256个线程
    const dim3 grid((n + block.x - 1) / block.x);  // 计算所需线程块数量

    // 分配中间结果缓冲区
    T* d_intermediate;
    // grid.x 整个核函数启动时的线程块（block）数量
    CUDA_CHECK(cudaMalloc(&d_intermediate, grid.x * sizeof(T)));

    // 第一阶段归约：每个线程块计算部分和
    const size_t smem_size1 = ((block.x + 31) / 32) * sizeof(T);
    reduce_first_pass<<<grid, block, smem_size1>>>(d_intermediate, d_input, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 第二阶段归约：合并所有部分和
    // 线程数量
    // 若第一阶段的线程块数量（grid.x）≤256，则 block2.x 等于 grid.x（刚好足够处理所有中间结果）。
    const dim3 block2(min(block.x, grid.x));
    const dim3 grid2(1);
    const size_t smem_size2 = ((block2.x + 31) / 32) * sizeof(T);
    reduce_second_pass<<<grid2, block2, smem_size2>>>(d_result, d_intermediate, grid.x);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放资源
    CUDA_CHECK(cudaFree(d_intermediate));
}

int main() {
    // 设置随机数种子
    srand(42);
    
    // 定义数组大小
    const size_t n = 1 << 20;  // 100万+元素
    std::cout << "数组大小: " << n << " 元素" << std::endl;

    // 主机端内存分配和初始化
    std::vector<float> h_input(n);
    for (size_t i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;  // 0到1之间的随机数
    }

    // 计算CPU上的结果作为验证
    float cpu_sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += h_input[i];
    }

    // 设备端内存分配
    float* d_input;
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

    // 数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // 执行GPU归约
    reduce_two_pass(d_result, d_input, n);

    // 结果从设备复制到主机
    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 输出结果并验证
    std::cout << "CPU计算结果: " << cpu_sum << std::endl;
    std::cout << "GPU计算结果: " << gpu_sum << std::endl;

    // 释放资源
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}