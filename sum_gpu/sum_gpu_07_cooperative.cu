#include <iostream>
#include <vector>
#include <cassert>
#include <cooperative_groups.h>  // 引入协作组头文件

namespace cg = cooperative_groups;  // 简化协作组命名空间

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

template <typename T>
__global__ void reduce_cooperative_kernel(T* output, const T* input, size_t n) {
    // 共享内存声明（动态分配，需确保调用时足够空间）
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem_raw[];
    T* coop_smem = reinterpret_cast<T*>(shared_mem_raw);

    // 获取网格、线程块、线程索引等基本信息
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    size_t tid = threadIdx.x;          // 线程块内线程 ID
    size_t bid = blockIdx.x;           // 线程块 ID
    size_t block_size = blockDim.x;    // 线程块大小
    size_t grid_size = gridDim.x;      // 网格大小

    // ========== 1. 线程块级归约（Block-level reduction） ========== 
    T sum = T(0);
    // 跨步遍历全局内存，避免线程发散
    for (size_t i = tid + bid * block_size; i < n; i += block_size * grid_size) {
        sum += input[i];
    }

    // 线程束内归约（利用 warp shuffle 指令）
    T warp_sum = warp_reduce(sum);

    // 每个 warp 选一个代表线程，将 warp 结果存入共享内存
    if (tid % 32 == 0) {
        coop_smem[tid / 32] = warp_sum;
    }
    block.sync();  // 同步确保共享内存写入完成

    // 对共享内存中各 warp 结果，再次归约（仅前 32 线程参与）
    if (tid < 32) {
        // 读取共享内存（若超出 warp 数量则补 0）
        T block_sum = (tid < (block_size + 31) / 32) 
                     ? coop_smem[tid] 
                     : T(0);
        block_sum = warp_reduce(block_sum);  // 最终线程块结果

        // 线程块 0 号线程将结果写入全局内存，供后续全局归约
        if (tid == 0) {
            output[bid] = block_sum;
        }
    }

    // ========== 2. 全局级归约（Global synchronization + Final reduction） ========== 
    grid.sync();  // 全局同步，确保所有线程块完成第一轮归约

    // 仅线程块 0 参与最终全局归约
    if (bid == 0) {
        T final_sum = T(0);
        // 类似线程块归约逻辑，遍历各线程块结果
        for (size_t i = tid; i < grid_size; i += block_size) {
            final_sum += output[i];
        }

        // 线程束内归约
        T warp_val = warp_reduce(final_sum);

        // 存入共享内存（仅前 32 线程操作）
        if (tid % 32 == 0) {
            coop_smem[tid / 32] = warp_val;
        }
        block.sync();  // 同步确保写入完成

        // 最终全局结果（仅 0 号线程写回）
        if (tid < 32) {
            T v = (tid < (block_size + 31) / 32) 
                 ? coop_smem[tid] 
                 : T(0);
            T total = warp_reduce(v);

            // 0 号线程输出最终全局结果
            if (tid == 0) {
                output[0] = total;
            }
        }
    }
}

int main() {
    // 检查设备是否支持协同启动能力
    int can_launch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&can_launch, 
                                 cudaDevAttrCooperativeLaunch, 
                                 0));  // 假设使用设备 0
    if (!can_launch) {
        std::cerr << "Error: Device does not support cooperative launches\n";
        return 1;
    }

    // 设置随机数种子
    srand(42);
    
    // 定义数组大小
    size_t n = 1 << 20;  // 100万+元素
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
    
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // 配置线程块大小
    const int block_size = 256;  // 典型的线程块大小
    
    // 计算每SM可运行的最大线程块数量
    int max_blocks_per_sm;
    size_t shared_mem_size = (block_size / 32 + 1) * sizeof(float);  // 计算共享内存需求
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        reduce_cooperative_kernel<float>,
        block_size,
        shared_mem_size
    ));
    
    // 计算最佳网格大小
    int grid_size = max_blocks_per_sm * prop.multiProcessorCount;
    // 确保网格大小不会超过实际需要的数量
    grid_size = std::min(grid_size, static_cast<int>((n + block_size - 1) / block_size));
    grid_size = std::max(grid_size, 1);
    
    std::cout << "核函数配置 - 线程块大小: " << block_size 
              << ", 网格大小: " << grid_size << std::endl;
    
    // 分配足够大的结果数组（至少为网格大小）
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, grid_size * sizeof(float)));  // 足够存储各线程块结果

    // 数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // 使用cudaLaunchCooperativeKernel启动协作核函数
    void* kernel_args[] = {&d_result, &d_input, &n};
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)reduce_cooperative_kernel<float>,
        grid_size,
        block_size,
        kernel_args,
        shared_mem_size,
        0
    ));
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待核函数完成

    // 结果从设备复制到主机
    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // 输出结果并验证（考虑浮点精度误差）
    std::cout << "CPU计算结果: " << cpu_sum << std::endl;
    std::cout << "GPU计算结果: " << gpu_sum << std::endl;
    std::cout << "结果误差: " << std::abs(gpu_sum - cpu_sum) << std::endl;
    
    // // 验证结果是否在可接受的误差范围内
    // assert(std::abs(gpu_sum - cpu_sum) < 1e-3f);

    // 释放资源
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
