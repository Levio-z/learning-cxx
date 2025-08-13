#include <stdio.h>
#include <iostream>
#include <vector>

// Kernel
template<typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t n, size_t step) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x+step;
    for(size_t i =idx;i<n;i+=step){
        c[i] = a[i] + b[i];
    }
}

template<typename T>
void vector_add(T* c,const T *a, const T *b, size_t n,const dim3& grid,const dim3& block){
    size_t step = grid.x * block.x;
    add_kernel<T><<<grid,block>>>(c,a,b,n,step);
}


// Error checking macro
#define CUDA_CHECK(call)                                             \
{                                                                    \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess)                                          \
{                                                                    \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " - " << cudaGetErrorString(err) << "\n";         \
        exit(1);                                                     \
}                                                                    \
}

// Main program
int main()
{
    // 1.Prepare and initialize data (CPU)
    const size_t SIZE = 1<<20;
	size_t size_bytes = SIZE*sizeof(float);

	// Allocate memory for arrays A, B, and C on host
    // 1.1 by malloc
	// double *A = (double*)malloc(bytes);
	// double *B = (double*)malloc(bytes);
	// double *C = (double*)malloc(bytes);

    // 1.2 by vector
    std::vector<float> h_a(SIZE,1);
    std::vector<float> h_b(SIZE,2);
    std::vector<float> h_c(SIZE,0);

    // 2.Transfer data to GPU
	// 2.1 Allocate memory for arrays d_a, d_b, and d_c on device
	float *d_a, *d_b, *d_c;
	CUDA_CHECK(cudaMalloc(&d_a, size_bytes));
	CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
	CUDA_CHECK(cudaMalloc(&d_c, size_bytes));

	// 2.2 Copy data from host to device arrays 
	CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));

	// Set execution configuration parameters
	//		grid_dim: number of CUDA threads per grid block
	//		block_dim: number of blocks in grid
	dim3 block_dim(256);
	dim3 grid_dim(256);


    // 3.GPU reads from global memory, performs computation, and writes back (invoke computation function)
	// call the cuda add kernel
	vector_add( d_c,d_a, d_b,SIZE,grid_dim,block_dim);

    // 4.Transfer data back from GPU to CPU
	// Copy data from device 
	CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));

    if (d_a)
        CUDA_CHECK(cudaFree(d_a));
    if (d_b)
        CUDA_CHECK(cudaFree(d_b));
    if (d_c)
        CUDA_CHECK(cudaFree(d_c));
	// 5.Verify results
    std::cout << "执行完毕" << '\n'; 
    std::cout <<"c[SIZE-1]:"<< h_c[SIZE-1] << '\n'; 
	return 0;
}
