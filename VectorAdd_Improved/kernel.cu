#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>

#define N	1025

__global__ void vectorAdd(int* d_a, int* d_b, int* d_c) {
	// 使用第幾個執行緒計算 
	// blockIdx.x：第幾個block，blockDim.x：block內的執行緒數量
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	printf("tid = %d\n", tid);
	d_c[tid] = d_a[tid] + d_b[tid];
	// printf("out = %d\n", d_a[tid] + d_b[tid]); 
}

cudaDeviceProp get_GPU_property() {
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	
	// get first GPU properties
	int dev = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	return deviceProp;
}

void init_val(int *h_a, int *h_b) {
	// 設定 a、b 的值
	for (int i = 0; i < N; i++) {
		h_a[i] = i;
		h_b[i] = i * 2;
	}

}

int main(void) {
	// h_：CPU變數(host variable)
	int h_a[N], h_b[N], h_c[N], h2_c[N];
	// d_：GPU變數(device variable)
	int* d_a, * d_b, * d_c;

	init_val(h_a, h_b);

	cudaDeviceProp deviceProp;
	deviceProp = get_GPU_property();

	int m = (int)ceil((double)N / deviceProp.maxThreadsPerBlock);
	printf("blockCount = %d\n", m);
	printf("Maximum number of threads per multiprocessor:  %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf("Maximum number of threads per block:           %d\n",
		deviceProp.maxThreadsPerBlock);
	printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);

	// 複製 a、b 至 GPU
	clock_t start_d = clock();

	// 分配記憶體
	cudaMalloc((void**)&d_a, N * sizeof(int));
	// 自 CPU變數 複製到 GPU變數
	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

	// 分配記憶體
	cudaMalloc((void**)&d_c, N * sizeof(int));
	// 使用GPU進行向量相加，m block x N threads
	vectorAdd << <m, deviceProp.maxThreadsPerBlock >> > (d_a, d_b, d_c);

	// 等待所有執行緒處理完畢
	cudaDeviceSynchronize();

	// 複製 GPU 變數值至 CPU
	cudaMemcpy(h2_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	clock_t end_d = clock();
	printf("GPU execution time = %f\n", (double)(end_d - start_d) / CLOCKS_PER_SEC);

	cudaError_t cudaStatus;
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// 列印結果
	printf("Vector addition on GPU \n");
	for (int i = 0; i < N; i++) {
		printf(" %d + %d = %d\n", h_a[i], h_b[i], h2_c[i]);
	}
	// 釋放記憶體
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}


