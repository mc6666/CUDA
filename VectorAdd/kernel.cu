#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>

#define N	1024

__global__ void vectorAdd(int* d_a, int* d_b, int* d_c) {
	// 使用第幾個執行緒計算 
	int tid = threadIdx.x;
	//printf("threadIdx = %d\n", tid);
	d_c[tid] = d_a[tid] + d_b[tid];
	// printf("out = %d\n", d_a[tid] + d_b[tid]); 
}

int main(void) {
	// h_：CPU變數(host variable)
	int h_a[N], h_b[N], h_c[N], h2_c[N];
	// d_：GPU變數(device variable)
	int *d_a, *d_b, *d_c;

	// 設定 a、b 的值
	for (int i = 0; i < N; i++) {
		h_a[i] = i;
		h_b[i] = i * 2;
	}

	// 使用 CPU 計算
	clock_t start_d = clock();
	for (int i = 0; i < N; i++) {
		h_c[i] = h_a[i] + h_b[i];
	}
	clock_t end_d = clock();
	printf("CPU execution time = %f\n", (double)(end_d - start_d) / CLOCKS_PER_SEC);

	// 列印結果
	printf("Vector addition on CPU \n");
	for (int i = 0; i < N; i++) {
		//printf(" %d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
	}

	// 複製 a、b 至 GPU
	start_d = clock();

	// 分配記憶體
	cudaMalloc((void**)&d_a, N * sizeof(int));
	// 自 CPU變數 複製到 GPU變數
	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

	// 分配記憶體
	cudaMalloc((void**) &d_c, N * sizeof(int));
	// 使用GPU進行向量相加，1 block x N threads
	vectorAdd <<<1, N >>> (d_a, d_b, d_c);

	// 等待所有執行緒處理完畢
	cudaDeviceSynchronize();

	// 複製 GPU 變數值至 CPU
	cudaMemcpy(h2_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	end_d = clock();
	printf("GPU execution time = %f\n", (double)(end_d - start_d) / CLOCKS_PER_SEC);

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


