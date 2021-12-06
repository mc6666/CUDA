#include "stdio.h"
#include<iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#define A_ROW_SIZE 5
#define A_COLUMN_SIZE 4
#define B_ROW_SIZE 4
#define B_COLUMN_SIZE 3

__global__ void gpu_inner_product(float* d_a, float* d_b, float* d_c)
{
	// 輸出所在格子的座標
	int row = threadIdx.x;
	int col = threadIdx.y;

	// // 點積
	// printf("row=%d, col=%d\n", row, col);
	for (int k = 0; k < A_COLUMN_SIZE; k++)
	{
		// 第一個輸入矩陣的【列】與第二個輸入矩陣的【行】相乘
		d_c[row * B_COLUMN_SIZE + col] += d_a[row * A_COLUMN_SIZE + k] * d_b[k * B_COLUMN_SIZE + col];
		//printf("result=%d, A=%d, B=%d\n", row * B_COLUMN_SIZE + col, row * A_COLUMN_SIZE + k, k * B_COLUMN_SIZE + col);
		//__syncthreads();
	}

	////Defining Shared Memory
	//__shared__ float shared_a[TILE_SIZE][TILE_SIZE];
	//__shared__ float shared_b[TILE_SIZE][TILE_SIZE];
	//col = TILE_SIZE * blockIdx.x + threadIdx.x;
	//row = TILE_SIZE * blockIdx.y + threadIdx.y;


	//		for (int j = 0; j < A_COLUMN_SIZE; j++)
	//			d_c[row * size + col] += shared_a[threadIdx.x][j] * shared_b[j][threadIdx.y];
	//		__syncthreads();
	//	}
	//}
}

__global__ void gpu_inner_product_shared(float* d_a, float* d_b, float* d_c)
{
	// 輸出所在格子的座標
	int row = threadIdx.x;
	int col = threadIdx.y;

	//Defining Shared Memory
	__shared__ float shared_a[A_ROW_SIZE][A_COLUMN_SIZE];
	__shared__ float shared_b[B_ROW_SIZE][B_COLUMN_SIZE];
	for (int k = 0; k < A_COLUMN_SIZE; k++)
	{
		shared_a[row][k] = d_a[row * A_COLUMN_SIZE + k];
	}
	for (int k = 0; k < B_ROW_SIZE; k++)
	{
		shared_b[k][col] = d_b[k * B_COLUMN_SIZE + col];
		//__syncthreads();
	}

	// // 點積
	// printf("row=%d, col=%d\n", row, col);
	for (int k = 0; k < A_COLUMN_SIZE; k++)
	{
		// 第一個輸入矩陣的【列】與第二個輸入矩陣的【行】相乘
		d_c[row * B_COLUMN_SIZE + col] += shared_a[row][k] * shared_b[k][col];
		//printf("result=%d, A=%d, B=%d\n", row * B_COLUMN_SIZE + col, row * A_COLUMN_SIZE + k, k * B_COLUMN_SIZE + col);
		//__syncthreads();
	}
}

// main routine
int main()
{
	//Define Host Array
	float h_a[A_ROW_SIZE][A_COLUMN_SIZE], h_b[B_ROW_SIZE][B_COLUMN_SIZE], h_result[A_ROW_SIZE][B_COLUMN_SIZE];
	//Defining device Array
	float* d_a, * d_b, * d_result;

	// 設定 a
	printf("a Matrix: \n");
	for (int i = 0; i < A_ROW_SIZE; i++)
	{
		for (int j = 0; j < A_COLUMN_SIZE; j++)
		{
			h_a[i][j] = i;
			printf("%d   ", (int)h_a[i][j]);
		}
		printf("\n");
	}
	// 設定 b 
	printf("b Matrix: \n");
	for (int i = 0; i < B_ROW_SIZE; i++)
	{
		for (int j = 0; j < B_COLUMN_SIZE; j++)
		{
			h_b[i][j] = j;
			printf("%d   ", (int)h_b[i][j]);
		}
		printf("\n");
	}

	// 分配 GPU 記憶體
	cudaMalloc((void**)&d_a, A_ROW_SIZE * A_COLUMN_SIZE * sizeof(int));
	cudaMalloc((void**)&d_b, B_ROW_SIZE * B_COLUMN_SIZE * sizeof(int));
	cudaMalloc((void**)&d_result, A_ROW_SIZE * B_COLUMN_SIZE * sizeof(int));


	// CPU 變數 複製至 GPU 變數
	cudaMemcpy(d_a, h_a, A_ROW_SIZE * A_COLUMN_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, B_ROW_SIZE * B_COLUMN_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	// 定義 Grid、Block
	// dim3 dimGrid(A_ROW_SIZE, A_COLUMN_SIZE, 1);
	dim3 dimBlock(A_ROW_SIZE, B_COLUMN_SIZE, 1);

	//gpu_inner_product <<<1, dimBlock >>> (d_a, d_b, d_result);
	gpu_inner_product_shared <<<1, dimBlock >>> (d_a, d_b, d_result);

	// 將結果複製至 CPU 變數
	cudaMemcpy(h_result, d_result, A_ROW_SIZE * B_COLUMN_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	printf("The result of Matrix multiplication is: \n");

	for (int i = 0; i < A_ROW_SIZE; i++)
	{
		for (int j = 0; j < B_COLUMN_SIZE; j++)
		{
			printf("%d   ", (int)h_result[i][j]);
		}
		printf("\n");
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
	return 0;
}
