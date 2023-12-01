#include "../common/common.h"
#include <stdio.h>


#define N   2
#define M   2

__device__ int foo(int row, int col)
{
    return (2 * row);
}

__global__ void kernel(int **arr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i;

    for ( ; tid < N; tid++)
    {
        for (i = 0; i < M; i++)
        {
            arr[5][i] = foo(tid, i);
        }
    }
}

int main(int argc, char **argv)
{
    //get GPU decice count
    int nDeviceNumber = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        return -1;
    }
    // set up device
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to set GPU 0 for computing\n");
        return -1;
    }
    else
    {
        printf("set GPU 0 for computing\n");
    }
    int i;

    int **h_matrix;

    int **d_matrix;

    h_matrix = (int **)malloc(N * sizeof(int *));

    cudaMalloc((void **)&d_matrix, N * sizeof(int *));
    cudaMemset(d_matrix, 0x00, N * sizeof(int *));

    int **d_ptrs;
    d_ptrs = (int **)malloc(N * sizeof(int *));

    for (i = 0; i < N; i++)
    {
        h_matrix[i] = (int *)malloc(M * sizeof(int));
        cudaMalloc((void **)&(d_ptrs[i]), M * sizeof(int));
        cudaMemset(d_ptrs[i], 0x00, M * sizeof(int));
    }
    cudaMemcpy(d_matrix, d_ptrs, N * sizeof(int *),cudaMemcpyHostToDevice);

    int threadsPerBlock = 2;
    int blocksPerGrid = 2;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix);

    // Copy rows back
    for (i = 0; i < N; i++)
    {
        cudaMemcpy(h_matrix[i], d_matrix[i], M * sizeof(int),cudaMemcpyDeviceToHost);
        cudaFree(d_matrix[i]);
        free(h_matrix[i]);
    }

    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}
