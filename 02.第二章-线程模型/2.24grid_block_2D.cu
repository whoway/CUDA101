#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sumMatrixOnGPU2D(int *MatA, int *MatB, int *MatC, int nx,int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}


int main(int argc, char **argv)
{
    if(argc != 3)
        return -1;
    int block_x = atoi(argv[1]);
    int block_y = atoi(argv[2]);

    int nDeviceNumber = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0)
    {
        printf("No CUDA compatable GPU found!\n");
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
    // set matrix dimension
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    // malloc host memory
    int *h_A, *h_B,*gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    // iniitialize host matrix with integer
    for (int i = 0; i < nxy; i++)
    {
        h_A[i] = i;
        h_B[i] = i + 1;
    }


    // malloc device memory
    int *d_MatA, *d_MatB,*d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    //copy data to GPU memory
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    //make each row data excuted in each block
    dim3 block(block_x, block_y);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // invoke the kernel
    double dTime_Begin = GetCPUSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC,nx, ny);
    cudaDeviceSynchronize();
    double dTime_End = GetCPUSecond();
    printf("Element Size:%d, Matrix add time Elapse is:%.5f\n", nxy, dTime_End - dTime_Begin);
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++)
    {
        printf("idx=%d, matrix_A:%d, matrix_B:%d, result=%d\n", i+1, h_A[i],h_B[i],gpuRef[i] );
    }
    // free host and devide memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    free(h_A);
    free(h_B);
    free(gpuRef);
    // reset device
    cudaDeviceReset();
    return 0;
}
