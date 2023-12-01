#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// grid 2D block 1D
__global__ void sumMatrixOnGPUMix(int *MatA, int *MatB, int *MatC, int nx,
                                  int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char **argv)
{
    if(argc != 2)
        return -1;
    int block_x = atoi(argv[1]);

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

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    int *h_A, *h_B, *gpuRef;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);
    // iniitialize host matrix with integer
    for (int i = 0; i < nxy; i++)
    {
        h_A[i] = i;
        h_B[i] = i + 1;
    }
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    ErrorCheck(cudaMalloc((void **)&d_MatA, nBytes),__FILE__, __LINE__);
    ErrorCheck(cudaMalloc((void **)&d_MatB, nBytes),__FILE__, __LINE__);
    ErrorCheck(cudaMalloc((void **)&d_MatC, nBytes),__FILE__, __LINE__);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    dim3 block(block_x, 1);
    dim3 grid((nx + block.x - 1) / block.x, ny);

    double dTime_Begin = GetCPUSecond();
    sumMatrixOnGPUMix<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    double dTime_End = GetCPUSecond();
    printf("Element Size:%d, Matrix add time Elapse is:%.5f\n", nxy, dTime_End - dTime_Begin);

    ErrorCheck(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
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
