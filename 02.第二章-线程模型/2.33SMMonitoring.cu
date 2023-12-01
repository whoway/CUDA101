#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>


void initialData(float *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}


void checkResult(float *gpuRef, const int N)
{
    printf("result is:\n");
    for (int i = 0; i < 100; i++)
    {
        if(i%10 == 0)
        {
            printf("\n");
        }
        printf("%.2f\t", gpuRef[i]);
    }
    printf("\n");
}



int main(int argc, char **argv)
{
    int nDeviceNumber = 0;
    int error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != 0 || nDeviceNumber == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    }
    // set up device
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != 0)
    {
        printf("fail to set GPU 0 for computing\n");
        exit(-1);
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

    // malloc host memory
    float *h_A, *h_B, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = GetCPUSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = GetCPUSecond() - iStart;

    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    error = ErrorCheck(cudaMalloc((void **)&d_MatA, nBytes),__FILE__,__LINE__);
    error |= ErrorCheck(cudaMalloc((void **)&d_MatB, nBytes),__FILE__, __LINE__);
    error |= ErrorCheck(cudaMalloc((void **)&d_MatC, nBytes),__FILE__, __LINE__);
    if(error != 0)
    {
        printf("fail to allocate GPU global memory\n");
        free(h_A);
        free(h_B);
        free(gpuRef);
        return -1;
    }

    // transfer data from host to device
    error = ErrorCheck(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice),__FILE__,__LINE__);
    error |= ErrorCheck(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice),__FILE__,__LINE__);
    if(error != 0)
    {
        printf("fail to copy data from host to  GPU global memory\n");
        free(h_A);
        free(h_B);
        free(gpuRef);
        cudaFree(d_MatA);
        cudaFree(d_MatB);
        cudaFree(d_MatC);
        return -1;
    }

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;

    if(argc > 2)
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // execute the kernel
    iStart = GetCPUSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = GetCPUSecond() - iStart;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %.3f ms\n", grid.x, grid.y,block.x, block.y, iElaps*1000);

    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device results
    checkResult(gpuRef, nxy);

    // free device global memory
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // free host memory
    free(h_A);
    free(h_B);
    free(gpuRef);

    // reset device
    cudaDeviceReset();

    return 0;
}
