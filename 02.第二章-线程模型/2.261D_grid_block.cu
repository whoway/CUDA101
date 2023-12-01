#include "common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *ip, const int size)
{
    int i;
    for(i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return;
}

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int nx,int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx )
    {
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

void checkResult(float *gpuRef, const int N)
{
    printf("result is:\n");
    for (int i = 0; i < N; i++)
    {
        if(i%10 == 0)
        {
            printf("\n");
        }
        printf("%.2f\t", gpuRef[i]);
    }
    printf("\n");
}


double GetCPUSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
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
    int nx = 1 << 12; //4096 elements
    int ny = 1 << 12;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    //initialize data
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    error = ErrorCheck(cudaMalloc((void **)&d_MatA, nBytes), __FILE__, __LINE__);
    error |= ErrorCheck(cudaMalloc((void **)&d_MatB, nBytes), __FILE__, __LINE__);
    error  |= ErrorCheck(cudaMalloc((void **)&d_MatC, nBytes), __FILE__, __LINE__);
    if(error != 0)
    {
        printf("fail to allocate GPU global memory\n");
        free(h_A);
        free(h_B);
        free(gpuRef);
    }

    // transfer data from host to device
    error = ErrorCheck(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    error |= ErrorCheck(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
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
    // set thread configuration
    dim3 block(32,1);
    dim3 grid((nx+block.x-1)/block.x,1);

    double start_Time = GetCPUSecond();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    double iElaps = GetCPUSecond() - start_Time;

    // copy kernel result back to host side
    error = ErrorCheck(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost),__FILE__, __LINE__);
    if(error != 0)
    {
        printf("fail to copy GPU data to host memory\n");
        free(h_A);
        free(h_B);
        free(gpuRef);
        cudaFree(d_MatA);
        cudaFree(d_MatB);
        cudaFree(d_MatC);
        return -1;
    }

    // check device results
    //checkResult(gpuRef, nxy);

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
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
    grid.y, block.x, block.y, iElaps);

    return (0);
}
