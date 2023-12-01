#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>


void printMatrix(int *C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%3d\t", C[iy*nx+ix]);
        }
        printf("\n");
    }

    printf("\n");
    return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"
           " %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
           ix, iy, idx, A[idx]);
}

int main(int argc, char **argv)
{
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
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // iniitialize host matrix with integer
    for (int i = 0; i < nxy; i++)
    {
        h_A[i] = i;
    }
    printMatrix(h_A, nx, ny);

    // malloc device memory
    int *d_MatA;
    error = ErrorCheck(cudaMalloc((void **)&d_MatA, nBytes), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to allocate memory for GPU\n");
        free(h_A);
        return -1;
    }

    // transfer data from host to device
    error = ErrorCheck(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    if(error != cudaSuccess)
    {
        printf("fail to copy data from host to GPU\n");
        free(h_A);
        cudaFree(d_MatA);
        return -1;
    }

    // set up execution configuration
    //make each row data excuted in each block
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // invoke the kernel
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();


    // free host and devide memory
    cudaFree(d_MatA);
    free(h_A);
    // reset device
    cudaDeviceReset();

    return 0;
}
