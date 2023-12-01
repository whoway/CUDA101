#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling 8
    if(idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2* blockDim.x];
        int a4 = g_idata[idx + 3* blockDim.x];
        int b1 = g_idata[idx + 4* blockDim.x];
        int b2 = g_idata[idx + 5* blockDim.x];
        int b3 = g_idata[idx + 6* blockDim.x];
        int b4 = g_idata[idx + 7* blockDim.x];
        g_idata[idx] = a1 + a2 + a3+ a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();
    // in-place reduction and complete unroll
    if (iBlockSize>=1024 && tid < 512) idata[tid] += idata[tid + 512];
        __syncthreads();
    if (iBlockSize>=512 && tid < 256) idata[tid] += idata[tid + 256];
        __syncthreads();
    if (iBlockSize>=256 && tid < 128) idata[tid] += idata[tid + 128];
        __syncthreads();
    if (iBlockSize>=128 && tid < 64) idata[tid] += idata[tid + 64];
    __syncthreads();
    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char **argv)
{
    int blocksize = atoi(argv[1]);
    int nDeviceNumber = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        return (-1);
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

    // initialization
    int size = 1 << 24; // total number of elements
    printf("    with array size %d  ", size);

    
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }
    memcpy (tmp, h_idata, bytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    iStart = GetCPUSecond();
    switch (blocksize)
    {
        case 1024:
            reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 512:
            reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 256:
            reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 128:
            reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
        case 64:
            reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size);
            break;
    }
    cudaDeviceSynchronize();
    iElaps = GetCPUSecond() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);
    free(tmp);

    // free device memory
    cudaFree(d_idata);
    cudaFree(d_odata);

    // reset device
    cudaDeviceReset();
    return 0;
}
