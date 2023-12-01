#include <cuda_runtime.h>
#include "common/common.h"
#include <stdio.h>

extern __shared__ int dynamic_array[];

__global__ void dynamic_shared_mem()
{
    dynamic_array[threadIdx.x] = threadIdx.x;
    printf("access dynamic_array in kernel, dynamic_array[%d]=%d\n", threadIdx.x, dynamic_array[threadIdx.x]);
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
    //config shared memory
    cudaFuncCache cacheConfig = cudaFuncCachePreferEqual;
    ErrorCheck(cudaDeviceSetCacheConfig(cacheConfig), __FILE__, __LINE__);
    cacheConfig = cudaFuncCachePreferShared;
    ErrorCheck(cudaFuncSetCacheConfig(dynamic_shared_mem, cacheConfig), __FILE__, __LINE__);

    //get CacheConfig
    ErrorCheck(cudaDeviceGetCacheConfig(&cacheConfig), __FILE__, __LINE__);
    printf("current cache config for device:%d\n", cacheConfig);
    cudaDeviceReset();

    return 0;
}
