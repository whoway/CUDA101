#include <sys/time.h>
#include <cuda_runtime.h>
#include <stdio.h>

cudaError_t ErrorCheck(cudaError_t status,
                       const char *filename, int lineNumber)
{
    if (status != cudaSuccess)
    {
        printf("CUDA API error:\r\n");
        printf("code=%d, name=%s, description=%s\r\n",
               status, cudaGetErrorName(status), cudaGetErrorString(status));
        printf("File: %s, Line=%d\r\n", filename, lineNumber);
    }

    return status;
}