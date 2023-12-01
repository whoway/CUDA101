#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void atomics(int *shared_var, int *values_read, int N, int iters)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    values_read[tid] = atomicAdd(shared_var, 1);

    for (i = 0; i < iters; i++)
    {
        atomicAdd(shared_var, 1);
    }
}


__global__ void unsafe(int *shared_var, int *values_read, int N, int iters)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    int old = *shared_var;
    *shared_var = old + 1;
    values_read[tid] = old;

    for (i = 0; i < iters; i++)
    {
        int old = *shared_var;
        *shared_var = old + 1;
    }
}

static void print_read_results(int *h_arr, int *d_arr, int N,
                               const char *label)
{
    int i;
    cudaMemcpy(h_arr, d_arr, N * sizeof(int),
                     cudaMemcpyDeviceToHost);
    printf("Threads performing %s operations read values", label);

    for (i = 0; i < N; i++)
    {
        printf(" %d", h_arr[i]);
    }

    printf("\n");
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
    int N = 64;
    int block = 32;
    int runs = 30;
    int iters = 100000;
    int r;
    int *d_shared_var;
    int h_shared_var_atomic, h_shared_var_unsafe;
    int *d_values_read_atomic;
    int *d_values_read_unsafe;
    int *h_values_read;

    cudaMalloc((void **)&d_shared_var, sizeof(int));
    cudaMalloc((void **)&d_values_read_atomic, N * sizeof(int));
    cudaMalloc((void **)&d_values_read_unsafe, N * sizeof(int));
    h_values_read = (int *)malloc(N * sizeof(int));

    double atomic_mean_time = 0;
    double unsafe_mean_time = 0;

    for (r = 0; r < runs; r++)
    {
        double start_atomic = GetCPUSecond();
        cudaMemset(d_shared_var, 0x00, sizeof(int));
        atomics<<<N / block, block>>>(d_shared_var, d_values_read_atomic, N,
                                          iters);
        cudaDeviceSynchronize();
        atomic_mean_time += GetCPUSecond() - start_atomic;
        cudaMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int),
                         cudaMemcpyDeviceToHost);

        double start_unsafe = GetCPUSecond();
        cudaMemset(d_shared_var, 0x00, sizeof(int));
        unsafe<<<N / block, block>>>(d_shared_var, d_values_read_unsafe, N,
                                         iters);
        cudaDeviceSynchronize();
        unsafe_mean_time += GetCPUSecond() - start_unsafe;
        cudaMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int),
                         cudaMemcpyDeviceToHost);
    }

    printf("In total, %d runs using atomic operations took %f s\n",
           runs, atomic_mean_time);
    printf("  Using atomic operations also produced an output of %d\n",
           h_shared_var_atomic);
    printf("In total, %d runs using unsafe operations took %f s\n",
           runs, unsafe_mean_time);
    printf("  Using unsafe operations also produced an output of %d\n",
           h_shared_var_unsafe);

    print_read_results(h_values_read, d_values_read_atomic, N, "atomic");
    print_read_results(h_values_read, d_values_read_unsafe, N, "unsafe");
    cudaFree(d_shared_var);
    cudaFree(d_values_read_atomic);
    cudaFree(d_values_read_unsafe);
    free(h_values_read);
    cudaDeviceReset();

    return 0;
}
