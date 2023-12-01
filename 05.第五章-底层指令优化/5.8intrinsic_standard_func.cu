#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void standard_kernel(float a, float *out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid == 0)
    {
        float tmp;

        for (i = 0; i < iters; i++)
        {
            tmp = powf(a, 2.0f);
        }

        *out = tmp;
    }
}

__global__ void intrinsic_kernel(float a, float *out, int iters)
{
    int i;
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(tid == 0)
    {
        float tmp;

        for (i = 0; i < iters; i++)
        {
            tmp = __powf(a, 2.0f);
        }

        *out = tmp;
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
    int iters = 1000;
    float input_value = 0.0f;
    if(argc == 3)
    {
        iters = atoi(argv[2]);
        input_value = (float)atof(argv[1]);
    }
    printf("iteration count is:%d\n", iters);
    printf("input value is:%.5f\n", input_value);
    int i;
    int runs = 30;



    float *d_standard_out, h_standard_out;
    cudaMalloc((void **)&d_standard_out, sizeof(float));

    float *d_intrinsic_out, h_intrinsic_out;
    cudaMalloc((void **)&d_intrinsic_out, sizeof(float));



    double mean_intrinsic_time = 0.0;
    double mean_standard_time = 0.0;

    for (i = 0; i < runs; i++)
    {
        double start_standard = GetCPUSecond();
        standard_kernel<<<1, 32>>>(input_value, d_standard_out, iters);
        cudaDeviceSynchronize();
        mean_standard_time += GetCPUSecond() - start_standard;

        double start_intrinsic = GetCPUSecond();
        intrinsic_kernel<<<1, 32>>>(input_value, d_intrinsic_out, iters);
        cudaDeviceSynchronize();
        mean_intrinsic_time += GetCPUSecond() - start_intrinsic;
    }

    cudaMemcpy(&h_standard_out, d_standard_out, sizeof(float),
                     cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float),
                     cudaMemcpyDeviceToHost);
    float host_value = powf(input_value, 2.0f);

    printf("Host calculated\t\t\t%f\n", host_value);
    printf("Standard Device calculated\t%f\n", h_standard_out);
    printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
    printf("Host equals Standard?\t\t%s diff=%e\n",
           host_value == h_standard_out ? "Yes" : "No",
           fabs(host_value - h_standard_out));
    printf("Host equals Intrinsic?\t\t%s diff=%e\n",
           host_value == h_intrinsic_out ? "Yes" : "No",
           fabs(host_value - h_intrinsic_out));
    printf("Standard equals Intrinsic?\t%s diff=%e\n",
           h_standard_out == h_intrinsic_out ? "Yes" : "No",
           fabs(h_standard_out - h_intrinsic_out));
    printf("\n");
    printf("Mean execution time for standard function powf:    %f s\n",
           mean_standard_time);
    printf("Mean execution time for intrinsic function __powf: %f s\n",
           mean_intrinsic_time);
    cudaFree(d_standard_out);
    cudaFree(d_intrinsic_out);
    cudaDeviceReset();

    return 0;
}
