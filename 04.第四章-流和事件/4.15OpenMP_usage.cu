#include "common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv)
{
    omp_set_num_threads(3);
    #pragma omp parallel
    {
        printf("thread is running\n");
    }

    return 0;
}
