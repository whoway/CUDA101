#pragma once

#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>

// whoway
#define ErrorCheck(call)                                                                  \
    do                                                                               \
    {                                                                                \
        const cudaError_t error_code = call;                                         \
        if (error_code != cudaSuccess)                                               \
        {                                                                            \
            fprintf(stderr, "CUDA Error:\n");                                        \
            fprintf(stderr, "    File:       %s\n", __FILE__);                       \
            fprintf(stderr, "    Line:       %d\n", __LINE__);                       \
            fprintf(stderr, "    Error code: %d\n", error_code);                     \
            fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define CHECK(call)                                                                  \
    do                                                                               \
    {                                                                                \
        const cudaError_t error_code = call;                                         \
        if (error_code != cudaSuccess)                                               \
        {                                                                            \
            fprintf(stderr, "CUDA Error:\n");                                        \
            fprintf(stderr, "    File:       %s\n", __FILE__);                       \
            fprintf(stderr, "    Line:       %d\n", __LINE__);                       \
            fprintf(stderr, "    Error code: %d\n", error_code);                     \
            fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

#define CHECK_FILE_OPEN(call)                                  \
    do                                                         \
    {                                                          \
        FILE *error_code = call;                               \
        if (error_code == NULL)                                \
        {                                                      \
            fprintf(stderr, "OPEN FILE Error:\n");             \
            fprintf(stderr, "    File:       %s\n", __FILE__); \
            fprintf(stderr, "    Line:       %d\n", __LINE__); \
            exit(1);                                           \
        }                                                      \
    } while (0)

#define PRINT_SCANF_ERROR(count, n, text)                      \
    do                                                         \
    {                                                          \
        if (count != n)                                        \
        {                                                      \
            fprintf(stderr, "Input Error:\n");                 \
            fprintf(stderr, "    File:       %s\n", __FILE__); \
            fprintf(stderr, "    Line:       %d\n", __LINE__); \
            fprintf(stderr, "    Error text: %s\n", text);     \
            exit(1);                                           \
        }                                                      \
    } while (0)

#define PRINT_INPUT_ERROR(text)                            \
    do                                                     \
    {                                                      \
        fprintf(stderr, "Input Error:\n");                 \
        fprintf(stderr, "    File:       %s\n", __FILE__); \
        fprintf(stderr, "    Line:       %d\n", __LINE__); \
        fprintf(stderr, "    Error text: %s\n", text);     \
        exit(1);                                           \
    } while (0)

#define PRINT_KEYWORD_ERROR(keyword)                                               \
    do                                                                             \
    {                                                                              \
        fprintf(stderr, "Input Error:\n");                                         \
        fprintf(stderr, "    File:       %s\n", __FILE__);                         \
        fprintf(stderr, "    Line:       %d\n", __LINE__);                         \
        fprintf(stderr, "    Error text: '%s' is an invalid keyword.\n", keyword); \
        exit(1);                                                                   \
    } while (0)

#ifdef STRONG_DEBUG
#define CUDA_CHECK_KERNEL               \
    {                                   \
        CHECK(cudaGetLastError());      \
        CHECK(cudaDeviceSynchronize()); \
    }
#else
#define CUDA_CHECK_KERNEL          \
    {                              \
        CHECK(cudaGetLastError()); \
    }
#endif