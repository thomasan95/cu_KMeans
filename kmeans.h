#ifndef _GPU_KMEANS_H
#define _GPU_KMEANS_H

#include <assert.h>

typedef float km_float; 


// Allocate a 2D array with pointers set spaced by dimension of data
#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

inline void CHECK(cudaError_t e) {
    if (e != cudaSuccess) {

        printf("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

struct parameters
{
	const int numSamples = 10000;
	const int dim = 784;
	const int classes = 10;
	const int iterations = 500;
	const km_float threshold = 0.0001;
    const int step = 5;
};

struct init_data
{
    int numSamples;
    int dim;
    int classes;
    int iterations;
    km_float threshold = 0.0001;
    km_float** data;
    int* labels;

};

km_float** cu_kmeans(km_float, int*, int*, km_float**, int, int, int);

#endif
