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
	int numSamples = 10000;
	int dim = 784;
	int classes = 10;
	int iterations = 500;
	float threshold = 0.0001;
};

float** cu_kmeans(float, int*, int*, float**, int, int, int);

extern int _debug;

#endif
