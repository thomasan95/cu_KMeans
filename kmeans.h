#ifndef _KMEANS_GPU_H
#define _KMEANS_GPU_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdexcept>
#include <cctype>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>


typedef float km_float;


#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)


struct parameters
{
	int numSamples = 50000;
	int dim = 2;
	int classes = 10;
	int iterations = 500;
	double threshold = 0.0001;
};

km_float** cu_kmeans(km_float**, int*, int, int, int, double, int*, int*);
//km_float* thrust_kmeans(int, int, int, km_float*, int*, km_float, int*);
void transposeHost(km_float*, km_float*, int, int);

extern int _debug;

#endif // _KMEANS_GPU_H