#ifndef _KMEANS_GPU_intH
#define _KMEANS_GPU_H
#include <stdio.h>
#include <assert.h>

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#ifdef __CUDACC__
inline void CHECK(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("[FATAL] line(%d) cudaErr %d: %s\n", __LINE__, err, cudaGetErrorString(cu_err));
		exit(EXIT_FAILURE);
    }
}
#endif

struct kmeans_model 
{
    float** data;
    float** h_Data;
    float** d_Data;

    float **centroids;
    float **h_Centroids;
    float **d_Centroids;

    int ***newClusters;

    int *d_currCluster;
    int *d_Intermediate;

    int dim;
    int numSamples;
    float threshold;
    int **currCluster;
    int loop_iterations;

    int* clusterCounts;
};

float** cuda_kmeans(float**, int, int, int, float, int*, int*);
float** read_file(char*, int, int*, int*);
extern int _debug;

#endif // _KMEANS_GPU_H