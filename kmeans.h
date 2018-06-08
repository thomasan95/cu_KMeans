#ifndef _KMEANS_GPU_H
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

inline void CHECK(cudaError_t cu_err) {
    if(cu_err != cudaSuccess) {
        printf("[FATAL] line(%d) cudaErr %d: %s\n", __LINE__, cu_err, cudaGetErrorString(cu_err));
		exit(EXIT_FAILURE);
    }
}

struct kmeans_model 
{
    float** data;				// [numSamples][dim]
    float** h_Data;				// [dim][numSamples]
    float* d_Data;				// [dim][numSamples]

    float **centroids;			// [numClusters][dim]
    float **h_Centroids;		// [dim][numClusters]
    float *d_Centroids;			// [dim[numClusters]

    float **newClusters;		// [dim][numClusters]
	int *newClusterCounts;		// [numSamples]

	int *currCluster;			// [numSamples]
    int *d_currCluster;			// [numSamples]
    int *d_Intermediate;		// [numReductionThreads]

    int dim;
    int numSamples;
    float threshold;
    int loop_iterations;

    int* clusterCounts;
	int numCentroids;
};

void cu_kmeans(kmeans_model*, int, int, int, float, int*, int*);
float** read_file(char*, int, int*, int*);
int save_model(kmeans_model*, char*, int);


#endif // _KMEANS_GPU_H