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

    data_point* data;
};

struct data_point{
    char label;
    char* pixel_values;
    //double atof(const char *str) to convert to double later
}

km_float** cu_kmeans(km_float**, int*, int, int, int, km_float, int*, int*);
km_float* thrust_kmeans(int, int, int, km_float*, int*, km_float, int*);
void transposeHost(km_float*, km_float*, int, int);


template <typename T> T* malloc_aligned_float(long long size) {
    long long const kALIGNByte = 32;
    long long const kALIGN = kALIGNByte / sizeof(T);

    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size * sizeof(T), kALIGNByte);
    if (ptr == nullptr) {
        printf("Bad Alloc!");
        exit(0);
    }
#else
    printf("Using Posix Memalign\n");
    int status = posix_memalign(&ptr, kALIGNByte, size * sizeof(T));
    if (status != 0) {
        printf("Bad Alloc!");
        exit(0);
    }
#endif
    return (T*)ptr;
}
    
template <typename T> void free_aligned_float(T* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#define malloc2Dalign(name, xDim, yDim, type) do {               \
    name = (type **)malloc_aligned_float(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc_aligned_float(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)


#endif // _KMEANS_GPU_H