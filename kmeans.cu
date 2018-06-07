#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kmeans.h"


#define THREADS_PER_BLOCK 128               // Want to be power 2 and most 128
#define MAX_ITERATIONS 500


static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;

    return ++n;
}

__host__ __device__ inline static float sq_euclid_dist(float* d_Data,
                                                       float* d_Centroids,
                                                       int sampleIdx,
                                                       int centroidIdx,
                                                       int numSamples,
                                                       int dim,
                                                       int numCentroids)
{
    float val = 0.0;
    for(int i = 0; i < dim; i++) {
        val += (d_Data[numSamples * i + sampleIdx] - d_Centroids[numCentroids * i + centroidIdx]) * 
                (d_Data[numSamples * i + sampleIdx] - d_Centroids[numCentroids * i + centroidIdx]);
    }

    return val;
}

__global__ static void find_nearest_centroid(float* d_Data,
                                             float* d_Centroids,
                                             int* d_currCluster,
                                             int* d_Intermediate,
                                             int numSamples,
                                             int dim,
                                             int numCentroids) 
{
    extern __shared__ char sMem[];

    unsigned char *deltaCluster = (unsigned char *)sMem;
#if SHARED_MEM_OPTIMIZATION
    float *centroids = (float *)(sMem + blockDim.x);
#else
    float *centroids = d_Centroids;
#endif
    
    deltaCluster[threadIdx.x] = 0;

#if SHARED_MEM_OPTIMIZATION
    for(long long i = threadIdx.x; i < numCentroids; i += blockDim.x) {
        // Iterate jumping through blocks, so no coinciding threads
        for(long long j = 0; j < dim; j++) {
            // Copy centroids over to shared memory
            centroids[numCentroids * j + i] = d_Centroids[numCentroids * j + i];
        }
    }
    __syncthreads();
#endif
    int sampleIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if(sampleIdx < numSamples) {
        float dist, min_dist;

        int index = 0;
        min_dist = sq_euclid_dist(d_Data, 
                                d_Centroids, 
                                sampleIdx,
                                0, 
                                numSamples,
                                dim, 
                                numCentroids);
        for(int i = 1; i < numCentroids; i++) {
            dist = sq_euclid_dist(d_Data, 
                                d_Centroids,
                                sampleIdx,
                                i,
                                numSamples,
                                dim,
                                numCentroids);
            if(dist < min_dist) {
                min_dist = dist;
                index = i;
            }
        }
        if(currCluster[sampleIdx] != index) {
            deltaCluster[threadIdx.x] = 1;
        }

        /* assign current cluster to sample sampleIdx */
        currCluster[sampleIdx] = index;

        __syncthreads();

        for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if(threadIdx.x < s) {
                deltaCluster[threadIdx.x] += deltaCluster[threadIdx.x + s];
            }
            __syncthreads();
        }
        if(threadIdx.x == 0) {
            d_Intermediate[blockIdx.x] = deltaCluster[0];
        }
    }
}

__global__ static void compute_delta(int* d_Intermediate,
                                     int numIntermediate,
                                     int numIntermediates_sq)
{
    // Number of elements in array should be equal to numIntermediate_sq
    // which is # of threads launched. Must be power of two
    extern __shared__ unsigned int intermediates[];
    if(threadIdx.x < numIntermediate) {
        intermediates[threadIdx.x] = d_Intermediate[threadIdx.x];
    } else {
        intermediates[threadIdx.x] = 0;
    }
    __syncthreads();

    for(unsigned int i = numIntermediate2 / 2; i > 0; i >>= 1) {
        if(threadIdx.x < i) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        d_Intermediate[0] = intermediates[0];
    }
}
/** @brief perform kmeans on the model passed in
*/
void cu_kmeans(kmeans_model* model,
                       int dim,                // Dimension of each sample
                       int numSamples,         // Number of Samples
                       int numCentroids,       // Number of Centroids to cluster
                       float threshold,        // 
                       int *currCluster,        //
                       int *loop_iterations)   //
{
    long long i, j, index, loop = 0;

    // Transpose Data so thread launched will be different across samples
    // If threads launched block one dimension in X direction, all threads will be
    // of the same sample, want of varying samples but same feature
    malloc2D(model->h_Data, numSamples, dim, float);
    for(i = 0; i < dim; i++) {
        for(j = 0; j < numSamples; j++) {
            model->h_Data[i][j] = model->data[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(model->h_Centroids, numCentroids, dim, float);
    for(i = 0; i < dim; i++) {
        for(j = 0; j < numCentroids; j++) {
            model->h_Centroids[i][j] = h_Data[i][j]
        }
    }

    // Initialize Initial Centroids
    for(i = 0; i < numSamples; i++) {
        model->currCluster[i] = -1;
    }
    // Initialize cluster counts
    model->clusterCounts = (int*)calloc(numClusters, sizeof(int));
    assert(model->clusterCounts != NULL);

    // Initialize New Clusters to all 0s
    malloc2D(model->newClusters, dim, numClusters, float);
    memset(model->newClusters[0], 0, numCoords * numClusters * sizeof(float));

    const unsigned int numClusterBlocks = ceil((float)numSamples/THREADS_PER_BLOCK);

#if SHARED_MEM_OPTIMIZATION
    const unsigned int sharedClusterBlockSize = THREADS_PER_BLOCK * sizeof(unsigned char) + numClusters * dim * sizeof(float);
    cudaDeviceProp devProp;
    int dev;
    cudaGetDevice(&dev);
    cudaGetDEviceProperties(&devProp, dev);
    if(sharedClusterBlockSize > devProp.sharedMemPerBlock) {
        err("Not enough shared memory")
    }
#else
    const unsigned int sharedClusterBlockSize = THREADS_PER_BLOCK * sizeof(unsigned char);
#endif

    const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize = 
        numReductionThreads * sizeof(unsigned int);

    CHECK(cudaMalloc(&model->d_Data, numSamples * dim * sizeof(float)));
    CHECK(cudaMalloc(&model->d_currCluster, numSamples * sizeof(int)));
    CHECK(cudaMalloc(&model->d_Centroids, numCentroids * dim * sizeof(float)));
    CHECK(cudaMalloc(&model->d_Intermediate, numReductionThreads * sizeof(unsigned int)));

    CHECK(cudaMemcpy(model->d_Data, model->h_Data[0], numSamples * dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(model->d_currCluster, model->currCluster, numSamples * sizeof(int), cudaMemcpyHostToDevice));

    do {
        CHECK(cudaMemcpy(model->d_Centroids, model->h_Centroids[0], numCentroids * dim * sizeof(float), cudaMemcpyHostToDevice));

        find_nearest_centroid <<< numClusterBlocks, THREADS_PER_BLOCK, sharedClusterBlockSize >>>(model->d_Data, 
                                                                                                  model->d_Centroids, 
                                                                                                  model->d_currCluster,
                                                                                                  model->d_Intermediate, 
                                                                                                  numSamples,
                                                                                                  dim,
                                                                                                  numCentroids);
        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError());

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>(model->d_Intermediate, 
                                                                                   numClusterBlocks, 
                                                                                   numReductionThreads);

        cudaDeviceSynchronize();
        checkCuda(cudaGetLastError());
        
        int d;

        CHECK(cudaMemcpy(&d, model->d_Intermediate, sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;

        for(i = 0; i < numSamples; i+) {
            index = model->currCluster[i];

            model->clusterCounts[i]++;
            for(j = 0; j < dim; j++) {
                model->newClusters[i][index] += model->data[i][j];
            }
        }

        for(i = 0; i < numCentroids; i++) {
            for(j = 0; j < dim; j++) {
                if(clusterCounts[i] > 0) {
                    model->h_Centroids[j][i] = model->newClusters[j][i] / model->clusterCounts[i];
                }
                model->newClusters[j][i] = 0.0; /* set back to 0 */
            }
            model->clusterCounts[i] = 0;
        }
        delta /= numSamples;
    } while (delta > threshold && loop++ < MAX_ITERATIONS);

    *loop_iterations = loop + 1;

    malloc2D(model->centroids, numCentroids, dim, float);
    for(i = 0; i < numCentroids; i++) {
        for(j = 0; j < dim; j++) {
            /* Transpose Centroids back to [numCentroids][dim] */
            model->centroids[i][j] = model->h_Centroids[j][i]
        }
    }

    CHECK(cudaFree(model->d_Data));
    CHECK(cudaFree(model->d_Centroids));
    CHECK(cudaFree(model->d_currCluster));
    CHECK(cudaFree(model->d_Intermediate));

    free(model->h_Data[0]);
    free(model->h_Data);
    free(model->h_Centroids[0]);
    free(model->h_Centroids);
    free(model->newClusters[0]);
    free(model->newClusters);
    free(model->clusterCounts);
}

int main(int argc, char **argv) {
    int _debug = 0;
    int isBinary;

    int numCentroids;
    char *path;
    float **data;
    float threshold;
    int loop_iterations;

    kmeans_model *model = new kmeans_model;

    _debug = 0;
    threshold = 0.001;
    numCentroids = 0;
    isBinary = 0;
    filename = NULL;

    model->data = file_read(path, isBinary, &model->numSamples, &model->dim);
    if(model->data == NULL) {
        exit(1);
    }
    model->currCluster = (int*)malloc(numSamples * sizeof(int));
    assert(currCluster != NULL);

    model->centroids = cuda_kmeans(model, dim, numSamples, numCentroids, threshold, currCluster, &loop_iterations);

    free(model->data[0]);
    free(model->objects);

    /////////////////////////////
    // TO DO ////////////////////
    /////////////////////////////
    save_model();
    free(model->intermediates);
    free(model->centroids[0]);
    free(model->centroids);

    return 0;

}