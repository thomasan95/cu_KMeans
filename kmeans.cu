#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <cstring>    
#include <sys/types.h> 
#include <sys/stat.h>
#include <fcntl.h>
#include <cfloat>
#include "../common/common.h"
//#include <thrust/device_ptr.h>
//#include <thrust/device_malloc.h>
//#include <thrust/device_free.h>
//#include <thrust/device_vector.h>
// #include <cuda.h>
// #include <thrust/copy.h>


#include "kmeans.h"

#define MAX_ITERATIONS 500 
#define THREADS_PER_BLOCK 2


/*! @brief: Get next largest integer that is a power of two
*	@param:
*		int n: integer to find next largest power of two
*	@return:
*		next largest power
*/
static inline int nextPowerOfTwo(unsigned int n) {
	n--;

	n = n >> 1 | n;
	n = n >> 2 | n;
	n = n >> 4 | n;
	n = n >> 8 | n;
	n = n >> 16 | n;

	return ++n;
}

void transposeHost(km_float* arr_out, km_float* arr_in, int n, int d) {
	printf("[INFO]: Transposing Dataset from [%d, %d] to [%d, %d]\n", n, d, d, n);
	for (int iy = 0; iy < n; ++iy) {
		for (int ix = 0; ix < d; ++ix) {
			arr_out[ix * n + iy] = arr_in[iy * d + ix];
		}
	}
	printf("[INFO]: Done Transposing\n");
}

/*
void transposeHost(thrust::host_vector<km_float> arr_out,
	thrust::host_vector<km_float> arr_in,
	const int rows,
	const int cols)
{
	for (int iy = 0; iy < rows; ++iy) {
		for (int ix = 0; ix < cols; ++ix) {
			arr_out[ix * rows + iy] = arr_in[iy * cols + ix];
		}
	}
}
*/

/*! @brief: Calculate euclidean distance between data and centroids
*	@params:
*		float* d_Data: Data in the form of [dim][n]
*		float* d_Centroids: Centroids in the form of [dim][n]
*		int sampleIdx: Current data point
*		int centroidIdx: Current centroid
*		int n: number of samples in data set
*		int dim: Dimension of the data
*		int k: number of total centroidss
*/
__host__ __device__ inline static km_float sq_euclid_dist(km_float* d_Data,
														   km_float* d_Centroids,
														   int sampleIdx,
														   int centroidIdx,
														   int n,
														   int d,
														   int k)
{
	int i;
	km_float val = 0.0;
    for(i = 0; i < d; i++) {
        val += (d_Data[n * i + sampleIdx] - d_Centroids[k * i + centroidIdx]) * 
                (d_Data[n * i + sampleIdx] - d_Centroids[k * i + centroidIdx]);
    }

    return val;
}


/*! @brief: Calculates the nearest centroid for each data point and stores into
*			d_Centroids
*	@params:
*		float* d_Data: Data in the form of [dim][n]
*		float* d_Centroids: Centroids in the form of [dim][n]
*		int* d_currCluster: Current cluster each data point belongs to
*		int* d_Intermediate: Store changes in which cluster member data is
*		int n: number of samples in data set
*		int dim: Dimension of the data
*		int k: number of total centroidss
*/
__global__ static void find_nearest_centroid(km_float* d_Data,
											 km_float* d_Centroids,
                                             int* d_currCluster,
                                             int* d_Intermediate,
                                             int n,
                                             int d,
                                             int k) 
{
	//if (blockIdx.x == 1 && threadIdx.x == 1) {
	//	printf("Here2\n");
	//}
	extern __shared__ char sMem[];

    unsigned char *deltaCluster = (unsigned char *)sMem;

#if USE_SHARED_MEM
	km_float *centroids = (km_float *)(sMem + blockDim.x);
#else
	km_float *centroids = d_Centroids;
#endif
		    
    deltaCluster[threadIdx.x] = 0;

#if USE_SHARED_MEM
	for (int i = threadIdx.x; i < k; i += blockDim.x) {
		// Iterate jumping through blocks, so no coinciding threads
		for (int j = 0; j < d; j++) {
			// Copy centroids over to shared memory
			centroids[k * j + i] = d_Centroids[k * j + i];
		}
	}
	__syncthreads();
#endif

	// Get current Sample point (stored along X)
    int sampleIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if(sampleIdx < n) {
		km_float dist, min_dist;

        int index, i;
		index = 0;
        min_dist = sq_euclid_dist(d_Data, 
								centroids,
                                sampleIdx,
                                0, 
                                n,
                                d, 
                                k);
        for(int i = 1; i < k; i++) {
            dist = sq_euclid_dist(d_Data, 
								centroids,
								sampleIdx,
                                i,
                                n,
                                d,
                                k);
            if(dist < min_dist) {
                min_dist = dist;
                index = i;
            }
        }
        if(d_currCluster[sampleIdx] != index) {
            deltaCluster[threadIdx.x] = 1;
        }

        // assign current cluster to sample sampleIdx
        d_currCluster[sampleIdx] = index;

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


/*! @brief: Calculates number of centroid changes
*	@params:
*		int* d_Intermediate: changes stored from device
*		int numIntermediate: number of intermediates
*		int numIntermediate_sq: next power of two for numIntermediate
*/
__global__ static void compute_delta(int* d_Intermediate,
                                     int numIntermediate,
                                     int numIntermediates_sq)
{
	//if (blockIdx.x == 1 && threadIdx.x == 1) {
	//	printf("Here3\n");
	//}

    // Number of elements in array should be equal to numIntermediate_sq
    // which is # of threads launched. Must be power of two
    extern __shared__ unsigned int intermediates[];
	intermediates[threadIdx.x] = (threadIdx.x < numIntermediate) ? d_Intermediate[threadIdx.x] : 0;
    __syncthreads();

    for(unsigned int s = numIntermediates_sq / 2; s > 0; s >>= 1) {
        if(threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        d_Intermediate[0] = intermediates[0];
    }
}


/** @brief perform kmeans on the model passed in and stored centroids inside model->centroids
*	@param
*		kmeans_model* model: model with data initialized and parameters set
*		int dim: dimension of the data
*		int n: number of samples inside the dataset
*		float threshold: Keep iterating if above this ratio of points change membeship
*		int *currCluster: current cluster memberships
*		int *loop_iterations: store loop iterations
*/
km_float** cu_kmeans(km_float** data,
						int d,           
						int n,       
						int k,       
						double threshold,
						int *currCluster,       
						int *loop_iterations)  
{
    int i, j, index, loop = 0;
	km_float delta;
	km_float **host_data;		// [d][n]
	km_float **h_centroids;		// [d][k]
	km_float **centroids;		// [k][d]
	km_float **newClusters;		// [d][k]

	km_float *d_data;			// [d * n]
	km_float *d_centroids;		// [d * k]
	int *d_currCluster;			// [n]
	int *d_intermediate;
	int *counts;

    // Transpose Data so thread launched will be different across samples
    // If threads launched block one dimension in X direction, all threads will be
    // of the same sample, want of varying samples but same feature
	// Also in direction X for coalesced memory access along feature
    malloc2D(host_data, d, n, km_float);
	memset(host_data[0], 0, d * n * sizeof(km_float));
    for(i = 0; i < d; i++) {
        for(j = 0; j < n; j++) {
			host_data[i][j] = data[j][i];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
	printf("[PROG]: Initializing Centroids\n");
    malloc2D(h_centroids, d, k, km_float);
	int minIdx = 0;
	int maxIdx = n;
	int *randValues = new int(k);
	for (i = 0; i < k; i++) {
		randValues[i] = (rand() % (maxIdx - minIdx)) + minIdx;
	}
	// Initilize Random Centroids
	for (j = 0; j < k; j++) {
		index = randValues[j];
		for (i = 0; i < d; i++) {
			h_centroids[i][j] = host_data[i][index];
		}
	}
	printf("[PROG]: Initializing currCluster, counts, and newClusters\n");
    // Initialize Initial Centroids to -1
    for(i = 0; i < n; i++) {
        currCluster[i] = -1;
    }
    // Initialize cluster counts
    counts = (int*)calloc(k, sizeof(int));
    assert(counts != NULL);

    // Initialize New Clusters to all 0s
    malloc2D(newClusters, d, k, km_float);
    memset(newClusters[0], 0, d * k * sizeof(km_float));
	const unsigned int numClusterBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#if USE_SHARED_MEM
	const unsigned int sharedClusterBlockSize = sizeof(unsigned char) * THREADS_PER_BLOCK + k * d * sizeof(km_float);

#else
	printf("Not using shared memory\n");
	const unsigned int sharedClusterBlockSize = THREADS_PER_BLOCK * sizeof(unsigned char);
#endif
	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int); 

	printf("Allocating Space for Device Variables\n");
	cudaError_t err;
    err = cudaMalloc(&d_data, n * d * sizeof(km_float)); 
	if (err != cudaSuccess) {
		printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
	}
	err = cudaMalloc(&d_currCluster, n * sizeof(int));
	if (err != cudaSuccess) {
		printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
	}
	err = cudaMalloc(&d_centroids, k * d * sizeof(km_float));
	if (err != cudaSuccess) {
		printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
	}
	err = cudaMalloc(&d_intermediate, numReductionThreads * sizeof(unsigned int));
	if (err != cudaSuccess) {
		printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
	}
	printf("Copying over host data to device!\n");
	err = cudaMemcpy(d_data, host_data[0], n * d * sizeof(km_float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
	}
	err = cudaMemcpy(d_currCluster, currCluster, n * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
	}
	// Begin Kernel Call
    do {
		err = cudaMemcpy(d_centroids, h_centroids[0], k * d * sizeof(km_float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
		}

		printf("ITER :%d Calling Kernel: Find Nearest Centroid\n", loop+1);
        find_nearest_centroid <<< numClusterBlocks, THREADS_PER_BLOCK, sharedClusterBlockSize >>>(d_data, 
                                                                                                  d_centroids, 
                                                                                                  d_currCluster,
                                                                                                  d_intermediate, 
                                                                                                  n,
                                                                                                  d,
                                                                                                  k);

        cudaDeviceSynchronize();
		compute_delta << < 1, numReductionThreads, reductionBlockSharedDataSize >> >(d_intermediate,
																					numClusterBlocks,
																					numReductionThreads);



        cudaDeviceSynchronize();

        int de;
		err = cudaMemcpy(&de, d_intermediate, sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
		}

        delta = (km_float)de;
		err = cudaMemcpy(currCluster, d_currCluster, n * sizeof(int), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("line(%d), Cuda Error: %s\n", __LINE__, cudaGetErrorString(err));
		}


        for(i = 0; i < n; i++) {
			counts[index]++;
            for(j = 0; j < d; j++) {
                newClusters[j][index] += data[i][j];
            }
        }
        for(i = 0; i < k; i++) {
            for(j = 0; j < d; j++) {
                if(counts[i] > 0) {
                    h_centroids[j][i] = newClusters[j][i] / counts[i];
                }
                newClusters[j][i] = 0.0; // Set back to 0.0;
            }
			counts[i] = 0;
        }

        delta /= n;
		printf("Delta: %f\n", delta);
    } while (delta > threshold && loop++ < MAX_ITERATIONS);
	printf("Done with find_closest_centroid! after %d iterations!\n", loop);
	// Keep track of number of iterations
    *loop_iterations = loop + 1;

    malloc2D(centroids, k, d, km_float);
    for(i = 0; i < k; i++) {
        for(j = 0; j < d; j++) {
            // Transpose Centroids back to [k][d]
			centroids[i][j] = h_centroids[j][i];
        }
    }
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_currCluster);
    cudaFree(d_intermediate);
    free(host_data[0]);
    free(host_data);
    free(h_centroids[0]); 
    free(h_centroids); 
    free(newClusters[0]);
    free(newClusters);
    free(counts); 

	return centroids;
}


/*
km_float* thrust_kmeans(int n,
						int d, 
						int k,
						km_float *data, 
						int *labels,
						km_float threshold, 
						int* loop_iterations)
{
	int i, j, index, loop_iter = 0;
	km_float delta;

	// Transpose data
	//thrust::host_vector<km_float> h_data(d * k);
	//thrust::host_vector<km_float> h_centroids(d * k);
	//thrust::host_vector<int> currCluster(n); // initialized to all zeros
	//thrust::host_vector<km_float> newClusters(d * k);
	//transposeHost(h_data, data, n, d);


	// Copy host data and labels to Device
	printf("Copy Data to Host Vector\n");
	thrust::host_vector<km_float> host_data(n * d);
	for (i = 0; i < n * d; i++) {
		host_data[i] = data[i];
	}
	thrust::device_vector<km_float> d_data(host_data.begin(), host_data.end());
	//thrust::copy(host_data.begin(), host_data.end(), d_data.begin());
	//d_data = host_data;
	printf("hi\n");
	thrust::device_ptr<int> labels_ptr(labels);
	thrust::device_vector<int> d_labels(labels_ptr, labels_ptr + n);
	//thrust::device_vector<km_float> d_centroids(k * d);
	//thrust::device_vector<int> d_currCluster(n);
	printf("DEBUG: 2\n");

	// Allocate memory for centroids and cluster tracking on host
	km_float* h_centroids = (km_float *)malloc((long long)k * d * sizeof(km_float));
	int* currCluster = (int *)malloc((long long)n * sizeof(int));

	// Initialize centroids to K random points
	int minIdx = 0;
	int maxIdx = n;
	int *randValues = new int(k);
	for (i = 0; i < k; i++) {
		randValues[i] = (rand() % (maxIdx - minIdx)) + minIdx;
	}
	for (i = 0; i < d; i++) {
		for (j = 0; j < k; j++) {
			index = randValues[j];
			h_centroids[i * k + k] = data[i * index + index];
		}
	}
	printf("DEBUG: 4\n");

	thrust::device_vector<km_float> d_centroids(h_centroids, h_centroids + (k * d));



	// Initialize initial clusters all to -1
	for (i = 0; i < n; i++) {
		currCluster[i] = -1;
	}
	thrust::device_vector<int> d_currCluster(currCluster, currCluster + n);

	km_float* newClusters = (km_float *)malloc((long long)d * k * sizeof(km_float));

	// Initialize cluster counts
	int *counts;
	counts = (int*)calloc(k, sizeof(int));
	assert(counts != NULL);
	printf("DEBUG: 5\n");

	const unsigned int numClusterBlocks = ceil((km_float)n / THREADS_PER_BLOCK);
#if USE_SHARED_MEM
	const unsigned int sharedClusterBlockSize = THREADS_PER_BLOCK * sizeof(unsigned char) + k * d * sizeof(km_float);
	cudaDeviceProp devProp;
	int dev;
	cudaGetDevice(&dev);
	cudaGetDeviceProperties(&devProp, dev);
	if (sharedClusterBlockSize > devProp.sharedMemPerBlock) {
		printf("Not enough shared memory. Recompile without USE_SHARED_MEM\n");
		exit(0);
	}
#else
	const unsigned int sharedClusterBlockSize = THREADS_PER_BLOCK * sizeof(unsigned char);
#endif

	const unsigned int numReductionThreads = nextPowerOfTwo(numClusterBlocks);
	const unsigned int reductionBlockSharedDataSize =
		numReductionThreads * sizeof(unsigned int);

	thrust::device_vector<int> d_intermediate(numReductionThreads * sizeof(unsigned int));

	// Cast raw pointers to prepare for kernel call
	km_float* pd_data = thrust::raw_pointer_cast(d_data.data());			// assigned
	int* pd_currCluster = thrust::raw_pointer_cast(d_currCluster.data());	// assigned
	km_float* pd_centroids = thrust::raw_pointer_cast(d_centroids.data());	// assigned
	int* pd_intermediate = thrust::raw_pointer_cast(d_intermediate.data());	// assigned
	printf("DEBUG: 6\n");

	do {
		// Copy from host centroid to device
		thrust::copy(h_centroids.end(), h_centroids.begin(), d_centroids.begin());
		find_nearest_centroid << <numClusterBlocks, THREADS_PER_BLOCK, sharedClusterBlockSize >> >(pd_data,
			pd_centroids,
			pd_currCluster,
			pd_intermediate,
			n,
			d,
			k);
		cudaDeviceSynchronize();
		compute_delta << <1, numReductionThreads, reductionBlockSharedDataSize >> > (pd_intermediate,
			numClusterBlocks,
			numReductionThreads);

		cudaDeviceSynchronize();

		int d;
		cudaMemcpy(&d, pd_intermediate, sizeof(int), cudaMemcpyDeviceToHost);

		delta = (km_float)d;
		for (i = 0; i < n; i++) {
			index = currCluster[i];
			counts[i]++;
			for (j = 0; j < d; j++) {
				newClusters[i * d + index] += data[i * d + j];
			}
		}
		// Compute new centroids from device new clusters
		// sum up all data points then divide by number of data points
		for (i = 0; i < k; i++) {
			for (j = 0; j < d; j++) {
				if (counts[i] > 0) {
					h_centroids[j * n + i] = newClusters[j * n + i] / counts[i];
				}
				newClusters[j * n + i] = 0.0;
			}
			counts[i] = 0;
		}
		// Get percentage of change
		delta /= n;
	} while (delta > threshold && loop_iter++ < MAX_ITERATIONS);
	*loop_iterations = loop_iter + 1;

	// Copy back centroids from Device to Host then transpose
	km_float* ret_centroids_tmp = malloc_aligned_float<km_float>((long long)d * k);
	thrust::copy(d_centroids.begin(), d_centroids.end(), ret_centroids_tmp);


	
	//for (i = 0; i < k; i++) {
	//	for (j = 0; j < d; j++) {
	//		centroids[i * d + j] = d_centroids[j * k + i];
	//	}
	//}

	transposeHost(h_centroids, ret_centroids_tmp, d, k);
	printf("DEBUG: 7\n");
	delete(randValues);
	//free<km_float>(ret_centroids_tmp);
	free(currCluster);
	free(newClusters);
	printf("DEBUG: 8\n");
	return h_centroids;

}
*/