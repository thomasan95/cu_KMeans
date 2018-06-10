
#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

#define THREADS_PER_BLOCK 128

/*! @brief: find the next largest number that is a power of 2
*	@params:
*		int n: integer to raise
*/
static inline int nextPowerOfTwo(int n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;

    return ++n;
}

/*! @brief: find euclidean norm between centroid and data point
*	@params:
*		int d: dimension of data and centroids
*		int n: number of data points
*		int k: number of centroids
*		float *d_data: data loaded onto device
*		float *centroids: device centroids to compute
*		int dataidx: idx of data point from warp address
*		int centroididx: idx of centroid from warp address
*	@return:
*		float val: euclidean distance (L2 norm)
*/
__host__ __device__ inline static float get_norm(int    d,
												int    n,
												int    k,
												float *d_data,     
												float *centroids, 
												int    dataidx,
												int    centroididx)
{
    int i;
    float val=0.0;

    for (i = 0; i < d; i++) {
		val += (d_data[n * i + dataidx] - centroids[k * i + centroididx]) *
               (d_data[n * i + dataidx] - centroids[k * i + centroididx]);
    }

    return(val);
}

/*! @brief: find closest centroid to each respective datapoint
*	@params:
*		int d: dimension of data and centroids
*		int n: number of data points
*		int k: number of centroids
*		float *d_data: data loaded onto device
*		float *d_centroids: device centroids to compute
*		int *d_currCluster: current label
*		int *d_deltas: keep track of amount of changes
*/
__global__ static void find_nearest_centroid(int d,
											  int n,
											  int k,
											  float *d_data,          
											  float *d_centroids,  
											  int *d_currCluster,       
											  int *d_deltas)
{
    extern __shared__ char smem[];

    unsigned char *change_cluster = (unsigned char *)smem;
#if USE_SHARED_MEM
    float *centroids = (float *)(smem + blockDim.x);
#else
    float *centroids = d_centroids;
#endif

	change_cluster[threadIdx.x] = 0;

#if USE_SHARED_MEM
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        for (int j = 0; j < d; j++) {
			centroids[k * j + i] = d_centroids[k * j + i];
        }
    }
    __syncthreads();
#endif

    int dataidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (dataidx < n) {
        int   index, i;
        float dist, min_dist;

        index    = 0;
        min_dist = get_norm(d, n, k, d_data, centroids, dataidx, 0);

        for (i=1; i<k; i++) {
            dist = get_norm(d, n, k, d_data, centroids, dataidx, i);
            if (dist < min_dist) {
                min_dist = dist;
                index    = i;
            }
        }

        if (d_currCluster[dataidx] != index) {
			change_cluster[threadIdx.x] = 1;
        }

		d_currCluster[dataidx] = index;

        __syncthreads(); 

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
				change_cluster[threadIdx.x] +=
					change_cluster[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
			d_deltas[blockIdx.x] = change_cluster[0];
        }
    }
}

/*! @brief: compute the amount of changed data points
*	@params:
*		int *d_deltas: device variable to store deltas
*		int numDeltas: number of changes
*		int numDeltas_sq: number of changes with next higher power of 2
*/
__global__ static void compute_delta(int *d_deltas,
								   int numDeltas,  
								   int numDeltas_sq)  
{

    extern __shared__ unsigned int deltas[];

	deltas[threadIdx.x] = (threadIdx.x < numDeltas) ? d_deltas[threadIdx.x] : 0;

    __syncthreads();

    for (unsigned int s = numDeltas_sq / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
			deltas[threadIdx.x] += deltas[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
		d_deltas[0] = deltas[0];
    }
}

/*! @brief: Performs KMeans on the dataset provided
*	@params:
*		int d: dimension of dataset
*		int n: number of data points
*		int k: number of centroids to find
*		float threshold: threshold to kmeans termination
*		int *currCluster: storage of labels
*		int *loop_iterations: store loops
*/
float** cu_kmeans(float **data,   
					float   threshold,
					int    *currCluster,
					int    *loop_iterations
					int     d,    
					int     n,      
					int     k)
{
    int      i, j, index, loop=0;
    int     *counts; 
                              
	float    delta = 0.0;;

    float  **centroids;     
    float  **h_centroids;
    float  **newClusters;  

    float *d_data;
    float *d_centroids;
    int *d_currCluster;
    int *d_deltas;

	float  **h_data;
    malloc2D(h_data, d, n, float);
    for (i = 0; i < d; i++) {
        for (j = 0; j < n; j++) {
            h_data[i][j] = data[j][i];
        }
    }

	int minIdx = 0;
	int maxIdx = n;
	int *randValues = new int(k);
	for (i = 0; i < k; i++) {
		randValues[i] = (rand() % (maxIdx - minIdx)) + minIdx;
	}

	// Initilize Random Centroids
	malloc2D(h_centroids, d, k, float);
	for (j = 0; j < k; j++) {
		index = randValues[j];
		for (i = 0; i < d; i++) {
			h_centroids[i][j] = h_data[i][index];
		}
	}

    /* initialize currCluster[] */
	for (i = 0; i < n; i++) {
		currCluster[i] = -1;

	}
	// Initialize all initial counts to 0
	counts = (int*) calloc(k, sizeof(int));
    assert(counts != NULL);

    malloc2D(newClusters, d, k, float);
    memset(newClusters[0], 0, d * k * sizeof(float));


    //const unsigned int THREADS_PER_BLOCK = 128;
    const unsigned int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
#if USE_SHARED_MEM
    const unsigned int clusterBlockSharedDataSize = THREADS_PER_BLOCK * sizeof(unsigned char) + k * d * sizeof(float);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
		printf("WARNING: Your CUDA hardware has insufficient block shared memory\n");
    }
#else
    const unsigned int clusterBlockSharedDataSize = THREADS_PER_BLOCK * sizeof(unsigned char);
#endif

    const unsigned int numReductionThreads = nextPowerOfTwo(numBlocks);
    const unsigned int reductionBlockSharedDataSize = numReductionThreads * sizeof(unsigned int);

	CHECK(cudaMalloc(&d_data, n*d*sizeof(float)));
	CHECK(cudaMalloc(&d_centroids, k*d*sizeof(float)));
	CHECK(cudaMalloc(&d_currCluster, n*sizeof(int)));
	CHECK(cudaMalloc(&d_deltas, numReductionThreads*sizeof(unsigned int)));

	CHECK(cudaMemcpy(d_data, h_data[0],
              n*d*sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_currCluster, currCluster,
              n*sizeof(int), cudaMemcpyHostToDevice));

    do {
		printf("[ITER: %d]: CALLING KERNEL\n", loop + 1);
		CHECK(cudaMemcpy(d_centroids, h_centroids[0], k*d*sizeof(float), cudaMemcpyHostToDevice));

		find_nearest_centroid<<< numBlocks, THREADS_PER_BLOCK, clusterBlockSharedDataSize >>> (d,
																								n,
																								k, 
																								d_data,
																								d_centroids, 
																								d_currCluster,
																								d_deltas);

        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>(d_deltas, 
																					numBlocks,
																					numReductionThreads);

        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        int de;
		CHECK(cudaMemcpy(&de, d_deltas, sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)de;

		CHECK(cudaMemcpy(currCluster, d_currCluster, n*sizeof(int), cudaMemcpyDeviceToHost));

        for (i=0; i<n; i++) {
            index = currCluster[i];

            counts[index]++;
            for (j=0; j<d; j++)
                newClusters[j][index] += data[i][j];
        }


        for (i=0; i<k; i++) {
            for (j=0; j<d; j++) {
                if (counts[i] > 0)
                    h_centroids[j][i] = newClusters[j][i] / counts[i];
                newClusters[j][i] = 0.0; 
            }
            counts[i] = 0;  
        }



        delta /= n;
    } while (delta > threshold && loop++ < 500);

    *loop_iterations = loop + 1;


    malloc2D(centroids, k, d, float);
    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
			centroids[i][j] = h_centroids[j][i];
        }
    }

    CHECK(cudaFree(d_data));
	CHECK(cudaFree(d_centroids));
	CHECK(cudaFree(d_currCluster));
	CHECK(cudaFree(d_deltas));

    free(h_data[0]);
    free(h_data);
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