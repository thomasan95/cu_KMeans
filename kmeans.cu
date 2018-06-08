#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>    
#include <sys/types.h> 
#include <sys/stat.h>
#include <fcntl.h>

#include "kmeans.h"


#define THREADS_PER_BLOCK 128               // Want to be power 2 and most 128
#define MAX_ITERATIONS 500
#define MAX_CHAR_PER_LINE 128
#define _debug 1

/*! @brief: Get next largest integer that is a power of two
*	@param:
*		int n: integer to find next largest power of two
*	@return:
*		next largest power
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


/*! @brief: Calculate euclidean distance between data and centroids
*	@params:
*		float* d_Data: Data in the form of [dim][numSamples]
*		float* d_Centroids: Centroids in the form of [dim][numSamples]
*		int sampleIdx: Current data point
*		int centroidIdx: Current centroid
*		int numSamples: number of samples in data set
*		int dim: Dimension of the data
*		int numCentroids: number of total centroidss
*/
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


/*! @brief: Calculates the nearest centroid for each data point and stores into
*			d_Centroids
*	@params:
*		float* d_Data: Data in the form of [dim][numSamples]
*		float* d_Centroids: Centroids in the form of [dim][numSamples]
*		int* d_currCluster: Current cluster each data point belongs to
*		int* d_Intermediate: Store changes in which cluster member data is
*		int numSamples: number of samples in data set
*		int dim: Dimension of the data
*		int numCentroids: number of total centroidss
*/
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

#if USE_SHARED_MEM
	float *centroids = (float *)(sMem + blockDim.x);
#else
	float *centroids = d_Centroids;
#endif
		    
    deltaCluster[threadIdx.x] = 0;

#if USE_SHARED_MEM
	for (long long i = threadIdx.x; i < numCentroids; i += blockDim.x) {
		// Iterate jumping through blocks, so no coinciding threads
		for (long long j = 0; j < dim; j++) {
			// Copy centroids over to shared memory
			centroids[numCentroids * j + i] = d_Centroids[numCentroids * j + i];
		}
	}
#endif
    __syncthreads();

	// Get current Sample point (stored along X)
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
        if(d_currCluster[sampleIdx] != index) {
            deltaCluster[threadIdx.x] = 1;
        }

        /* assign current cluster to sample sampleIdx */
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
    // Number of elements in array should be equal to numIntermediate_sq
    // which is # of threads launched. Must be power of two
    extern __shared__ unsigned int intermediates[];
    if(threadIdx.x < numIntermediate) {
        intermediates[threadIdx.x] = d_Intermediate[threadIdx.x];
    } else {
        intermediates[threadIdx.x] = 0;
    }
    __syncthreads();

    for(unsigned int i = numIntermediates_sq / 2; i > 0; i >>= 1) {
        if(threadIdx.x < i) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + i];
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
*		int numSamples: number of samples inside the dataset
*		float threshold: Keep iterating if above this ratio of points change membeship
*		int *currCluster: current cluster memberships
*		int *loop_iterations: store loop iterations
*/
void cu_kmeans(kmeans_model* model,
                       int dim,           
                       int numSamples,       
                       int numCentroids,       
                       float threshold,        
                       int *currCluster,       
                       int *loop_iterations)  
{
    int i, j, index, loop = 0;
	float delta;

    // Transpose Data so thread launched will be different across samples
    // If threads launched block one dimension in X direction, all threads will be
    // of the same sample, want of varying samples but same feature
	// Also in direction X for coalesced memory access along feature
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
			model->h_Centroids[i][j] = model->h_Data[i][j];
        }
    }

    // Initialize Initial Centroids
    for(i = 0; i < numSamples; i++) {
        model->currCluster[i] = -1;
    }
    // Initialize cluster counts
    model->clusterCounts = (int*)calloc(numCentroids, sizeof(int));
    assert(model->clusterCounts != NULL);

    // Initialize New Clusters to all 0s
    malloc2D(model->newClusters, dim, numCentroids, float);
    memset(model->newClusters[0], 0, dim * numCentroids * sizeof(float));

    const unsigned int numClusterBlocks = ceil((float)numSamples/THREADS_PER_BLOCK);

#if USE_SHARED_MEM
	const unsigned int sharedClusterBlockSize = THREADS_PER_BLOCK * sizeof(unsigned char) + numCentroids * dim * sizeof(float);
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
        CHECK(cudaGetLastError());

        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>(model->d_Intermediate, 
                                                                                   numClusterBlocks, 
                                                                                   numReductionThreads);

        cudaDeviceSynchronize();
        CHECK(cudaGetLastError());
        
        int d;

        CHECK(cudaMemcpy(&d, model->d_Intermediate, sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d;

        for(i = 0; i < numSamples; i++) {
            index = model->currCluster[i];
            model->clusterCounts[i]++;
            for(j = 0; j < dim; j++) {
                model->newClusters[i][index] += model->data[i][j];
            }
        }

        for(i = 0; i < numCentroids; i++) {
            for(j = 0; j < dim; j++) {
                if(model->clusterCounts[i] > 0) {
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
			model->centroids[i][j] = model->h_Centroids[j][i];
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


/** @brief reads file from specified path
*	@param
*		char* path: path to read file from
*		int isBinary: Whether file in binary format or not
*		int *numSamples: write number of samples to numSamples
*		int *dim: write dimension to dim
*	@return
*		float** data stored in 2D array of [numSamples][dim]
*/
float** read_file(char* path, int isBinary, int* numSamples, int* dim) {
	float **data;
	int count;
	int len;

	if (isBinary) {
		FILE* fptr = fopen(path, "rb");

		if (fptr == NULL) {
			printf("Error reading file %s\n", path);
			exit(0);
		}
		count = fread(numSamples, sizeof(int), 1, fptr);
		assert(count == 1);
		count = fread(dim, sizeof(int), 1, fptr);
		assert(count == 1);
		if (_debug) {
			printf("File %s numSamples  = %d\n", path, *numSamples);
			printf("File %s dims        = %d\n", path, *dim);
		}

		data = (float**)malloc((*numSamples) * sizeof(float*));
		assert(data != NULL);
		data[0] = (float*)malloc((*numSamples) * (*dim) * sizeof(float));
		assert(data[0] != NULL);
		for (int i = 1; i < (*numSamples); i++) {
			// Set pointers to each data point
			data[i] = data[i - 1] + (*dim);
		}
		count = fread(data[0], sizeof(float), (*dim)*(*numSamples), fptr);
		assert(count == (*dim)*(*numSamples));

		fclose(fptr);
	}
	else {
		FILE *fptr = fopen(path, "r");
		char *line, *ret;
		int curLen;

		if (fptr == NULL) {
			printf("Error reading file %s\n", path);
			return NULL;
		}
		curLen = MAX_CHAR_PER_LINE;
		line = (char*)malloc(curLen);
		assert(line != NULL);
		(*numSamples) = 0;
		while (fgets(line, curLen, fptr) != NULL) {
			while (strlen(line) == curLen - 1) {
				// Not complete line read
				len = strlen(line);
				fseek(fptr, -len, SEEK_CUR);

				curLen += MAX_CHAR_PER_LINE;
				// Reallocate to larger memory
				line = (char*)realloc(line, curLen);
				assert(line != NULL);

				ret = fgets(line, curLen, fptr);
				assert(ret != NULL);
			}
			if (strtok(line, "\t\n") != 0) {
				(*numSamples)++;
			}
		}
		rewind(fptr);
		if (_debug) {
			printf("curLen = %d\n", curLen);
		}
		(*dim) = 0;
		while (fgets(line, curLen, fptr) != NULL) {
			if (strtok(line, "\t\n") != 0) {
				/* ignore the id (first coordinate): dim = 1; */
				while (strtok(NULL, " ,\t\n") != NULL) (*numSamples)++;
				break;
			}
		}
		rewind(fptr);
		if (_debug) {
			printf("File %s numSamples = %d\n", path, *numSamples);
			printf("File %s dim        = %d\n", path, *dim);
		}

		data = (float**)malloc((*numSamples) * sizeof(float*));
		assert(data != NULL);
		// Set [0]th pointer to start of data
		data[0] = (float*)malloc((*numSamples) * (*dim) * sizeof(float));
		assert(data[0] != NULL);
		for (int i = 1; i < (*numSamples); i++) {
			// Set subsequent pointer to next data point
			data[i] = data[i - 1] + (*dim);
		}
		int i = 0;
		while (fgets(line, curLen, fptr) != NULL) {
			if (strtok(line, " \t\n") == NULL) continue;
			for (int j = 0; j < (*dim); j++) {
				data[i][j] = atof(strtok(NULL, " ,\t\n"));
			}
			i++;
		}
		fclose(fptr);
		free(line);
	}
	return data;
}


/** @brief Function for saving the model
*	@param 
*		kmeans_model const* model: model to save
*		char const *path: save path
*	@return 0 if success 1 if failed
*/
int save_model(kmeans_model const* model, char const *path, int isBinary) {
	printf("\n==========Saving Model ==========\n");
	int numSamples = model->numSamples;
	int dim = model->dim;
	int numCentroids = model->numCentroids;

	clock_t start;
	start = clock();

	char command[1024];
	sprintf(command, "del %s", path);
	int sys_ret = system(command);
	if (isBinary) {
		FILE *f = fopen(path, "wb");
		if (f == NULL) {
			printf("Save Failed\n");
			return 1;
		}
		fwrite(&(model->numCentroids), sizeof(int), 1, f);
		fwrite(&(model->dim), sizeof(int), 1, f);
		auto write = [&](float *ptr, int size) {
			for (int i = 0; i < size; i++) {
				float *ptr1 = ptr + i*model->dim;
				fwrite(ptr1, sizeof(float), model->dim, f);
			}
		};
		printf("Saving Centroids\n");
		write(model->centroids[0], model->numCentroids);
		fclose(f);
	}
	else {

		FILE *f = fopen(path, "w");
		if (f == NULL) {
			printf("Save Failed\n");
			return 1;
		}
		for (int i = 0; i < numCentroids; i++) {
			fprintf(f, "%d ", i);
			for (int j = 0; j < dim; j++) {
				fprintf(f, "%f ", model->centroids[i][j]);
			}
			fprintf(f, "\n");
		}
		fclose(f);
	}
	printf("Time Elapsed: %.8lfs\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	return 0;
}