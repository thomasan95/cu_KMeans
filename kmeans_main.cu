
#include <stdio.h>
#include <stdlib.h>
#include <string.h>   
#include <sys/types.h> 
#include <sys/stat.h>
#include <random>
#include <vector>

#include "kmeans.h"
#include "file_utils.h"
#include "Read_MNIST.h"

#define MAX_MEAN 1000

int main(int argc, char **argv) {

	/*
	CHECK(cudaDeviceReset());
	int deviceNum = 0;
	cudaDeviceProp deviceProp;
	
	cudaGetDeviceProperties(&deviceProp, deviceNum);
	printf("[INFO] Device Number: %d\n", deviceNum);
	*/
	km_float** data;
	km_float** centroids;
	/*
#ifdef LOAD_MNIST
	printf("this shouldn't be printed.\n");
    init_data mnistdata;
    readMNISTFloat(&data, "/path/to/labels", "/path/to/images");
    int n = mnistdata.numSamples;
    int k = mnistdata.classes;
    int d = mnistdata.dim;
    data = mnistdata.data;
    int* true_labels;
    true_labels = mnist.labels;
    kmfloat threshold = mnist.threshold;
    int* pred_labels; 
    pred_labels = (int *)malloc(n * sizeof(int));

#else
*/
    printf("Not using any dataset, generating random data specified in parameters!\n");
    // Load Parameters
	parameters params;

	km_float threshold = params.threshold;
	int n = params.numSamples;
	int k = params.classes;
	int d = params.dim;
   
	int loop_iterations;

	int* pred_labels;
	pred_labels = (int *)malloc(n * sizeof(int));
    assert(pred_labels != NULL);
	// Allocate Memory
	printf("[INFO]: Allocating Memory\n");
	try {
		malloc2D(data, n, d, km_float);
	}
	catch (std::bad_alloc const &e) {
		free(data[0]);
		free(data);
	}

	// Generate Random Data of varying mean, with stddev 2.0
	printf("[INFO]: Generating Random Values\n");
	std::default_random_engine generator;
	//km_float *means = new km_float(k);
    //km_float *means_y = new km_float(k);
	km_float *means;
	km_float *means_y;
	means = (km_float*)malloc(sizeof(km_float)*k);
	means_y = (km_float*)malloc(sizeof(km_float)*k);;
	int count = 0;
    /*
	for (int i = 0; i < k; i++) {
		means[i] = (km_float)count;
		count += 5;
	}
    */
    for(int i = 0; i < k; i++) {
        means[i] = rand() % MAX_MEAN + 1;
        means_y[i] = rand() % MAX_MEAN + 1;
    }
	int pointsPerLabel = n / k;
	//km_float mean;
    //km_float mean_y;
	for (int i = 0; i < k; i++) {
        // Sample from random distribution for varying X and Y means
		//mean_x = means[i];
        //mean_y = means_y[i];

		std::normal_distribution<km_float> distribution_x(means[i], 150.0);
        std::normal_distribution<km_float> distribution_y(means_y[i], 150.0);
		for (int j = 0; j < pointsPerLabel; j++) {
			for (int z = 0; z < d; z++) {
                km_float num;
                if(z == 0) { 
                    num = distribution_x(generator);
                } else {
                    num = distribution_y(generator);
                }
				data[i * pointsPerLabel + j][z] = num;
			}

		}
	}
	if (data == NULL) {
        exit(1);
    }
//#endif
	const char* file_name = "kmeans1.bin";
	FILE *f = fopen(file_name, "wb");
	int saved = log_points(data, f, 1, k, n, d);
	//int saved = save_points(data, file_name, 1, k, n, d);
	if (saved == 0) {
		printf("[FILE] %d data points saved\n\n", n);
	}

	centroids = cu_kmeans(data,
		threshold,
		pred_labels,
		&loop_iterations,
		d,
		n,
		k,
		f);
	/*
	for (int a = 0; a < k; a++) {
		printf("centroids %d: ", a);
		for (int b = 0; b < d; b++) {
			printf("%f ", centroids[a][b]);
		}
		printf("\n\n");
	}
	*/
	fclose(f);
    free(data[0]);
    free(data);
    free(pred_labels);
    free(centroids[0]);
    free(centroids);
    free(means);
    free(means_y);

    return(0);
}

