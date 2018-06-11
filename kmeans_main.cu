
#include <stdio.h>
#include <stdlib.h>
#include <string.h>   
#include <sys/types.h> 
#include <sys/stat.h>
#include <random>
#include <vector>

#include "kmeans.h"
#include "file_utils.h"
// #include "read_mnist.h"

#define MAX_MEAN 100

int main(int argc, char **argv) {

	km_float ** data;
	km_float ** centroids;

#ifdef LOAD_MNIST
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
    printf("Not using any dataset, generating random data specified in parameters!\n");
    // Load Parameters
	parameters params;

	km_float threshold = params.threshold;
	int n = params.numSamples;
	int k = params.classes;
	int d = params.dim;
    int step = params.step;

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
	km_float *means = new km_float(k);
    km_flot *means_y = new km_float(k);
    /*
	for (int i = 0; i < k; i++) {
		means[i] = (km_float)(i * 5);
	}
    */
    for(int i = 0; i < k; i++) {
        means[i] = rand() % MAX_MEAN + 1;
        means_y[i] = rand() % MAX_MEAN + 1;
    }
	int pointsPerLabel = n / k;
	km_float mean = 0.0;
    km_float mean_y = 0.0;
	for (int i = 0; i < k; i++) {
        // Sample from random distribution for varying X and Y means
		mean_x = means[i];
        mean_y = means_y[i];

		std::normal_distribution<km_float> distribution_x(mean_x, 1.0);
        std::normal_distribution<km_float> distribution_y(mean_y, 1.0);
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
        printf("Data failed to allocate and set properly!\n")
        exit(1);
    }
#endif
    centroids = cu_kmeans(data, d, n, k, threshold, pred_labels, &loop_iterations);

	for (int i = 0; i < k; i++) {
		printf("centroids %d: [ ", i);
		for (int j = 0; j < d; j++) {
			printf("%f ", centroids[i][j]);
		}
		printf("]\n\n");
    }
    
    // save centroids as binary file
    int saved = save_centroids(centroids, labels, "saved_centroids.bin", 1, k, d);

    if(saved == 0) {
        printf("Save Successful\n");
    }

    free(data[0]);
    free(data);
    free(labels);
    free(centroids[0]);
    free(centroids);
    delete means;
    delete means_y;

    return(0);
}

