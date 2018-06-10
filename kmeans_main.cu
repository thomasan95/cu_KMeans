
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <random>
#include <vector>


int      _debug;
#include "kmeans.h"


int main(int argc, char **argv) {

	float ** data;
	float ** centroids;

    _debug = 1;
	// Load Parameters
	parameters params;

	float threshold = params.threshold;
	int n = params.numSamples;
	int k = params.classes;
	int d = params.dim;
   
	int loop_iterations;

	int* labels;
	labels = (int *)malloc(sizeof(int) * n);

	// Allocate Memory
	printf("[INFO]: Allocating Memory\n");
	try {
		malloc2D(data, n, d, float);
	}
	catch (std::bad_alloc const &e) {
		free(data[0]);
		free(data);
	}

	// Generate Random Data of varying mean, with stddev 2.0
	printf("[INFO]: Generating Random Values\n");
	std::default_random_engine generator;
	float *means = new float(k);
	int count = 0;
	for (int i = 0; i < k; i++) {
		means[i] = (float)count;
		count += 5;
	}
	int pointsPerLabel = n / k;
	float mean = 0.0;

	for (int i = 0; i < k; i++) {
		mean = means[i];
		std::normal_distribution<float> distribution(mean, 2.0);
		for (int j = 0; j < pointsPerLabel; j++) {
			for (int z = 0; z < d; z++) {
				float num = distribution(generator);
				data[i * pointsPerLabel + j][z] = num;
			}

		}
	}    
	if (data == NULL) exit(1);

	labels = (int*) malloc(n * sizeof(int));
    assert(labels != NULL);

    centroids = cu_kmeans(data, d, n, k, threshold, labels, &loop_iterations);

	for (int a = 0; a < k; a++) {
		printf("centroids %d: ", a);
		for (int b = 0; b < d; b++) {
			printf("%f ", centroids[a][b]);
		}
		printf("\n\n");
	}



    free(data[0]);
    free(data);
    free(labels);
    free(centroids[0]);
    free(centroids);


    return(0);
}

