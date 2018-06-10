#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <random>

#include <iostream>
#include "kmeans.h"
#include "file_utils.h"


int main(int argc, char **argv) {
	// Data Set for Random Testing
	parameters params;
	int n = params.numSamples; // 60k samples
	int d = params.dim; // 784 dimension
	int k = params.classes;  // 10 labels and 10 centroids
	km_float threshold = params.threshold;

	km_float** h_data = nullptr;
	//km_float** h_data_tmp = nullptr;
	int* h_labels = nullptr;

	km_float **h_centroids;
	malloc2D(h_centroids, k, d, km_float);

	int* currCluster;
	currCluster = (int *)malloc(sizeof(int) * n);

	// Allocate Memory
	printf("[INFO]: Allocating Memory\n");
	try {
		// h_data = (km_float *)malloc((long long)n * d * sizeof(km_float));
		// h_data_tmp = (km_float *)malloc((long long)n * d * sizeof(km_float));
		malloc2D(h_data, n, d, km_float);
		h_labels = (int *)malloc((long long)n * sizeof(int));
	}
	catch (std::bad_alloc const &e) {
		std::cerr << e.what() << std::endl;
		free(h_data[0]);
		free(h_data);
		free(h_labels);
		// free(h_data_tmp);
	}

	// Generate Random Data of varying mean, with stddev 2.0
	printf("[INFO]: Generating Random Values\n");
	std::default_random_engine generator;
	double *means = new double(k);
	long long count = 0;
	for (int i = 0; i < k; i++) {
		means[i] = (double)count;
		count += 5;
	}
	int pointsPerLabel = n / k;
	double mean = 0.0;

	count = 0;
	for (int i = 0; i < k; i++) {
		mean = means[i];
		std::normal_distribution<double> distribution(mean, 2.0);
		for (int j = 0; j < pointsPerLabel * d; j++) {
			double num = distribution(generator);
			h_data[i][j] = num;
			count++;
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < pointsPerLabel; j++) {
			h_labels[i * pointsPerLabel + j] = i;
		}
	}
	printf("[INFO]: Done\n");

	//transposeHost(h_data, h_data_tmp, n, d);
	//free(h_data_tmp);

	int loop_iterations = 0;
	printf("[INFO]: Calling KMeans\n");


	h_centroids = cu_kmeans(h_data,
							h_labels,
							d,
							n,
							k,
							threshold,
							currCluster,
							&loop_iterations);
	/*
	h_centroids = thrust_kmeans(n,
								d,
								k,
								h_data,
								h_labels,
								threshold,
								&loop_iterations);

	*/
	free(h_data);
	free(h_labels);

	/*
	thrust::host_vector<km_float> h_data(n * d);
	thrust::host_vector<int> h_labels(n);
	thrust::host_vector<km_float> h_centroids(k * d);

	printf("Generating Random Values\n");
	std::default_random_engine generator;
	double *means = new double(k);
	long long count = 0;
	for (int i = 0; i < k; i++) {
		means[i] = (double)count;
		count += 5;
	}
	int pointsPerLabel = n / k;
	double mean = 0.0;

	count = 0;
	for (int i = 0; i < k; i++) {
		mean = means[i];
		std::normal_distribution<double> distribution(mean, 2.0);
		for (int j = 0; j < pointsPerLabel * d; j++) {
			double num = distribution(generator);
			h_data[i * pointsPerLabel * d + j] = num;
			count++;
		}
	}
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < pointsPerLabel; j++) {
			h_labels[i * pointsPerLabel + j] = i;
			if (j == 0) {
				printf("Assigning %d\n", j);
			}
		}
	}
	for (int i = 0; i < 10; i++) {
		printf("%d: Labels: %d\n", h_labels[i]);
	}
	*/
}
/*
int main(int argc, char **argv) {

    int _debug = 0;
    int isBinary;

    int numCentroids;
    char *path;
    float **data;
    float threshold;
    int loop_iterations;

    _debug = 0;
    threshold = 0.001;
    numCentroids = 0;
    isBinary = 0;

//    model->data = file_read(path, isBinary, &model->numSamples, &model->dim);
    if(model->data == NULL) {
        exit(1);
    }
    model->currCluster = (int*)malloc(model->numSamples * sizeof(int));
    assert(model->currCluster != NULL);

	model->numCentroids = numCentroids;
    cu_kmeans(model, model->dim, model->numSamples, model->numCentroids, threshold, model->currCluster, &loop_iterations);

    free(model->data[0]);
    free(model->data);

    /////////////////////////////
    // TO DO ////////////////////
    /////////////////////////////
    // save_model();
    // free(model->intermediates);
    free(model->centroids[0]);
    free(model->centroids);
    return 0;

}
*/