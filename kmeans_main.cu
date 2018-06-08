#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kmeans.h"


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