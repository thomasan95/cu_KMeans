// #include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>    
#include <sys/types.h> 
#include <sys/stat.h>

#include "file_utils.h"

#define MAX_CHAR_PER_LINE 128
#define _debug 1


/** @brief reads file from specified path
*	@param
*		char* path: path to read file from
*		int isBinary: Whether file in binary format or not
*		int *numSamples: write number of samples to numSamples
*		int *dim: write dimension to dim
*	@return
*		km_float** data stored in 2D array of [numSamples][dim]
*/
km_float** read_file(char* path, int isBinary, int* numSamples, int* dim) {
	km_float **data;
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

		data = (km_float**)malloc((*numSamples) * sizeof(km_float*));
		assert(data != NULL);
		data[0] = (km_float*)malloc((*numSamples) * (*dim) * sizeof(km_float));
		assert(data[0] != NULL);
		for (int i = 1; i < (*numSamples); i++) {
			// Set pointers to each data point
			data[i] = data[i - 1] + (*dim);
		}
		count = fread(data[0], sizeof(km_float), (*dim)*(*numSamples), fptr);
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

		data = (km_float**)malloc((*numSamples) * sizeof(km_float*));
		assert(data != NULL);
		// Set [0]th pointer to start of data
		data[0] = (km_float*)malloc((*numSamples) * (*dim) * sizeof(km_float));
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
int log_centroids(km_float** centroids, FILE* f, int isBinary, int k, int d) {

	clock_t start;
	start = clock();

	// Write in Binary
	if (isBinary) {
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < d; j++) {
				fwrite(&centroids[j][i], sizeof(km_float), 1, f);
			}
		}
	}
	else {

		for (int i = 0; i < k; i++) {
			fprintf(f, "%d ", i);
			for (int j = 0; j < d; j++) {
				fprintf(f, "%f ", centroids[i][j]);
			}
			fprintf(f, "\n");
		}

	}
	//printf("Time Elapsed: %.8lfs\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	return 0;
}

/** @brief Function for saving the cluster of the current point
*	@param
*		int* currCluster: current labels of the points
*		char const *path: path to save the label to
*		int isBinary: Whether file in binary format or not
*		int n: number of points to save
*	@return 0 if success 1 if failed
*/
int log_labels(int* labels, FILE *f, int isBinary, int n) {
	//printf("\n========== Saving Labels ==========\n");

	clock_t start;
	start = clock();

	char command[1024];
	//sprintf(command, "del %s", path);
	//int sys_ret = system(command);

	// Write in Binary
	if (isBinary) {
		//data points and centroids already written to file
		fwrite(labels, sizeof(int), n, f);
	}
	else {

		for (int i = 0; i < n; i++) {
			fprintf(f, "%d ", i);
			fprintf(f, "%i\n", labels[i]);
		}
	}
	//printf("Time Elapsed: %.8lfs\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	return 0;
}


/** @brief Function for saving the random generated points
*	@param
*		km_float** data: data points to save
*		char const *path: path to save the points to
*		int isBinary: Whether file in binary format or not
*		int k: number of classes
*		int n: number of points to save
*		int d: dimension of the point
*	@return 0 if success 1 if failed
*/
int log_points(km_float** data, FILE*f, int isBinary, int k, int n, int d) {
	printf("\n==========Saving Points ==========\n");

	clock_t start;
	start = clock();

	//char command[1024];
	//sprintf(command, "del %s", path);
	//int sys_ret = system(command);

	// Write in Binary
	if (isBinary) {

		fwrite(&k, sizeof(int), 1, f);
		fwrite(&n, sizeof(int), 1, f);
		fwrite(&d, sizeof(int), 1, f);

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < d; j++) {
				fwrite(&data[i][j], sizeof(km_float), 1, f);
			}
		}
	}
	else {

		for (int i = 0; i < n; i++) {
			fprintf(f, "%d ", i);
			for (int j = 0; j < d; j++) {
				fprintf(f, "%f ", data[i][j]);
			}
			fprintf(f, "\n");
		}

	}
	printf("Time Elapsed: %.8lfs\n", (clock() - start) / (double)CLOCKS_PER_SEC);
	return 0;
}