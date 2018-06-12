#ifndef _FILE_UTILS_H
#define _FILE_UTILS_H
#include <stdio.h>
#include <assert.h>
typedef float km_float;

float** read_file(char*, int, int*, int*);
int log_centroids(km_float**, FILE*, int, int, int);
int log_points(km_float**, FILE*, int, int, int, int);
int log_labels(int*, FILE*, int, int);

#endif // _FILE_UTILS_H