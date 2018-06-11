#ifndef _FILE_UTILS_H
#define _FILE_UTILS_H
#include <stdio.h>
#include <assert.h>

float** read_file(char*, int, int*, int*);
int save_centroids(float**, char*, int, int, int);


#endif // _FILE_UTILS_H