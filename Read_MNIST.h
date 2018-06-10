#include <stdio.h>
#include <assert.h>
#include "kmeans.h"

void cleanup(FILE*, FILE*);
void Valid_File(int, int, FILE*, FILE*);
static unsigned int mnist_bin_to_int(char*);
void Read_Data(kmeans_model*, char*, char*);