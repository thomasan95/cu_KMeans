#ifndef _READ_MNIST_H
#define _READ_MNIST_H

#include <stdio.h>
#include <assert.h>
#include "kmeans.h"

void cleanup(FILE*, FILE*);
void Valid_File(int, int, FILE*, FILE*);
static unsigned int mnist_bin_to_int(char*);
void readMNISTFloat(init_data, char*, char*);
void Read_Data(init_data*, char*, char*);

#endif