
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <string>
#include "Read_MNIST.h"

void cleanup(FILE* file1, FILE* file2) {
	if (file1)
		fclose(file1);
	if (file2)
		fclose(file2);
}

void Valid_File(int check_value, int true_value, FILE* file1, FILE* file2) {
	if (check_value != true_value)
		cleanup(file1, file2);
}

static unsigned int mnist_bin_to_int(char *v)
{
	int i;
	unsigned int ret = 0;

	for (i = 0; i < 4; ++i) {
		ret <<= 8;
		ret |= (unsigned char)v[i];
	}
	return ret;
}

void readMNISTFloat(init_data* model, char* labels, char* images) {
	printf("Reading in data...\n");

	FILE* label_file;
	FILE* image_file;
	label_file = fopen(labels, "rb");
	image_file = fopen(images, "rb");


	if (!label_file || !image_file) {
		printf("Invalid file. \n\n");
		cleanup(label_file, image_file);
	}

	unsigned int label_magic_num = 2049;
	unsigned int image_magic_num = 2051;

	// Labels file
	char temp[4];
	unsigned int check;

	fread(&temp, 1, sizeof(int), label_file); //magic number
	check = mnist_bin_to_int(temp);
	Valid_File(check, label_magic_num, label_file, image_file);

	fread(&temp, 1, sizeof(int), label_file); //number of samples = 10,000
	model->numSamples = mnist_bin_to_int(temp);
	//printf("%d\n", model->numSamples);

	// Images file
	fread(&temp, 1, sizeof(int), image_file); //magic number
	check = mnist_bin_to_int(temp);
	Valid_File(check, image_magic_num, label_file, image_file);

	fread(&temp, 1, sizeof(int), image_file); //number of images == number of labels
	check = mnist_bin_to_int(temp);
	Valid_File(check, model->numSamples, label_file, image_file);

	fread(&temp, 1, sizeof(int), image_file); //num rows = 28
	model->dim = mnist_bin_to_int(temp);
	fread(&temp, 1, sizeof(int), image_file); // num cols = 28
	check = mnist_bin_to_int(temp);
	Valid_File(check, model->dim, label_file, image_file);
	//printf("hi\n");

	model->data = (char**)malloc(model->numSamples * sizeof(char*));
	model->labels = (char*)malloc(model->numSamples * sizeof(char));

	malloc2D(model->data, model->numSamples, model->dim * model->dim, char);
	model->labels = (char*)malloc(model->numSamples * sizeof(char));

	for (int i = 0; i< model->numSamples; i++) {
		unsigned char read_data[model->dim * model->dim];
		unsigned char read_label;
		fread(read_data, 1, model->dim * model->dim, image_file);
		for(int j = 0; j < model->dim * model->dim; j++) {
			model->data[i][j] = (float)(read_data[j] / 255.0);
		}

		fread(temp, 1, 1, label_file);
		model->label[i] = mnist_bin_to_int(tmp);
	}

	cleanup(image_file, label_file);
	printf("Model values read in. \n\n");
}


void Read_Data(kmeans_model* model, char* labels, char* images) {
	printf("Reading in data...\n");

	FILE* label_file;
	FILE* image_file;
	label_file = fopen(labels, "rb");
	image_file = fopen(images, "rb");


	if (!label_file || !image_file) {
		printf("Invalid file. \n\n");
		cleanup(label_file, image_file);
	}

	unsigned int label_magic_num = 2049;
	unsigned int image_magic_num = 2051;

	// Labels file
	char temp[4];
	unsigned int check;

	fread(&temp, 1, sizeof(int), label_file); //magic number
	check = mnist_bin_to_int(temp);
	Valid_File(check, label_magic_num, label_file, image_file);

	fread(&temp, 1, sizeof(int), label_file); //number of samples = 10,000
	model->numSamples = mnist_bin_to_int(temp);
	//printf("%d\n", model->numSamples);

	// Images file
	fread(&temp, 1, sizeof(int), image_file); //magic number
	check = mnist_bin_to_int(temp);
	Valid_File(check, image_magic_num, label_file, image_file);

	fread(&temp, 1, sizeof(int), image_file); //number of images == number of labels
	check = mnist_bin_to_int(temp);
	Valid_File(check, model->numSamples, label_file, image_file);

	fread(&temp, 1, sizeof(int), image_file); //num rows = 28
	model->dim = mnist_bin_to_int(temp);
	fread(&temp, 1, sizeof(int), image_file); // num cols = 28
	check = mnist_bin_to_int(temp);
	Valid_File(check, model->dim, label_file, image_file);
	//printf("hi\n");

	model->data = (char**)malloc(model->numSamples * sizeof(char*));
	model->labels = (char*)malloc(model->numSamples * sizeof(char));
	for (int i = 0; i< model->numSamples; i++) {
		model->data[i] = (char*)malloc(sizeof(char)*model->dim*model->dim);
		fread(&model->labels[i], sizeof(char), 1, label_file);
		fread(model->data[i], sizeof(char), model->dim*model->dim, image_file);
	}

	cleanup(image_file, label_file);
	printf("Model values read in. \n\n");
}
