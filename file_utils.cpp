#include <stdio.h>
#include <stdlib.h>
#include <cstring>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */

#include "kmeans.h"

#define MAX_CHAR_PER_LINE 128 // 128 threads per block

float** read_file(char* path, int isBinary, int* numSamples, int* dim) {
    float **data;
    int count;
    int len;

    if(isBinary) {
        FILE* fptr = fopen(path, "rb");

        if(fptr == NULL) {
            printf("Error reading file %s\n", path);
            exit(0);
        }
        count = fread(numSamples, sizeof(int), 1, fptr);
        assert(count == 1);
        count = fread(dim, sizeof(int), 1, fptr);
        assert(count == 1);
        if(_debug) {
            printf("File %s numSamples  = %d\n", path, *numSamples);
            printf("File %s dims        = %d\n", path, *dim);
        }
        
        data = (float**)malloc((*numSamples)*sizeof(float*));
        assert(data != NULL);
        data[0] = (float*)malloc((*numSamples) * (*dim) * sizeof(float));
        assert(data[0] != NULL);
        for(int i = 1; i < (*numSamples); i++) {
            // Set pointers to each data point
            data[i] = data[i-1] + (*dim);
        }
        count = fread(data[0], sizeof(float), (*dim)*(*numSamples), fptr);
        assert(count == (*dim)*(*numSamples));

        fclose(fptr);
    }
    else {
        FILE *fptr = fopen(path, "r");
        char *line, *ret;
        int curLen;

        if(fptr == NULL) {
            printf("Error reading file %s\n", path);
            return NULL;
        }
        curLen = MAX_CHAR_PER_LINE;
        line = (char*)malloc(curLen);
        assert(line != NULL);
        (*numSamples) = 0;
        while(fgets(line, curLen, fptr) != NULL) {
            while(strlen(line) == curLen-1) {
                // Not complete line read
                len = strlen(line);
                fseek(fptr, -len, SEEK_CUR);

                curLen += MAX_CHAR_PER_LINE;
                // Reallocate to larger memory
                line = (char*) realloc(line, curLen);
                assert(line != NULL);

                ret = fgets(line, curLen, fptr);
                assert(ret != NULL);
            }
            if(strtok(line, "\t\n") != 0) {
                (*numSamples)++;
            }
        }
        rewind(fptr);
        if(_debug) {
            printf("curLen = %d\n", curLen);
        }
        (*dim) = 0;
        while(fgets(line, curLen, fptr) != NULL) {
            if (strtok(line, "\t\n") != 0) {
                /* ignore the id (first coordinate): dim = 1; */
                while(strtok(NULL, " ,\t\n") != NULL) (*numSamples)++;
                break;
            }
        }
        rewind(fptr);
        if(_debug) {
            printf("File %s numSamples = %d\n", path, *numSamples);
            printf("File %s dim        = %d\n", path, *dim);
        }

        data = (float**)malloc((*numSamples) * sizeof(float*));
        assert(data != NULL);
        // Set [0]th pointer to start of data
        data[0] = (float*)malloc((*numSamples) * (*dim) * sizeof(float));
        assert(data[0] != NULL);
        for(int i = 1; i < (*numSamples); i++) {
            // Set subsequent pointer to next data point
            data[i] = data[i-1] + (*dim);
        }
        int i = 0;
        while(fgets(line, curLen, fptr) != NULL) {
            if(strtok(line, " \t\n") == NULL) continue;
            for( int j = 0; j < (*dim); j++) {
                data[i][j] = atof(strtok(NULL, " ,\t\n"));
            }
            i++;
        }
        fclose(fptr);
        free(line);
    }
    return data;
}

int save_model(float** centroids, char const *path) {
    // TODO
    return 0;
}