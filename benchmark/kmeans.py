from sklearn.cluster import KMeans
import struct
import numpy as np
import itertools
import matplotlib.pyplot as plt

import os
import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('-d', '--directory', default='files', 
                    help='folder with binary files')

def ReadData():

    directory = os.fsencode(args.directory)

    GPU_times = []
    CPU_times = []
    n = []

    for file in os.listdir(directory):  
        file = file.decode("utf-8") 
        file = args.directory + "/" + file
        with open(file, "rb") as f:

            num_classes = struct.unpack('i',f.read(4))[0]
            num_points = struct.unpack('i',f.read(4))[0]
            num_dimension = struct.unpack('i',f.read(4))[0]

            #store points
            data_points = np.zeros((num_points,num_dimension))
            centroids = np.zeros((num_classes,num_dimension))
            for i in range(num_points):     
                for d in range(num_dimension):           
                    data_points[i,d]= struct.unpack('f',f.read(4))[0]
            '''
            for i in range(num_classes):
                for d in range(num_dimension):
                    centroids[i,d] = struct.unpack('f',f.read(4))[0] 
            '''

            GPU_time = struct.unpack('f',f.read(4))[0]

            GPU_times.append(GPU_time/1000)
            CPU_times.append(cpu_KMeans(data_points, centroids, num_classes))
            n.append(num_points)

    plt.scatter(n, CPU_times, c='r', label="CPU")
    plt.scatter(n, GPU_times, c='b', label="GPU")
    plt.legend()
    plt.xlabel("Number of Points")
    plt.ylabel("Runtime")
    plt.show()

def cpu_KMeans(data, centroids, k):
    start_time = time.time()
    #print(start_time)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    #kmeans = KMeans(n_clusters=k, init=centroids, random_state=0).fit(data)
    end_time = time.time()
    print(end_time-start_time)
    return (end_time-start_time)


def main():
    global args
    args = parser.parse_args()
    ReadData()



if __name__ == '__main__':
    main()