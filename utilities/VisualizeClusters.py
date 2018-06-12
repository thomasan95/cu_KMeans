import numpy as np
import struct
import random
import matplotlib.pyplot as plt
import itertools

import argparse
import time

parser = argparse.ArgumentParser(description='Visualization of KMeans')
parser.add_argument('-e', '--epochs', type=int, help='number of epochs for file')
parser.add_argument('-f', '--file', default='flowers', help='binary file name')


colors = ['#e6194b', '#3cb44b', '#ffe119','#0082c8','#f58231','#911eb4',
            '#46f0f0', '#f032e6','#d2f53c','#fabebe','#008080' ,'#aa6e28',
            '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000080']

def ReadData():

    with open(args.file, "rb") as f:

        file_name = args.file.split('.')[0] 

        num_classes = struct.unpack('i',f.read(4))[0]
        num_points = struct.unpack('i',f.read(4))[0]
        num_dimension = struct.unpack('i',f.read(4))[0]

        #store points
        data_points = np.zeros((num_points,num_dimension))
        for i in range(num_points):     
            for d in range(num_dimension):           
                data_points[i,d]= struct.unpack('f',f.read(4))[0] #x-label

        centroids = np.zeros((num_classes,num_dimension))
        labels = np.zeros((num_points, 1))
        for epoch in range(args.epochs+1):
            print("=============== Epoch "+str(epoch)+" ===============")
            # store centroids
            for i in range(num_classes):
                for d in range(num_dimension):
                    centroids[i,d] = struct.unpack('f',f.read(4))[0] 
            print("[FILE] Centroids read in.")

            #store labels (initial does not have label)
            if (epoch!=0):
                for i in range(num_points):
                    labels[i] = int(struct.unpack('i',f.read(4))[0])
                print("[FILE] Labels read in.")

            ScatterPlot(file_name, num_points, num_classes, data_points, epoch, centroids, labels)
            print("[FILE] Plot for epoch " +str(epoch) +  " saved.\n")

def ScatterPlot(file_name, n, k, data_points, epoch, centroids, labels):
    #point_color = list(map(lambda x: colors[x], data_points[:,2]))
    #plt.scatter(data_points[:,0], data_points[:,1], s=size, color=next(point_color))

    file_name = file_name + "_epoch" +str(epoch) +'.png'
    
    size = 8
    plt.figure()
    plt.title("Epoch "+str(epoch))

    if (epoch==0):
        c = ['#808080']*n
    else:
        c = list(map(lambda x: colors[int(x)], labels))

    plt.scatter(data_points[:,0], data_points[:,1], s=size, color=c)
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=5*size, color='black')
    plt.savefig(file_name)


def main():
    global args
    args = parser.parse_args()
    #file_name = 'kmeans3.bin' #temp
    #epochs = 15 #number of kernel calls - how to not hard code this?

    ReadData()

if __name__ == '__main__':
    main()