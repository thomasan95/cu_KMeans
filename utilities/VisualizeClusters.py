import numpy as np
import struct
import random
import matplotlib.pyplot as plt
import itertools

colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
          (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
          (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0)]

def ReadData(file_name):

    with open(file_name, "rb") as f:
        epoch = struct.unpack('i',f.read(4))[0]

        num_classes = struct.unpack('i',f.read(4))[0]
        centroids = np.zeros((num_classes,2))
        for i in range(num_classes):
            centroids[i,0] = struct.unpack('i',f.read(4))[0] 
            centroids[i,1] = struct.unpack('i',f.read(4))[0] 

        num_points = struct.unpack('l',f.read(4))[0]
        data_points = np.zeros((num_points,3))
        for i in range(num_points):
            data_points[i,0]= struct.unpack('i',f.read(4))[0] #x-label
            data_points[i,1]= struct.unpack('i',f.read(4))[0] #y-label
            data_points[i,2]= struct.unpack('i',f.read(4))[0] #class

    ScatterPlot(epoch, centroids, data_points)

def ScatterPlot(epoch, centroids, data_points):
    #point_color = list(map(lambda x: colors[x], data_points[:,2]))
    #plt.scatter(data_points[:,0], data_points[:,1], s=size, color=next(point_color))
    

    size = 8
    plt.figure()
    plt.title('Epoch '+epoch)
    for i in range(num_classes):
        temp = data_points[data_points[:,2]==i, :]
        plt.scatter(temp[:,0], temp[:,1], s=size, color=colors[i])
        plt.scatter(centroids[i,0], centroids[i,0], marker='x', s=2*size, color=colors[i])

def main():
    file_name = 'pqmodel_hf.bin' #temp
    ReadData(file_name)

if __name__ == '__main__':
    main()