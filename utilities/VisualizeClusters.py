import numpy as np
import struct
import random
import matplotlib.pyplot as plt
import itertools

#colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
#          (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
#          (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0)]

colors = ['r']*10

def ReadData(file_name, epochs):

    with open(file_name, "rb") as f:
        #epoch = struct.unpack('i',f.read(4))[0]

        num_classes = struct.unpack('i',f.read(4))[0]
        num_points = struct.unpack('i',f.read(4))[0]
        num_dimension = struct.unpack('i',f.read(4))[0]
        print(num_classes)
        print(num_points)
        print(num_dimension)
        #store points
        data_points = np.zeros((num_points,num_dimension))
        for i in range(num_points):     
            for d in range(num_dimension):           
                data_points[i,d]= struct.unpack('f',f.read(4))[0] #x-label
        print("[FILE] Data points read in.")

        centroids = np.zeros((num_classes,num_dimension))
        labels = np.zeros((num_points, 1))
        for epoch in range(epochs):
            print("=============== Epoch "+str(epoch)+" ===============")
            # store centroids
            for i in range(num_classes):
                for d in range(num_dimension):
                    centroids[i,d] = struct.unpack('f',f.read(4))[0] 
            print("[FILE] Centroids read in.")

            #store labels
            for i in range(num_points):
                labels[i] = int(struct.unpack('i',f.read(4))[0])
            print("[FILE] Labels read in.")

            #ScatterPlot(file_name, num_points, data_points, epoch, centroids, labels)
            print("[FILE] Plot for %d epoch saved.\n", epoch)

def ScatterPlot(file_name, n, data_points, epoch, centroids, labels):
    #point_color = list(map(lambda x: colors[x], data_points[:,2]))
    #plt.scatter(data_points[:,0], data_points[:,1], s=size, color=next(point_color))
    
    file_name = file_name.split('.')[0] + "_epoch" +str(epoch)
    size = 8
    plt.figure()
    plt.title("Epoch %"+str(epoch))
    for i in range(n):
        #print(labels[i])
        plt.scatter(data_points[:,0], data_points[:,1], s=size, color=colors[int(labels[i])])
        plt.scatter(centroids[i,0], centroids[i,0], marker='x', s=2*size, color=colors[int(labels[i])])
    plt.savefig(file_name)


def main():
    
    file_name = 'k_means_1.bin' #temp
    epochs = 9 #number of kernel calls - how to not hard code this?
    
    #number of bytes you SHOULD have
    #num_bytes = (4*2)*10000
    #num_bytes += 4*10000*epochs
    #print("num bytes you should have: " + str(num_bytes))

    #ReadData(file_name, epochs)

if __name__ == '__main__':
    main()