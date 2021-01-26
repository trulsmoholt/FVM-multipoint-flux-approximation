import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Mesh:
    def __init__(self,num_nodes_x,num_nodes_y,P):
        self.num_nodes_x = num_nodes_x
        self.num_nodes_y = num_nodes_y
        bottom_left = (0,0)
        top_right = (1,1)
        nodes_x, nodes_y = np.meshgrid(np.linspace(bottom_left[0],top_right[0],num=num_nodes_x),np.linspace(bottom_left[1],top_right[1],num=num_nodes_y))
        nodes = np.stack([nodes_x,nodes_y],axis=2)

        self.nodes = self.__perturb(nodes,P)

        self.cell_centers = self.__compute_cell_centers(self.nodes)

    def __perturb(self,nodes, P):
        for y,row in enumerate(nodes):
            transform = lambda x: P(x,y)
            T = np.vectorize(transform)
            nodes[y,:,0] = T(row[:,0])
        return nodes

    def __compute_cell_centers(self,nodes):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        cell_centers = np.zeros((num_nodes_y-1,num_nodes_x - 1,2))
        h = float(nodes[1,0,1]-nodes[0,0,1])
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                b = float(nodes[i,j+1,0]-nodes[i,j,0])
                a = float(nodes[i+1,j+1,0]-nodes[i+1,j,0])
                c = np.linalg.norm(nodes[i,j]-nodes[i+1,j])
                d = np.linalg.norm(nodes[i,j+1]-nodes[i+1,j+1])
                x = nodes[i,j,0] + b/2 + ((2*a + b)*(c**2-d**2))/(6*(b**2-a**2))
                y = nodes[i,j,1] + h*(b+2*a)/(3*(a+b))
                cell_centers[i,j] = np.array([x,y])
        return cell_centers
    def plot(self):
        plt.scatter(self.cell_centers[:,:,0],self.cell_centers[:,:,1])
        plt.scatter(self.nodes[:,:,0], self.nodes[:,:,1])

        segs1 = np.stack((self.nodes[:,:,0],self.nodes[:,:,1]), axis=2)
        segs2 = segs1.transpose(1,0,2)
        plt.gca().add_collection(LineCollection(segs1))
        plt.gca().add_collection(LineCollection(segs2))
        plt.show()
