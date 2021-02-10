import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import math
import random


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
        self.volumes = self.__compute_volumes(self.nodes)
        self.midpoints = self.__compute_interface_midpoints(self.nodes)
        self.normals = self.__compute_normals(self.nodes,self.midpoints)

    def __perturb(self,nodes, P):
        for y,row in enumerate(nodes):
            transform = lambda x: P(x,y/(self.num_nodes_y-1))
            T = np.vectorize(transform)
            nodes[y,:,0] = T(row[:,0])
        return nodes

    def __compute_cell_centers(self,nodes):
        #NB unstable for almost right angles
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        cell_centers = np.zeros((num_nodes_y-1,num_nodes_x - 1,2))
        h = float(nodes[1,0,1]-nodes[0,0,1])
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                # b = float(nodes[i,j+1,0]-nodes[i,j,0])
                # a = float(nodes[i+1,j+1,0]-nodes[i+1,j,0])
                # c = np.linalg.norm(nodes[i,j]-nodes[i+1,j])
                # d = np.linalg.norm(nodes[i,j+1]-nodes[i+1,j+1])
                # x = nodes[i,j,0] + b/2 + ((2*a + b)*(c**2-d**2))/(6*(b**2-a**2))
                # y = nodes[i,j,1] + h*(b+2*a)/(3*(a+b))
                x = (nodes[i,j]+nodes[i+1,j]+nodes[i,j+1]+nodes[i+1,j+1])*0.25
                cell_centers[i,j] = np.array([x[0],x[1]])
        return cell_centers
    def __compute_volumes(self,nodes):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        h = float(nodes[1,0,1]-nodes[0,0,1])
        V = np.zeros((num_nodes_y-1,num_nodes_x - 1))
        total = 0
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                base = nodes[i,j+1,0]-nodes[i,j,0]
                top = nodes[i+1,j+1,0]-nodes[i+1,j,0]
                V[i,j] = h*(base+top)*0.5
        return V

    def __compute_interface_midpoints(self,nodes):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        midpoints = np.zeros((num_nodes_y-1,num_nodes_x-1,2,2))
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                midpoints[i,j,0,:] = 0.5*(nodes[i+1,j,:]+nodes[i,j,:])
                midpoints[i,j,1,:] = 0.5*(nodes[i,j+1,:]+nodes[i,j,:])
        return midpoints

    def __compute_normals(self,nodes,midpoints):
        num_nodes_y = nodes.shape[0]
        num_nodes_x = nodes.shape[1]
        normals = np.zeros((num_nodes_y-1,num_nodes_x-1,2,2))
        for i in range(num_nodes_y-1):
            for j in range(num_nodes_x-1):
                v = midpoints[i,j,0,:]-nodes[i,j,:]
                normals[i,j,0,:] = np.array([v[1],-v[0]])
                v = midpoints[i,j,1,:]-nodes[i,j,:]
                normals[i,j,1,:] = np.array([-v[1],v[0]])
        return normals
        
        
        return

    def plot(self):
        plt.scatter(self.cell_centers[:,:,0],self.cell_centers[:,:,1])
        plt.scatter(self.nodes[:,:,0], self.nodes[:,:,1])

        segs1 = np.stack((self.nodes[:,:,0],self.nodes[:,:,1]), axis=2)
        segs2 = segs1.transpose(1,0,2)
        plt.gca().add_collection(LineCollection(segs1))
        plt.gca().add_collection(LineCollection(segs2))
        # plt.quiver(*self.midpoints[1,1,0,:],self.normals[1,1,0,0],self.normals[1,1,0,1])
        # plt.quiver(*self.midpoints[1,1,1,:],self.normals[1,1,1,0],self.normals[1,1,1,1])
        plt.savefig('perturbed_grid_aspect_0.2_mesh.pdf')

        plt.show()

    def meshToVec(self,j,i)->int:
        return i*self.cell_centers.shape[0] + j

    def vecToMesh(self,h)->(int,int):
        return (h % self.cell_centers.shape[0], math.floor(h/self.cell_centers.shape[0]))

    def plot_vector(self,vec,text = 'text'):
        vec_center = np.zeros((self.cell_centers.shape[0],self.cell_centers.shape[1]))
        num_unknowns = self.cell_centers.shape[1]*self.cell_centers.shape[0]
        for i in range(num_unknowns):
            vec_center[self.vecToMesh(i)] = vec[i]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.contourf(self.cell_centers[:,:,0],self.cell_centers[:,:,1],vec_center,20,)
        plt.colorbar()
        fig.suptitle(text)
        plt.show()
    def plot_funtion(self,fun,text = 'text'):
        vec_center = np.zeros((self.cell_centers.shape[0],self.cell_centers.shape[1]))
        num_unknowns = self.cell_centers.shape[1]*self.cell_centers.shape[0]
        for i in range(num_unknowns):
            xx,yy = self.cell_centers[self.vecToMesh(i)]
            vec_center[self.vecToMesh(i)] = fun(xx,yy)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        plt.contourf(self.cell_centers[:,:,0],self.cell_centers[:,:,1],vec_center,20,)
        plt.colorbar()
        fig.suptitle(text)

        plt.show()

    