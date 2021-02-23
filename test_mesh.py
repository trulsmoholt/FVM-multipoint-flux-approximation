from mesh import Mesh
import numpy as np
import math
import matplotlib.pyplot as plt

T = lambda p: np.array([p[0],p[1]])
mesh = Mesh(5,5,T)
cell_centers = mesh.cell_centers

boundary = np.zeros((cell_centers.shape[0],cell_centers.shape[1]))
boundary[0,:] = 1
boundary[cell_centers.shape[0]-1,:] = 3
boundary[:,0] = 4
boundary[:,cell_centers.shape[1]-1] = 2



boundary = np.ravel(boundary)
points = np.reshape(cell_centers,(cell_centers.shape[0]*cell_centers.shape[1],2),order='C')
elements = np.zeros(((cell_centers.shape[0]-1)*(cell_centers.shape[1]-1)*2,3))
e = 0
for i in filter(lambda x: boundary[x]!=2 and mesh.num_unknowns-x>cell_centers.shape[1],range(mesh.num_unknowns)):
    elements[e,:] = np.array([i,i+cell_centers.shape[1]+1,i+cell_centers.shape[1]])
    e = e + 1
    elements[e,:] = np.array([i,i+1,i+cell_centers.shape[1]+1])
    e = e + 1



plt.triplot(points[:,0], points[:,1], elements,color = 'green',linestyle = 'dashed')
plt.show()
print(points)
print(elements)