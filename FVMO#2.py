import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from differentiation import gradient, divergence
import math
K = np.array([[1,0],[0,1]])

x = sym.Symbol('x')
y = sym.Symbol('y')
u_fabric = (-x*y*(1-x)*(1-y))
f = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
f = sym.lambdify([x,y],f)


from mesh import Mesh
import numpy as np


T = lambda x,y: 0.01*x*y + 1*x

mesh = Mesh(4,4,T)
mesh.plot()
nodes = mesh.nodes
cell_centers = mesh.cell_centers
k_global = np.ones((cell_centers.shape[0],cell_centers.shape[1]))
k_global[cell_centers[:,:,0] < 0] = 0
k_global[cell_centers[:,:,0] > 1] = 0
k_global[cell_centers[:,:,1] < 0] = 0
k_global[cell_centers[:,:,1] > 1] = 0

print(k_global)

num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]
matrix = np.zeros((num_unknowns,num_unknowns))

num_cells_x = cell_centers.shape[1]
num_cells_y = cell_centers.shape[0]


def meshToVec(j,i)->int:
    return i*num_cells_y + j
def vecToMesh(h)->(int,int):
    return (h % num_cells_y, math.floor(h/num_cells_y))

def local_assembler(j,i,vec):
    global_vec = np.zeros(num_unknowns)
    
    global_vec[meshToVec(j-1,i-1)] = vec[0]
    global_vec[meshToVec(j-1,i)] = vec[1]
    global_vec[meshToVec(j,i)] = vec[2]
    global_vec[meshToVec(j,i-1)] = vec[3]

    return global_vec

for i in range(1,nodes.shape[0]-1):
    for j in range(1,nodes.shape[1]-1):
        omega = np.zeros((4,4,4))
        interface = np.zeros((4,2))
        centers = np.zeros((4,2))
        n = np.zeros((4,2))
        k_loc = np.zeros((4))

        #D
        v = nodes[i,j-1]-nodes[i,j]
        interface[3,:] = nodes[i,j-1] + 0.5*(v)
        n[3,:] = 0.5*np.array([-v[1],v[0]])
        #A
        v = nodes[i-1,j]-nodes[i,j]
        interface[0,:] = nodes[i,j] + 0.5*(v)
        n[0,:] = 0.5*np.array([-v[1],v[0]])
        #B
        v = nodes[i,j+1]-nodes[i,j]
        interface[1,:] = nodes[i,j] + 0.5*(v)
        n[1,:] = 0.5*np.array([-v[1],v[0]])
        #C
        v = nodes[i+1,j]-nodes[i,j]
        interface[2,:] = nodes[i,j] + 0.5*(v)
        n[2,:] = 0.5*np.array([-v[1],v[0]])

        centers[0,:] = cell_centers[i-1,j-1]
        k_loc[0] = k_global[i-1,j-1]

        centers[1,:] = cell_centers[i-1,j]
        k_loc[1] = k_global[i-1,j]

        centers[2,:] = cell_centers[i,j]
        k_loc[2] = k_global[i,j]

        centers[3,:] = cell_centers[i,j-1]
        k_loc[3] = k_global[i,j-1]

        V = np.zeros((4,2,2))

        for ii in range(4):
            i_2 = interface[ii-1]
            i_1 = interface[ii]
            X = np.array([i_1-centers[ii],i_2-centers[ii]])
            V[ii,:,:] = np.linalg.inv(X)
        for ii in range(4):
            for jj in range(4):
                for kk in range(2):
                    omega[ii,jj,kk] = -n[ii,:].T@K@V[jj,kk,:]*k_loc[jj]


        A = np.array([[omega[0,0,0]-omega[0,1,1],-omega[0,1,0]              ,0                          ,omega[0,0,1]               ],
                      [omega[1,1,1]             ,omega[1,1,0]-omega[1,2,1]  ,-omega[1,2,0]              ,0                          ],
                      [0                        ,omega[2,2,1]               ,omega[2,2,0]-omega[2,3,1]  ,-omega[2,3,0]              ],
                      [-omega[3,0,0]            ,0                          ,omega[3,3,1]               ,omega[3,3,0]-omega[3,0,1]  ]])
        A[np.diag(A[np.diag_indices(4)]==0)] = 1


        B = np.array([[omega[0,0,0]+omega[0,0,1] ,-omega[0,1,0]-omega[0,1,1] ,0                          ,0                          ],
                      [0                         ,omega[1,1,0]+omega[1,1,1]  ,-omega[1,2,0]-omega[1,2,1] ,0                          ],
                      [0                         ,0                          ,omega[2,2,0]+omega[2,2,1]  ,-omega[2,3,0]-omega[2,3,1] ],
                      [-omega[3,0,0]-omega[3,0,1],0                          ,0                          ,omega[3,3,0]+omega[3,3,1]  ]])



        C = np.array([[omega[0,0,0],0           ,0           ,omega[0,0,1]],
                      [omega[1,1,1],omega[1,1,0],0           ,0           ],
                      [0           ,omega[2,2,1],omega[2,2,0],0           ],
                      [0           ,0           ,omega[3,3,1],omega[3,3,0]]])       


        D = np.array([[omega[0,0,0]+omega[0,0,1],0                        ,0                        ,0                        ],
                      [0                        ,omega[1,1,0]+omega[1,1,1],0                        ,0                        ],
                      [0                        ,0                        ,omega[2,2,1]+omega[2,2,0],0                        ],
                      [0                        ,0                         ,0                       ,omega[3,3,0]+omega[3,3,1]]])
        T = C@np.linalg.inv(A)@B-D
        print(T)
        assembler = lambda vec: local_assembler(j,i,vec)

        matrix[meshToVec(j-1,i-1),:] += assembler(T[0,:] - T[3,:])
        matrix[meshToVec(j-1,i),:] += assembler(-T[0,:] + T[1,:])
        matrix[meshToVec(j,i),:] += assembler(-T[1,:] + T[2,:])
        matrix[meshToVec(j,i-1),:] += assembler(-T[2,:] + T[3,:])



vector = np.zeros(num_unknowns)
h_y = nodes[1,0,1]-nodes[0,0,1]
for i in range(cell_centers.shape[0]):
    for j in range(cell_centers.shape[1]):
        base = nodes[i,j+1,0]-nodes[i,j,0]
        top = nodes[i+1,j+1,0]-nodes[i+1,j,0]
        vector[meshToVec(i,j)] += h_y*0.5*(base+top)*f(cell_centers[i,j,0],cell_centers[i,j,1])


print(matrix)

u = np.linalg.solve(matrix,vector)
for i in range(cell_centers.shape[0]):
    matrix[meshToVec(i,0),meshToVec(i,0)] = 1
    matrix[meshToVec(i,cell_centers.shape[1]-1),meshToVec(i,cell_centers.shape[1]-1)] = 1
for i in range(cell_centers.shape[1]):
    matrix[meshToVec(0,i),meshToVec(0,i)] = 1
    matrix[meshToVec(cell_centers.shape[0]-1,i),meshToVec(cell_centers.shape[0]-1,i)] = 1




u_center = np.zeros((cell_centers.shape[0],cell_centers.shape[1]))
for i in range(num_unknowns):
    u_center[vecToMesh(i)] = u[i]


fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot_surface(cell_centers[:,:,0],cell_centers[:,:,1],u_center,cmap='viridis', edgecolor='none')
ax.set_title('computed solution')
ax.set_zlim(0.00, -0.07)
plt.show()

print(u_center)
print(vector)