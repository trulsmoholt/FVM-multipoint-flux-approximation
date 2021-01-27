from FVMO import run
from mesh import Mesh
import numpy as np

K = np.array([[1,0],[0,1]])

T = lambda x,y: 0.1*x*y + 0.4*x

mesh = Mesh(10,5,T)
mesh.plot()
nodes = mesh.nodes
cell_centers = mesh.cell_centers

num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]
matrix = np.zeros((num_unknowns,num_unknowns))

num_cells_x = cell_centers.shape[1]
num_cells_y = cell_centers.shape[0]


def meshToVec(j,i)->int:
    return i*num_cells_x + j
def vecToMesh(h)->(int,int):
    return (h % num_cells_x, math.floor(h/num_cells_x))

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

        #D
        v = nodes[i,j]-nodes[i,j-1]
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

        centers[1,:] = cell_centers[i-1,j]

        centers[2,:] = cell_centers[i,j]

        centers[3,:] = cell_centers[i,j-1]

        V = np.zeros((4,2,2))

        for ii in range(4):
            i_2 = interface[ii-1]
            i_1 = interface[ii]
            X = np.array([i_1-centers[ii],i_2-centers[ii]])
            V[ii,:,:] = np.linalg.inv(X)


        for ii in range(4):
            for jj in range(4):
                for kk in range(2):
                    omega[ii,jj,kk] = -n[ii,:].T@K@V[jj,kk,:]


        A = np.array([[omega[0,0,0]-omega[0,1,1],-omega[0,1,0]              ,0                          ,omega[0,0,1]               ],
                      [omega[1,1,1]             ,omega[1,1,0]-omega[1,2,1]  ,-omega[1,2,0]              ,0                          ],
                      [0                        ,omega[2,2,1]               ,omega[2,2,0]-omega[2,3,1]  ,-omega[2,3,0]              ],
                      [-omega[3,0,0]            ,0                          ,omega[3,3,1]               ,omega[3,3,0]-omega[3,0,1]  ]])
                      



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

        assembler = lambda vec: local_assembler(j,i,vec)

        matrix[meshToVec(j-1,i-1),:] += assembler(T[0,:] - T[3,:])
        matrix[meshToVec(j-1,i),:] += assembler(-T[0,:] + T[1,:])
        matrix[meshToVec(j,i),:] += assembler(-T[1,:] + T[2,:])
        matrix[meshToVec(j,i-1),:] += assembler(-T[2,:] + T[3,:])



steps = [1/4,1/8,1/16,1/32,1/64,1/128,1/(2*128),1/(4*128)]
L2_errors = []
print(matrix)
# for h in steps:
#     print('stepsize:    ',h)
#     L2_errors.append(run(h))
#     if len(L2_errors)>1:
#         print('improvement: ',L2_errors[-2]/L2_errors[-1])

