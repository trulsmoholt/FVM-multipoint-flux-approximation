import numpy as np



from mesh import Mesh


def compute_matrix(mesh,K):
    nodes = mesh.nodes
    cell_centers = mesh.cell_centers
    k_global = np.ones((cell_centers.shape[0],cell_centers.shape[1]))

    k_global = np.ones((cell_centers.shape[0],cell_centers.shape[1]))
    nx = nodes.shape[1]
    ny = nodes.shape[0]

    num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]

    meshToVec = mesh.meshToVec
    vecToMesh = mesh.vecToMesh
    matrix = np.zeros((num_unknowns,num_unknowns))


    def local_assembler(j,i,vec):
        global_vec = np.zeros(num_unknowns)
        
        global_vec[meshToVec(j-1,i-1)] = vec[0]
        global_vec[meshToVec(j-1,i)] = vec[1]
        global_vec[meshToVec(j,i)] = vec[2]
        global_vec[meshToVec(j,i-1)] = vec[3]

        return global_vec

    for i in range(1,nodes.shape[0]-1):
        for j in range(1,nodes.shape[1]-1):
            omega = np.zeros((4,4,2))
            interface = np.zeros((4,2))
            centers = np.zeros((4,2))
            n = np.zeros((4,2))
            k_loc = np.zeros((4))

            #D
            v = nodes[i,j-1]-nodes[i,j]
            interface[3,:] = nodes[i,j] + 0.5*(v)
            n[3,:] = 0.5*np.array([v[1],-v[0]])
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
            n[2,:] = 0.5*np.array([v[1],-v[0]])

            centers[0,:] = cell_centers[i-1,j-1]
            k_loc[0] = k_global[i-1,j-1]

            centers[1,:] = cell_centers[i-1,j]
            k_loc[1] = k_global[i-1,j]

            centers[2,:] = cell_centers[i,j]
            k_loc[2] = k_global[i,j]

            centers[3,:] = cell_centers[i,j-1]
            k_loc[3] = k_global[i,j-1]

            V = np.zeros((4,2,2))

            for jj in range(4):
                i_2 = interface[jj-1]
                i_1 = interface[jj]

                X = np.array([i_1-centers[jj],i_2-centers[jj]])
                V[jj,:,:] = np.linalg.inv(X)
                

            for ii in range(4):
                for jj in range(4):
                    for kk in range(2):
                        omega[ii,jj,kk] = -n[ii,:].T@K@V[jj,kk,:]*k_loc[jj]
            
            #print(omega)

            A = np.array([[omega[0,0,0]+omega[0,1,1],omega[0,1,0]              ,0                          ,omega[0,0,1]               ],
                        [omega[1,1,1]             ,omega[1,1,0]+omega[1,2,1]  ,omega[1,2,0]              ,0                          ],
                        [0                        ,omega[2,2,1]               ,omega[2,2,0]+omega[2,3,1]  ,omega[2,3,0]              ],
                        [omega[3,0,0]            ,0                          ,omega[3,3,1]               ,omega[3,3,0]+omega[3,0,1]  ]])
            


            B = np.array([[omega[0,0,0]+omega[0,0,1] ,omega[0,1,0]+omega[0,1,1] ,0                          ,0                          ],
                        [0                         ,omega[1,1,0]+omega[1,1,1]  ,omega[1,2,0]+omega[1,2,1] ,0                          ],
                        [0                         ,0                          ,omega[2,2,0]+omega[2,2,1]  ,omega[2,3,0]+omega[2,3,1] ],
                        [omega[3,0,0]+omega[3,0,1],0                          ,0                          ,omega[3,3,0]+omega[3,3,1]  ]])



            C = np.array([[omega[0,0,0],0           ,0           ,omega[0,0,1]],
                        [omega[1,1,1],omega[1,1,0],0           ,0           ],
                        [0           ,omega[2,2,1],omega[2,2,0],0           ],
                        [0           ,0           ,omega[3,3,1],omega[3,3,0]]])       


            D = np.array([[omega[0,0,0]+omega[0,0,1],0                        ,0                        ,0                        ],
                        [0                        ,omega[1,1,0]+omega[1,1,1],0                        ,0                        ],
                        [0                        ,0                        ,omega[2,2,1]+omega[2,2,0],0                        ],
                        [0                        ,0                         ,0                       ,omega[3,3,0]+omega[3,3,1]]])
            T = C@np.linalg.inv(A)@B-D

            assembler = lambda vec: local_assembler(i,j,vec)

            matrix[meshToVec(i-1,j-1),:] += assembler(T[0,:] - T[3,:])

            matrix[meshToVec(i-1,j),:] += assembler(-T[0,:] - T[1,:])


            matrix[meshToVec(i,j),:] += assembler(T[1,:] - T[2,:])

            matrix[meshToVec(i,j-1),:] += assembler(T[2,:] + T[3,:])
    for i in range(cell_centers.shape[0]):
        for j in range(cell_centers.shape[1]):
            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                matrix[meshToVec(i,j),:] = 0

                matrix[meshToVec(i,j),meshToVec(i,j)] = 1

    return matrix



def compute_vector(mesh,f,boundary):
    nodes = mesh.nodes
    cell_centers = mesh.cell_centers
    num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]
    nx = nodes.shape[1]
    ny = nodes.shape[0]
    meshToVec = mesh.meshToVec
    vecToMesh = mesh.vecToMesh
    vector = np.zeros(num_unknowns)
    h_y = nodes[1,0,1]-nodes[0,0,1]
    for i in range(cell_centers.shape[0]):
        for j in range(cell_centers.shape[1]):
            base = nodes[i,j+1,0]-nodes[i,j,0]
            top = nodes[i+1,j+1,0]-nodes[i+1,j,0]
            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                vector[meshToVec(i,j)]= boundary(cell_centers[i,j,0],cell_centers[i,j,1])
                continue
            vector[meshToVec(i,j)] += h_y*0.5*(base+top)*f(cell_centers[i,j,0],cell_centers[i,j,1])
    return vector






