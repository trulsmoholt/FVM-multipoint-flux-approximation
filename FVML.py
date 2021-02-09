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

    R = np.array([[0,1],[-1,0]])

    T = np.zeros((4,2,3))

    def local_assembler(j,i,vec,start):
        global_vec = np.zeros(num_unknowns)

        indexes = [meshToVec(j-1,i-1),meshToVec(j-1,i),meshToVec(j,i),meshToVec(j,i-1)]


        for ii,jj in zip(range(start,start+2),range(2)):

            global_vec[indexes[ii%4]] = vec[jj]

        global_vec[indexes[(start-1)%4]] = vec[2]
        return global_vec

    def compute_triangle_normals(start_index, interface, centers, node_midpoint):
        V = np.zeros((7,2))

        V[0,:] = R@(interface[(start_index-1)%4,:]-centers[start_index%4])
        V[1,:] = -R@(interface[start_index%4,:]-centers[start_index%4])
        V[2,:] = R@(interface[start_index%4,:]-centers[(start_index+1)%4])
        V[3,:] = -R@(nodes[i,j]-centers[(start_index+1)%4])
        V[4,:] = R@(nodes[i,j]-centers[(start_index-1)%4])
        V[5,:] = -R@(interface[(start_index-1)%4,:]-centers[(start_index-1)%4])
        V[6,:] = R@(nodes[i,j]-centers[start_index%4])

        return V
    
    def compute_T(omega, xi_1, xi_2):
        C = np.array([[-omega[0,0,0],   -omega[0,0,1]],
                [-omega[1,0,0],   -omega[1,0,1]]])

        D = np.array([[omega[0,0,0]+omega[0,0,1],   0,  0],
                        [omega[1,0,0]+omega[1,0,1],   0,  0]])

        A = np.array([[omega[0,0,0]-omega[0,1,3]-omega[0,1,2]*xi_1,  omega[0,0,1]-omega[0,1,2]*xi_2],
                        [omega[1,0,0]-omega[1,2,5]*xi_1,  omega[1,0,1]-omega[1,2,4]-omega[1,2,5]*xi_2]])

        B = np.array([[omega[0,0,0]+omega[0,0,1]+omega[0,1,2]*(1-xi_1-xi_2),    -omega[0,1,2]-omega[0,1,3], 0],
                        [omega[1,0,0]+omega[1,0,1]+omega[1,2,5]*(1-xi_1-xi_2), 0,  -omega[1,2,4]-omega[1,2,5]]])

        T = C@np.linalg.inv(A)@B+D
        return T
    def choose_triangle(T,i):
        if abs(T[i,0,0])<abs(T[(i+1)%4,1,0]):
            return (T[i,0,:],i)
        else:
            return (T[(i+1)%4,1,:],(i+1)%4)


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
            n[3,:] = mesh.normals[i,j-1,1,:]
            centers[3,:] = cell_centers[i,j-1]

            #A
            v = nodes[i-1,j]-nodes[i,j]
            interface[0,:] = nodes[i,j] + 0.5*(v)
            n[0,:] = mesh.normals[i-1,j,0,:]
            centers[0,:] = cell_centers[i-1,j-1]
            #B
            v = nodes[i,j+1]-nodes[i,j]
            interface[1,:] = nodes[i,j] + 0.5*(v)
            n[1,:] = mesh.normals[i,j,1,:]
            centers[1,:] = cell_centers[i-1,j]

            #C
            v = nodes[i+1,j]-nodes[i,j]
            interface[2,:] = nodes[i,j] + 0.5*(v)
            n[2,:] = mesh.normals[i,j,0,:]
            centers[2,:] = cell_centers[i,j]



            k_loc[0] = k_global[i-1,j-1]

            k_loc[1] = k_global[i-1,j]

            k_loc[2] = k_global[i,j]

            k_loc[3] = k_global[i,j-1]




            omega  = np.zeros((2,3,7))
            V = compute_triangle_normals(1,interface,centers,nodes[i,j])
            t = [V[0,:].T@R@V[1,:],V[2,:].T@R@V[3,:],V[4,:].T@R@V[5,:]]

            for ii in range(2):
                for jj in range(3):
                    for kk in range(7):
                        if ii == 0:
                            omega[ii,jj,kk] = n[1,:].T@K@V[kk,:]*1/t[jj]
                        else:
                            omega[ii,jj,kk] = n[0,:].T@K@V[kk,:]*1/t[jj]


            xi_1 = (V[6,:].T@R@V[0,:])/(V[0,:].T@R@V[1,:])
            xi_2 = (V[6,:].T@R@V[1,:])/(V[0,:].T@R@V[1,:])

            T[1,:,:] = compute_T(omega,xi_1,xi_2)




            omega  = np.zeros((2,3,7))
            V = compute_triangle_normals(3,interface,centers,nodes[i,j])
            t = [V[0,:].T@R@V[1,:],V[2,:].T@R@V[3,:],V[4,:].T@R@V[5,:]]
            for ii in range(2):
                for jj in range(3):
                    for kk in range(7):
                        if ii == 0:
                            omega[ii,jj,kk] = n[3,:].T@K@V[kk,:]*1/t[jj]
                        else:
                            omega[ii,jj,kk] = n[2,:].T@K@V[kk,:]*1/t[jj]
                   

            xi_1 = (V[6,:].T@R@V[0,:])/(V[0,:].T@R@V[1,:])
            xi_2 = (V[6,:].T@R@V[1,:])/(V[0,:].T@R@V[1,:])

            T[3,:,:] = compute_T(omega,xi_1,xi_2)

            omega  = np.zeros((2,3,7))
            V = compute_triangle_normals(0,interface,centers,nodes[i,j])
            t = [V[0,:].T@R@V[1,:],V[2,:].T@R@V[3,:],V[4,:].T@R@V[5,:]]
            for ii in range(2):
                for jj in range(3):
                    for kk in range(7):
                        if ii == 0:
                            omega[ii,jj,kk] = n[0,:].T@K@V[kk,:]*1/t[jj]
                        else:
                            omega[ii,jj,kk] = n[3,:].T@K@V[kk,:]*1/t[jj]

            xi_1 = (V[6,:].T@R@V[0,:])/(V[0,:].T@R@V[1,:])
            xi_2 = (V[6,:].T@R@V[1,:])/(V[0,:].T@R@V[1,:])

            T[0,:,:] = compute_T(omega,xi_1,xi_2)

            omega  = np.zeros((2,3,7))
            V = compute_triangle_normals(2,interface,centers,nodes[i,j])
            t = [V[0,:].T@R@V[1,:],V[2,:].T@R@V[3,:],V[4,:].T@R@V[5,:]]
            for ii in range(2):
                for jj in range(3):
                    for kk in range(7):
                        if ii == 0:
                            omega[ii,jj,kk] = n[2,:].T@K@V[kk,:]*1/t[jj]
                        else:
                            omega[ii,jj,kk] = n[1,:].T@K@V[kk,:]*1/t[jj]

            xi_1 = (V[6,:].T@R@V[0,:])/(V[0,:].T@R@V[1,:])
            xi_2 = (V[6,:].T@R@V[1,:])/(V[0,:].T@R@V[1,:])

            T[2,:,:] = compute_T(omega,xi_1,xi_2)

            index = [meshToVec(i-1,j-1),meshToVec(i-1,j),meshToVec(i,j),meshToVec(i,j-1)]
            assembler = lambda vec,center: local_assembler(i,j,vec,center)
            for jj in range(len(index)):
                sgn =( -1 if jj == 2 or jj == 3 else 1)
                t,choice = choose_triangle(T,jj)
                matrix[index[jj],:] += assembler(t*sgn,choice)
                matrix[index[(jj+1)%4],:] += assembler(-t*sgn,choice)





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

            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                vector[meshToVec(i,j)]= boundary(cell_centers[i,j,0],cell_centers[i,j,1])
                continue
            vector[meshToVec(i,j)] += mesh.volumes[i,j]*f(cell_centers[i,j,0],cell_centers[i,j,1])
    return vector





