import numpy as np


from scipy.sparse import csr_matrix,lil_matrix

from mesh import Mesh


def compute_matrix(mesh,K,matrix,compute_flux=None):
    nodes = mesh.nodes
    cell_centers = mesh.cell_centers
    k_global = np.ones((cell_centers.shape[0],cell_centers.shape[1]))

    k_global = np.ones((cell_centers.shape[0],cell_centers.shape[1]))
    nx = nodes.shape[1]
    ny = nodes.shape[0]

    num_unknowns = cell_centers.shape[1]*cell_centers.shape[0]

    meshToVec = mesh.meshToVec
    if compute_flux is not None:
        flux_matrix_x = compute_flux['x']
        flux_matrix_y = compute_flux['y']

    def local_assembler(j,i,vec,matrix_handle,index):
        global_vec = np.zeros(num_unknowns)
        
        matrix_handle[index,meshToVec(j-1,i-1)] += vec[0]
        matrix_handle[index,meshToVec(j-1,i)] += vec[1]
        matrix_handle[index,meshToVec(j,i)] += vec[2]
        matrix_handle[index,meshToVec(j,i-1)] += vec[3]


    for i in range(1,nodes.shape[0]-1):
        for j in range(1,nodes.shape[1]-1):
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


            #print(omega)
            

            grad_u_1 = (centers[1]-centers[0])/np.linalg.norm(centers[1]-centers[0])**2
            grad_u_2 = (centers[2]-centers[1])/np.linalg.norm(centers[2]-centers[1])**2
            grad_u_3 = (centers[3]-centers[2])/np.linalg.norm(centers[3]-centers[2])**2
            grad_u_4 = (centers[0]-centers[3])/np.linalg.norm(centers[0]-centers[3])**2
            T = np.array([[-n[0,:].T@K@grad_u_1,n[0,:].T@K@grad_u_1,0,0],
                          [0    ,-n[1,:].T@K@grad_u_2,n[1,:].T@K@grad_u_2,0],
                            [0  ,0  ,-n[2,:].T@K@grad_u_3,n[2,:].T@K@grad_u_3,],
                            [n[3,:].T@K@grad_u_4,0,0,-n[3,:].T@K@grad_u_4]])

            assembler = lambda vec,matrix,cell_index: local_assembler(i,j,vec,matrix,cell_index)

            assembler(-T[0,:]-T[3,:],matrix,meshToVec(i-1,j-1))

            assembler( -T[1,:]  +T[0,:],matrix,meshToVec(i-1,j))

            assembler(T[2,:]+T[1,:],matrix,meshToVec(i,j))

            assembler( +T[3,:]-T[2,:],matrix,meshToVec(i,j-1))

            if compute_flux is not None:
                assembler(-T[0,:],flux_matrix_x,meshToVec(i-1,j-1))
                assembler(-T[2,:],flux_matrix_x,meshToVec(i,j-1))
                assembler(-T[3,:],flux_matrix_y,meshToVec(i-1,j-1))
                assembler(-T[1,:],flux_matrix_y,meshToVec(i-1,j))


    for i in range(cell_centers.shape[0]):
        for j in range(cell_centers.shape[1]):
            if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
                matrix[meshToVec(i,j),:] = 0

                matrix[meshToVec(i,j),meshToVec(i,j)] = 1
    if compute_flux is not None:
        return (matrix, flux_matrix_x, flux_matrix_y)
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

if __name__=='__main__':
    import sympy as sym
    from differentiation import gradient,divergence
    from FVMO import compute_matrix as O
    import math
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    K = np.array([[1,0],[0,1]])
    u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
    source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
    source = sym.lambdify([x,y],source)
    u_lam = sym.lambdify([x,y],u_fabric)

    mesh = Mesh(6,6,lambda p: np.array([p[0] ,p[1]]))
    mesh.plot()
    AT = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    flux_matrixT = {'x': np.zeros((mesh.num_unknowns,mesh.num_unknowns)),'y':np.zeros((mesh.num_unknowns,mesh.num_unknowns))}
    AT,fxT,fyT = compute_matrix(mesh,np.array([[1,0],[0,1]]),AT,flux_matrixT)
    AO = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    flux_matrixO = {'x': np.zeros((mesh.num_unknowns,mesh.num_unknowns)),'y':np.zeros((mesh.num_unknowns,mesh.num_unknowns))}
    AO,fxO,fyO = O(mesh,np.array([[1,0],[0,1]]),AO,flux_matrixO)
    diff = fyO-fyT
    sumA = AO+AT
    f = compute_vector(mesh,source,u_lam)
    ut = np.linalg.solve(AT,f)
    mesh.plot_vector(ut,'TPFA')
    f = compute_vector(mesh,source,u_lam)
    uo = np.linalg.solve(AT,f)
    mesh.plot_vector(uo,'MPFA-O')
    mesh.plot_vector(ut-uo,'difference')
    mesh.plot_funtion(u_lam,'exact solution')



