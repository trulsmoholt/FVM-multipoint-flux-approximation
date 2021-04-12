import numpy as np

def mass_matrix(mesh):
    mass = np.diag(np.ravel(mesh.volumes,order='F'))
    for i in range(mesh.cell_centers.shape[0]):
        for j in range(mesh.cell_centers.shape[1]):
            if (i==0) or (i==mesh.num_nodes_x-2) or (j==0) or (j==mesh.num_nodes_y-2):
                mass[mesh.meshToVec(i,j),:] = 0
    return mass

def gravitation_matrix(mesh):
    matrix = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    nodes = mesh.nodes
    normals = mesh.normals
    gravity = np.array([0,1])
    meshToVec = mesh.meshToVec
    for i in range(1,nodes.shape[0]-2):
        for j in range(1,nodes.shape[1]-2):
            flux_e = 2*normals[i,j,0,:].T@gravity
            flux_s = 2*normals[i,j,1,:].T@gravity
            flux_w = 2*normals[i,j+1,0,:].T@gravity
            flux_n = 2*normals[i+1,j,1,:].T@gravity
            matrix[meshToVec(i,j),meshToVec(i,j)] += -flux_e - flux_s + flux_n + flux_w
            matrix[meshToVec(i,j),meshToVec(i-1,j)] += flux_s
            matrix[meshToVec(i,j),meshToVec(i,j-1)] += flux_e
            matrix[meshToVec(i,j),meshToVec(i,j+1)] -= flux_w
            matrix[meshToVec(i,j),meshToVec(i+1,j)] -= flux_n
            if (i==0) or (i==nodes.shape[0]-2) or (j==0) or (j==nodes.shape[1]-2):
                matrix[meshToVec(i,j),:] = 0
    return matrix