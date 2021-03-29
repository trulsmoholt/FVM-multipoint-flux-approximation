import numpy as np
import sympy as sym
import math
import random

from FVML import compute_matrix as stiffness, compute_vector
from mesh import Mesh

from differentiation import gradient, divergence

#Construct the data given a solution
K = np.array([[1,0],[0,1]])
x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')
p = sym.Symbol('p')
u_exact = -(t)*x*y*(1-x)*(1-y) -1
K = p**2
theta = 1/(1-p)
f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact + 5*y ,[x,y]),[x,y],K.subs(p,u_exact))
f = sym.lambdify([x,y,t],f)
u_exact = sym.lambdify([x,y,t],u_exact)


#Define the mesh
def random_perturbation(h,aspect):
    return lambda p: np.array([random.uniform(0,h)*random.choice([-1,1]) + p[0] - 0.5*p[1],(1/aspect)*random.uniform(0,h)*random.choice([-1,1]) + p[1]])

T = lambda p: np.array([p[0],p[1]])
nx = 6
ny = 6
mesh = Mesh(nx,ny,T)


#Make the mass matrix
mass = np.diag(np.ravel(mesh.volumes,order='F'))
for i in range(mesh.cell_centers.shape[0]):
    for j in range(mesh.cell_centers.shape[1]):
        if (i==0) or (i==ny-2) or (j==0) or (j==nx-2):
            mass[mesh.meshToVec(i,j),:] = 0


def compute_error(mesh,u,u_fabric):
    cx = mesh.cell_centers.shape[1]
    cy = mesh.cell_centers.shape[0]
    u_fabric_vec = u.copy()
    volumes = u.copy()

    for i in range(cy):
        for j in range(cx):
            u_fabric_vec[mesh.meshToVec(i,j)] = u_fabric(mesh.cell_centers[i,j,0],mesh.cell_centers[i,j,1])
            volumes[mesh.meshToVec(i,j)] = mesh.volumes[i,j]
    L2err = math.sqrt(np.square(u-u_fabric_vec).T@volumes/(np.ones(volumes.shape).T@volumes))
    maxerr = np.max(np.abs(u-u_fabric_vec))
    return (L2err,maxerr)

def gravitation(mesh,matrix):
    nodes = mesh.nodes
    normals = mesh.normals
    gravity = 5*np.array([0,1])
    meshToVec = mesh.meshToVec
    for i in range(1,nodes.shape[0]-2):
        for j in range(1,nodes.shape[1]-2):
            print(mesh.midpoints[i,j,:])
            flux_e = 2*normals[i,j,0,:].T@gravity
            flux_s = 2*normals[i,j,1,:].T@gravity
            flux_w = 2*normals[i,j+1,0,:].T@gravity
            flux_n = 2*normals[i+1,j,1,:].T@gravity
            matrix[meshToVec(i,j),meshToVec(i,j)] += -flux_e - flux_s + flux_n + flux_w
            matrix[meshToVec(i,j),meshToVec(i-1,j)] += flux_s
            matrix[meshToVec(i,j),meshToVec(i,j-1)] += flux_e
            matrix[meshToVec(i,j),meshToVec(i,j+1)] -= flux_w
            matrix[meshToVec(i,j),meshToVec(i+1,j)] -= flux_n
    for i in range(mesh.cell_centers.shape[0]):
        for j in range(mesh.cell_centers.shape[1]):
            if (i==0) or (i==nodes.shape[0]-2) or (j==0) or (j==nodes.shape[1]-2):
                matrix[meshToVec(i,j),:] = 0
                matrix[meshToVec(i,j),meshToVec(i,j)] = 0
    return matrix

G = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
G = gravitation(mesh,G)
print(G)

mesh.plot()


#Time discretization
time_partition = np.linspace(0,1,16)
tau = time_partition[1]-time_partition[0]




L = 3
TOL = 0.00005

K = lambda x: x**2

theta = lambda x: 1/(1-x)


u = mesh.interpolate(lambda x,y:u_exact(x,y,0)) #Initial condition

for t in time_partition[1:]:
    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    rhs = tau*compute_vector(mesh,lambda x,y:f(x,y,t),lambda x,y:u_exact(x,y,t))+L*mass@u
    lhs = tau*stiffness(mesh,K,A,k_global=K(np.reshape(u,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))))+L*mass
    u_n_i = np.linalg.solve(lhs,rhs)
    u_n = u
    while np.linalg.norm(u_n_i-u_n)>TOL+TOL*np.linalg.norm(u_n):
        u_n = u_n_i
        A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
        permability = K(np.reshape(u_n,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1])))#new permability dependent on previous iteration
        lhs = tau*(stiffness(mesh,K,A,k_global=permability)-G)+L*mass
        rhs = -mass@theta(u_n)+mass@theta(u)+tau*compute_vector(mesh,lambda x,y:f(x,y,t),lambda x,y:u_exact(x,y,t))+L*mass@u_n
        u_n_i = np.linalg.solve(lhs,rhs)
        print(np.linalg.norm(u_n-u))
    u = u_n

    mesh.plot_vector(u,'u numerical')
    mesh.plot_vector(mesh.interpolate(lambda x,y:u_exact(x,y,t)),'u exact')
    (l2,m) = compute_error(mesh,u,lambda x,y:u_exact(x,y,t))
    print(l2)



