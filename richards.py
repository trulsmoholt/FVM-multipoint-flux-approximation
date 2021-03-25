import numpy as np
import sympy as sym
import math
from numba import jit

from FVML import compute_matrix as stiffness, compute_vector
from mesh import Mesh

from differentiation import gradient, divergence

K = np.array([[1,0],[0,1]])
x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')
p = sym.Symbol('p')
u_exact = -(t)*x*y*(1-x)*(1-y) -1


K = p**2
theta = 1/(1-p)
# f = sym.diff(u_exact,t,1)-(sym.diff(u_exact,x,2)+sym.diff(u_exact,y,2))
f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact ,[x,y]),[x,y],K.subs(p,u_exact))
f = sym.lambdify([x,y,t],f)
u_exact = sym.lambdify([x,y,t],u_exact)

T = lambda p: np.array([p[0],p[1]])

nx = 6
ny = 6
mesh = Mesh(nx,ny,T)
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
    #mesh.plot_vector(u-u_fabric_vec,'error')
    L2err = math.sqrt(np.square(u-u_fabric_vec).T@volumes/(np.ones(volumes.shape).T@volumes))
    maxerr = np.max(np.abs(u-u_fabric_vec))
    return (L2err,maxerr)



time_partition = np.linspace(0,1,16)
tau = time_partition[1]-time_partition[0]
u = mesh.interpolate(lambda x,y:u_exact(x,y,0))
mesh.plot_vector(u)


flux_matrix = {'x': np.zeros((mesh.num_unknowns,mesh.num_unknowns)),'y':np.zeros((mesh.num_unknowns,mesh.num_unknowns))}
permability = np.ones((mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))

L = 3
TOL = 0.00005

K = lambda x: x**2

theta = lambda x: 1/(1-x)


for t in time_partition[1:]:
    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))

    rhs = tau*compute_vector(mesh,lambda x,y:f(x,y,t),lambda x,y:u_exact(x,y,t))+L*mass@u
    lhs = tau*stiffness(mesh,K,A,k_global=K(np.reshape(u,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))))+L*mass
    u_n_i = np.linalg.solve(lhs,rhs)
    u_n = u
    while np.linalg.norm(u_n_i-u_n)>TOL+TOL*np.linalg.norm(u_n):
        u_n = u_n_i
        A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
        lhs = tau*stiffness(mesh,K,A,k_global=K(np.reshape(u_n,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))))+L*mass
        rhs = -mass@theta(u_n)+mass@theta(u)+tau*compute_vector(mesh,lambda x,y:f(x,y,t),lambda x,y:u_exact(x,y,t))+L*mass@u_n
        u_n_i = np.linalg.solve(lhs,rhs)
        print(np.linalg.norm(u_n-u))
    u = u_n

    # rhs = tau*compute_vector(mesh,lambda x,y:f(x,y,t),lambda x,y:u_exact(x,y,t))+mass@u
    # lhs = tau*stiffness(mesh,K,A,k_global=permability)+mass
    # u = np.linalg.solve(lhs,rhs)
    mesh.plot_vector(u)
    mesh.plot_vector(mesh.interpolate(lambda x,y:u_exact(x,y,t)),'u exact')
    (l2,m) = compute_error(mesh,u,lambda x,y:u_exact(x,y,t))
    print(l2)



