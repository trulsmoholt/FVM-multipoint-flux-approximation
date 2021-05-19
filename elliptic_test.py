from mesh import Mesh
from differentiation import gradient,divergence
from plot_L import compute_vector,compute_matrix
from FEM import compute_matrix as compute_matrix_FEM
from FEM import compute_vector as compute_vector_FEM


import random
import numpy as np
import math
import sympy as sym

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

def elliptic(n):
    K = np.array([[1,0],[0,1]])
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    k = sym.Piecewise(
        (0.1,x<2*y),
        (10,x>=2*y)
    )
    # k = (sym.sin(math.pi*x)*sym.sin(math.pi*y))+1.5
    u_exact = -x*y*(1-x)*(1-y) -1
    # u_exact = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
    f =-divergence(gradient(u_exact ,[x,y]),[x,y],k)
    f = sym.lambdify([x,y],f)
    u_exact = sym.lambdify([x,y],u_exact)

    T = lambda p: np.array([p[0]-0.5*p[1]+0.4*p[1]**2,p[1]])
    #T = lambda p: np.array([p[0]-0.5*p[1],p[1]])

    def random_perturbation(h,aspect):
        return lambda p: np.array([random.uniform(0,h)*random.choice([-1,1]) + p[0] - 0.5*p[1],(1/aspect)*random.uniform(0,h)*random.choice([-1,1]) + p[1]])


    T2 = random_perturbation(1/(4*n),1)
    mesh = Mesh(n,n,T,centers_at_bc=True)
    mesh.plot()
    # mesh.plot_funtion(f,'source')

    k = sym.lambdify([x,y],k)
    k = mesh.interpolate(k)

    problem = k.copy()
    k = np.ones(k.shape[0])*0.1
    #k = np.random.random(k.shape[0])*0.001
    #k[1:50:4] = 10
    permability = np.reshape(k,(mesh.midpoints.shape[0],mesh.midpoints.shape[1]),order='F')
    #permability[2:6,14:19] = 0.1

    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    A = compute_matrix(mesh,K,A,k_global=permability)

    for i,row in enumerate(A):
        points = len(list(filter(lambda x: abs(x)>0,row)))
        if points > 7:
            print(points,' at ',mesh.vecToMesh(i))
            problem[i] = 13
    

    F = compute_vector(mesh,f,u_exact)
    u = np.linalg.solve(A,F)
    # Af = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    # Af = compute_matrix_FEM(mesh,K,Af,k_global = permability)
    # Ff = compute_vector_FEM(mesh,f,u_exact)
    # uf = np.linalg.solve(Af,Ff)
    u = np.reshape(u,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
    u = np.ravel(u,order='F')
    # mesh.plot_vector(u,'FEM')
    # mesh.plot_funtion(u_exact,'excact')
    # e = u-mesh.interpolate(u_exact)
    # mesh.plot_vector(e,'error')

    # mesh.plot_vector(uf,'FEM')
    # mesh.plot_vector(uf-u,'difference')
    return compute_error(mesh,u,u_exact)
# (l2,m) = elliptic(8)
# print('h',1/6)
# print('l2',l2,'max',m)
(l2,m) = elliptic(6)
print('h',1/12)
print('l2',l2,'max',m)
# (l2,m) = elliptic(26)
# print('h',1/24)
# print('l2',l2,'max',m)
def elliptic_heterogenous(n):
    K = np.array([[1,0],[0,1]])
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    k = np.eye(2)
    u_exact = x*y*(1-x)*(1-y) -1
    # u_exact = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
    f =-divergence(gradient(u_exact ,[x,y]),[x,y])
    print(f)
    f = sym.lambdify([x,y],f)
    u_exact = sym.lambdify([x,y],u_exact)

    T = lambda p: np.array([p[0],p[1]])

    mesh = Mesh(n,n,T,centers_at_bc=True)
    mesh.plot_funtion(f,'source')

    # k = sym.lambdify([x,y],k)
    # k = mesh.interpolate(k)

    # permability = np.reshape(k,(mesh.midpoints.shape[0],mesh.midpoints.shape[1]),order='F')
    #permability[2:6,14:19] = 0.1

    A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    A = compute_matrix(mesh,K,A)
    F = compute_vector(mesh,f,u_exact)
    u = np.linalg.solve(A,F)
    Af = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
    Af = compute_matrix_FEM(mesh,K,Af)
    Ff = compute_vector_FEM(mesh,f,u_exact)
    uf = np.linalg.solve(Af,Ff)
    uf = np.reshape(uf,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
    uf = np.ravel(uf,order='F')
    mesh.plot_vector(u,'FVML')
    mesh.plot_vector(uf,'FEM')

    mesh.plot_funtion(u_exact,'excact')
    e = u-mesh.interpolate(u_exact)
    mesh.plot_vector(e,'error')

    # mesh.plot_vector(uf,'FEM')
    mesh.plot_vector(uf-u,'difference')
    (l2,m) = compute_error(mesh,uf,u_exact)
    print('l2 fem',l2)
    print('max fem',m)
    return compute_error(mesh,u,u_exact)
# (l2,m) = elliptic_heterogenous(10)
# print(l2)
# print(m)