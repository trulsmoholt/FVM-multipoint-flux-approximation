import numpy as np
import sympy as sym
import math
import random
from scipy.sparse import csr_matrix,lil_matrix
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from FVML import compute_matrix as stiffness, compute_vector
from mesh import Mesh
from operators import gravitation_matrix, mass_matrix

from differentiation import gradient, divergence

def run_richards(num_nodes):

    #Construct the data given a solution
    K = np.array([[1,0],[0,1]])
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    t = sym.Symbol('t')
    p = sym.Symbol('p')
    u_exact = -0.1*sym.sin(math.pi*x)*sym.sin(math.pi*y)*t**2-1
    # u_exact = -(t)**2*(x*y*(1-x)*(1-y))**2 - 1
    # u_exact = (t**2*sym.cos(y*math.pi)*sym.cosh(x*math.pi))/12-1
    K = p**2
    theta = 1/(1-p)
    # theta = p
    f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact ,[x,y]),[x,y],K.subs(p,u_exact))
    # f = sym.diff(u_exact,t)-divergence(gradient(u_exact ,[x,y]),[x,y])
    
    f = sym.lambdify([x,y,t],f)
    u_exact = sym.lambdify([x,y,t],u_exact)


    #Define the mesh
    def random_perturbation(h,aspect):
        return lambda p: np.array([random.uniform(0,h)*random.choice([-1,1]) + p[0] - 0.5*p[1],(1/aspect)*random.uniform(0,h)*random.choice([-1,1]) + p[1]])

    T = lambda p: np.array([p[0]+0.5*p[1],p[1]])
    nx = num_nodes
    ny = num_nodes


    mesh = Mesh(nx,ny,T,ghostboundary=True)

    h = mesh.max_h()
    timestep = h**2 #cfl condition
    print('h ',h)

    #make mass and gravitation matrices
    mass = mass_matrix(mesh,sparse=True)
    #gravitation = gravitation_matrix(mesh)


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

    #Time discretization
    time_partition = np.linspace(0,1,math.ceil(1/timestep))
    tau = time_partition[1]-time_partition[0]


    L = 0.5
    TOL = 0.000000005

    K = lambda x: x**2

    theta = lambda x: 1/(1-x)
    # theta = lambda x:x

    u = mesh.interpolate(lambda x,y:u_exact(x,y,0)) #Initial condition

    def L_scheme(u_j,u_n,TOL,L,K,tau,F):
        A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
        rhs = L*mass@u_j + mass@theta(u_n)-mass@theta(u_j) + tau*F
        permability = K(np.reshape(u_j,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F'))
        lhs = tau*stiffness(mesh,np.eye(2),A,k_global=permability)+L*mass
        u_j_n = np.linalg.solve(lhs,rhs)
        # print(np.linalg.norm(u_j_n-u_j))
        if np.linalg.norm(u_j_n-u_j)>TOL + TOL*np.linalg.norm(u_j_n):
            return L_scheme(u_j_n,u_n,TOL,L,K,tau,compute_vector(mesh,lambda x,y: f(x,y,t),lambda x,y:u_exact(x,y,t)))
        else:
            return u_j_n

    # for t in time_partition[1:]:
    #     u =  L_scheme(u,u,TOL,L,K,tau,compute_vector(mesh,lambda x,y: f(x,y,t),lambda x,y:u_exact(x,y,t)))
    #     mesh.plot_vector(u,'numerical solution')
    #     mesh.plot_funtion(lambda x,y: u_exact(x,y,t),'exact solution')
    #     e = u - mesh.interpolate(lambda x,y: u_exact(x,y,t))
    #     mesh.plot_vector(e,'error')
    #     print(t)
    #     (l2,m) = compute_error(mesh,u,lambda x,y:u_exact(x,y,t))
    #     print(l2)
    #     print(m)
    for t in time_partition[1:]:
        source = compute_vector(mesh,lambda x,y:f(x,y,t),lambda x,y:u_exact(x,y,t))
        A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))
        permability = K(np.reshape(u.copy(),(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F'))
        # permability = np.ones((mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F')
        A = stiffness(mesh,np.eye(2),A,k_global=permability)
        A = csr_matrix(A,dtype=float)
        #A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
        rhs = tau*source+L*mass@u
        lhs = tau*A+L*mass
        u_n_i = spsolve(lhs,rhs)
        # u_n_i = np.reshape(u_n_i,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
        # u_n_i = np.ravel(u_n_i,order='F')
        u_n = u
        while np.linalg.norm(u_n_i-u_n)>TOL+TOL*np.linalg.norm(u_n):
            u_n = u_n_i
            A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))
            permability = K(np.reshape(u_n.copy(),(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F'))
            # permability = np.ones((mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]),order='F')

            A = stiffness(mesh,np.eye(2),A,k_global=permability)
            A = csr_matrix(A,dtype=float)
            lhs = tau*A+L*mass
            rhs = -mass@theta(u_n)+mass@theta(u)+tau*source+L*mass@u_n
            u_n_i = spsolve(lhs,rhs)
            # u_n_i = np.reshape(u_n_i,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
            # u_n_i = np.ravel(u_n_i,order='F')
        u = u_n_i
        # mesh.plot_vector(u,'numerical solution')
        # mesh.plot_funtion(lambda x,y: u_exact(x,y,t),'exact solution')
        # e = u - mesh.interpolate(lambda x,y: u_exact(x,y,t))
        # mesh.plot_vector(e,'error')
        # print(t)
        # (l2,m) = compute_error(mesh,u,lambda x,y:u_exact(x,y,t))
        # print(l2)
        # print(m)
    
    return ((h,timestep),compute_error(mesh,u,lambda x,y:u_exact(x,y,1)))

n = 4
arr = np.zeros((n-1,3))
for i in range(1,n):
    num_nodes = 3*2**i
    a,err = run_richards(num_nodes)
    arr[i-1,:2] = np.array([math.log(1/a[0],2),math.log(err[0],2)])
    if i>1:
        arr[i-1,2] = (arr[i-1,1]-arr[i-2,1])/(arr[i-1,0]-arr[i-2,0])
        print('convergence',arr[i-1,2])
print(arr)
plt.plot(arr[:,0],arr[:,1])
plt.show()

def run_heat(timestep,num_nodes):
    #Construct the data given a solution
    K = np.array([[1,0],[0,1]])
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    t = sym.Symbol('t')
    u_exact = -0.1*sym.sin(math.pi*x)*sym.sin(math.pi*y)*t-1
    # u_exact = -t*x*y*(1-x)*(1-y)
    f = sym.diff(u_exact,t)-divergence(gradient(u_exact ,[x,y]),[x,y])
    f = sym.lambdify([x,y,t],f)
    u_exact = sym.lambdify([x,y,t],u_exact)


    #Define the mesh
    def random_perturbation(h,aspect):
        return lambda p: np.array([random.uniform(0,h)*random.choice([-1,1]) + p[0] - 0.5*p[1],(1/aspect)*random.uniform(0,h)*random.choice([-1,1]) + p[1]])

    T = lambda p: np.array([p[0],p[1]])
    nx = num_nodes
    ny = num_nodes
    mesh = Mesh(nx,ny,T,ghostboundary=True)
    h = np.linalg.norm(mesh.nodes[1,1,:]-mesh.nodes[0,0,:])
    print('h ',h)

    #make mass and gravitation matrices
    mass = mass_matrix(mesh)


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

    #Time discretization
    time_partition = np.linspace(0,1,math.ceil(1/timestep))
    tau = time_partition[1]-time_partition[0]

    u = mesh.interpolate(lambda x,y: u_exact(x,y,0))

    for t in time_partition[1:]:
        A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
        lhs = tau*stiffness(mesh,K,A) + mass
        rhs = tau*compute_vector(mesh,lambda x,y:f(x,y,t),lambda x,y: u_exact(x,y,t)) + mass@u
        u = np.linalg.solve(lhs,rhs)
        # mesh.plot_vector(u,'numerical solution')
        # mesh.plot_funtion(lambda x,y:u_exact(x,y,t),'exact solution')
        # e = u - mesh.interpolate(lambda x,y: u_exact(x,y,t))
        # mesh.plot_vector(e,'error')
    return compute_error(mesh,u,lambda x,y:u_exact(x,y,1))

(l2,m) = run_heat(1/4,2+3)
print('l2 ',l2, ' max ',m)
old_l2 = l2
old_m = m
(l2,m) = run_heat(1/4,2**2+3)
print('l2 ',l2, ' max ',m,'improvement',old_l2/l2)
old_l2 = l2
old_m = m
(l2,m) = run_heat(1/4,2**3+3)
print('l2 ',l2, ' max ',m,'improvement',old_l2/l2)
old_l2 = l2
old_m = m
(l2,m) = run_heat(1/4,2**4+3)
print('l2 ',l2, ' max ',m,'improvement',old_l2/l2)
old_l2 = l2
old_m = m
(l2,m) = run_heat(1/4,2**5+3)
print('l2 ',l2, ' max ',m,'improvement',old_l2/l2)
old_l2 = l2
old_m = m
(l2,m) = run_heat(1/4,2**6+3)
print('l2 ',l2, ' max ',m,'improvement',old_l2/l2)
old_l2 = l2
old_m = m
(l2,m) = run_heat(1/4,2**7+3)
print('l2 ',l2, ' max ',m,'improvement',old_l2/l2)
