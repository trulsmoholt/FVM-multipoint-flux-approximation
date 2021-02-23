import numpy as np
from mesh import Mesh
from FVMO import compute_matrix as compute_matrix_o
from FVML import compute_matrix, compute_vector
from differentiation import gradient, divergence
from flux_error import compute_flux_error
import sympy as sym
import math
import matplotlib.pyplot as plt
import random
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve
import time as time


K = np.array([[1,0],[0,1]])
transform = np.array([[1,0],[0.1,1]])
nx = 12
ny = 12
x = sym.Symbol('x')
y = sym.Symbol('y')
u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)


#u_fabric = (x*y*(1-x)*(1-y))
source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
print(source)
source = sym.lambdify([x,y],source)
u_lam = sym.lambdify([x,y],u_fabric)


#T = lambda x,y: (0.9*y+0.1)*math.sqrt(x) + (0.9-0.9*y)*x**2
#T = lambda p: np.array([p[0]*0.1*p[1]+p[0],p[1]+p[1]*0.1*p[0]])
T = lambda p: np.array([p[0],p[1]])



def random_perturbation(h):
    return lambda p: np.array([random.uniform(0,h)*random.choice([-1,1]) + p[0] - 0.32*p[1],random.uniform(0,h)*random.choice([-1,1]) + p[1]])

mesh = Mesh(6,6,random_perturbation(1/20))
num_unknowns = mesh.cell_centers.shape[0]*mesh.cell_centers.shape[1]
matrix = lil_matrix((num_unknowns,num_unknowns))
flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
A,fx,fy = compute_matrix(mesh,K,matrix,flux_matrix)
A = csr_matrix(A,dtype=float)
f = compute_vector(mesh,source,u_lam)
u = spsolve(A,f)
mesh.plot()
mesh.plot_vector(u)

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

def run_test_l(K,source,u_fabric,n):
    T = random_perturbation(0.5/(n))
    mesh = Mesh(n,5*n,T)
    start = time.time()
    num_unknowns = mesh.cell_centers.shape[0]*mesh.cell_centers.shape[1]
    matrix = lil_matrix((num_unknowns,num_unknowns))
    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    A,fx,fy = compute_matrix(mesh,K,matrix,flux_matrix)
    end = time.time()
    print('matrix assembly l-scheme: ',end-start)
    f = compute_vector(mesh,source,sym.lambdify([x,y],u_fabric))
    A  = csr_matrix(A,dtype = float)
    start = time.time()

    u = spsolve(A,f)
    end = time.time()
    print('linear solver l-scheme: ',end-start)

    fx = csr_matrix(fx,dtype=float)
    fy = csr_matrix(fy,dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,sym.lambdify([x,y],u_fabric))
    return (l2err,maxerr,l2err_flux,maxerr_flux)

result_pressure = np.zeros((3,5))
result_flux = np.zeros((3,5))

for i in range(3,6):
    result_pressure[i-3,0] = math.log(2**i,2)
    result_flux[i-3,0] = math.log(2**i,2)
    l2err, maxerr,l2err_flux,maxerr_flux = run_test_l(K,source,u_fabric,2**i)
    result_pressure[i-3,1]=math.log(l2err,2)
    result_pressure[i-3,2]=math.log(maxerr,2) 
    result_flux[i-3,1] = math.log(l2err_flux,2)
    result_flux[i-3,2] = math.log(maxerr_flux,2) 

def run_test_o(K,source,u_fabric,n):
    T = random_perturbation(0.5/(n))
    mesh = Mesh(n,5*n,T)

    num_unknowns = mesh.cell_centers.shape[0]*mesh.cell_centers.shape[1]
    matrix = lil_matrix((num_unknowns,num_unknowns))
    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    A,fx,fy = compute_matrix_o(mesh,K,matrix,flux_matrix)
    A = csr_matrix(A,dtype=float)
    f = compute_vector(mesh,source,sym.lambdify([x,y],u_fabric))
    u = spsolve(A,f)

    fx = csr_matrix(fx,dtype=float)
    fy = csr_matrix(fy,dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,sym.lambdify([x,y],u_fabric))
    return (l2err,maxerr,l2err_flux,maxerr_flux)

for i in range(3,6):
    l2err, maxerr,l2err_flux,maxerr_flux = run_test_o(K,source,u_fabric,2**i)
    result_pressure[i-3,3]=math.log(l2err,2)
    result_pressure[i-3,4]=math.log(maxerr,2) 
    result_flux[i-3,3] = math.log(l2err_flux,2)
    result_flux[i-3,4] = math.log(maxerr_flux,2)


ax = plt.subplot(1,1,1)
p1, = ax.plot(result_pressure[:,0],result_pressure[:,1],'--')
p1.set_label('L-method $L_2$')
p2, = ax.plot(result_pressure[:,0],result_pressure[:,2],'o-')
p2.set_label('L-method $max$')
p3, = ax.plot(result_pressure[:,0],result_pressure[:,3])
p3.set_label('O-method $L_2$')
p4,=ax.plot(result_pressure[:,0],result_pressure[:,4],'x-')
p4.set_label('O-method $max$')
ax.legend()
plt.grid()
plt.xlabel('$log_2 n$')
plt.ylabel('$log_2 e$')
plt.savefig('perturbed_grid_aspect_1d5_extra_perturbed_1.pdf')
plt.show()


ax = plt.subplot(1,1,1)
p1, = ax.plot(result_pressure[:,0],result_flux[:,1],'--')
p1.set_label('L-method $L_2$')
p2, = ax.plot(result_pressure[:,0],result_flux[:,2],'o-')
p2.set_label('L-method $max$')
p3, = ax.plot(result_pressure[:,0],result_flux[:,3])
p3.set_label('O-method $L_2$')
p4,=ax.plot(result_pressure[:,0],result_flux[:,4],'x-')
p4.set_label('O-method $max$')
ax.legend()
plt.grid()
plt.xlabel('$log_2 n$')
plt.ylabel('$log_2 e$')
plt.savefig('perturbed_grid_aspect_1d5_extra_perturbed_flux_1.pdf')
plt.show()

