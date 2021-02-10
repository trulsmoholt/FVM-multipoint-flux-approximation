import numpy as np
from mesh import Mesh
from FVMO import compute_matrix as compute_matrix_o
from FVML import compute_matrix, compute_vector
from differentiation import gradient, divergence
import sympy as sym
import math
import matplotlib.pyplot as plt
import random
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve

K = np.array([[1,0],[0,1]])
transform = np.array([[1,0],[0.1,1]])
nx = 6
ny = 6
x = sym.Symbol('x')
y = sym.Symbol('y')
u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)


#u_fabric = (x*y*(1-x)*(1-y))
source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
print(source)
source = sym.lambdify([x,y],source)
u_lam = sym.lambdify([x,y],u_fabric)


#T = lambda x,y: (0.9*y+0.1)*math.sqrt(x) + (0.9-0.9*y)*x**2
T = lambda x,y: x + 0.3*y



def random_perturbation(h):
    return lambda x,y: random.uniform(0,h)*random.choice([-1,1]) + x + 0.32*y

mesh = Mesh(8,8*5,random_perturbation(0.5/(8*5)))
mesh.plot()

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

def run_test(K,source,u_fabric,n):
    T = random_perturbation(0.5/(50*n))
    mesh = Mesh(n,50*n,T)
    A = compute_matrix(mesh,K)
    f = compute_vector(mesh,source,u_fabric)
    A  = csr_matrix(A,dtype = float)
    u = spsolve(A,f)
    return compute_error(mesh,u,u_fabric)

result = np.zeros((4,5))
for i in range(3,7):
    result[i-3,0] = math.log(2**i,2)
    l2err, maxerr = run_test(K,source,u_lam,2**i)
    result[i-3,1]=math.log(l2err,2)
    result[i-3,2]=math.log(maxerr,2) 

def run_test(K,source,u_fabric,n):
    T = random_perturbation(0.5/(50*n))
    mesh = Mesh(n,50*n,T)

    A = compute_matrix_o(mesh,K)
    A = csr_matrix(A,dtype=float)
    f = compute_vector(mesh,source,u_fabric)
    u = spsolve(A,f)
    return compute_error(mesh,u,u_fabric)

for i in range(3,7):
    l2err, maxerr = run_test(K,source,u_lam,2**i)
    result[i-3,3]=math.log(l2err,2)
    result[i-3,4]=math.log(maxerr,2) 



ax = plt.subplot(1,1,1)
p1, = ax.plot(result[:,0],result[:,1],'--')
p1.set_label('L-method $L_2$')
p2, = ax.plot(result[:,0],result[:,2],'o-')
p2.set_label('L-method $max$')
p3, = ax.plot(result[:,0],result[:,3])
p3.set_label('O-method $L_2$')
p4,=ax.plot(result[:,0],result[:,4],'x-')
p4.set_label('O-method $max$')
ax.legend()
plt.grid()
plt.xlabel('$log_2 n$')
plt.ylabel('$log_2 e$')
plt.savefig('perturbed_grid_aspect_0,05.pdf')

plt.show()
