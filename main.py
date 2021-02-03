import numpy as np
from mesh import Mesh
from FVMO import compute_matrix, compute_vector
from differentiation import gradient, divergence
import sympy as sym
import math
import matplotlib.pyplot as plt

K = np.array([[1,0],[0,1]])
nx = 6
ny = 6
x = sym.Symbol('x')
y = sym.Symbol('y')
u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
f = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
source = sym.lambdify([x,y],f)

u_lam = sym.lambdify([x,y],u_fabric)



T1 = lambda x,y: (0.9*y+0.1)*math.sqrt(x) + (0.9-0.9*y)*x**2
T = lambda x,y: x

    
def compute_error(mesh,u,u_fabric):
    cx = mesh.cell_centers.shape[1]
    cy = mesh.cell_centers.shape[0]
    u_fabric_vec = u.copy()
    volumes = u.copy()

    for i in range(cy):
        for j in range(cx):
            u_fabric_vec[mesh.meshToVec(i,j)] = u_fabric(mesh.cell_centers[i,j,0],mesh.cell_centers[i,j,1])
            volumes[mesh.meshToVec(i,j)] = mesh.volumes[i,j]

    return math.sqrt(np.square(u-u_fabric_vec).T@volumes/(np.ones(volumes.shape).T@volumes))

def run_test(K,source,u_fabric,n,T):
    mesh = Mesh(n,n,T)
    A = compute_matrix(mesh,K)
    f = compute_vector(mesh,source,u_fabric)
    u = np.linalg.solve(A,f)
    return compute_error(mesh,u,u_fabric)

result = np.zeros((4,2))
for i in range(3,7):
    result[i-3,0] = math.log(2**i,2)
    result[i-3,1] = math.log(run_test(K,source,u_lam,2**i,T),2)

plt.plot(result[:,0],result[:,1],'o-')

plt.show()
