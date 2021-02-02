#from FVMO import run

# steps = [1/4,1/8,1/16,1/32,1/64,1/128,1/(2*128),1/(4*128)]
# L2_errors = []
# print(matrix)
# for h in steps:
#     print('stepsize:    ',h)
#     L2_errors.append(run(h))
#     if len(L2_errors)>1:
#         print('improvement: ',L2_errors[-2]/L2_errors[-1])
import numpy as np

from mesh import Mesh
from FVMO import compute_matrix, compute_vector
from differentiation import gradient, divergence
import sympy as sym
import math

K = np.array([[1,0],[0,1]])
nx = 24
ny = 24
x = sym.Symbol('x')
y = sym.Symbol('y')
u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
f = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
f = sym.lambdify([x,y],f)

u_lam = sym.lambdify([x,y],u_fabric)



T = lambda x,y: (0.9*y+0.1)*math.sqrt(x) + (0.9-0.9*y)*x**2
mesh = Mesh(nx,ny,lambda x,y:0.001*y*x + x)
mesh.plot()
A = compute_matrix(mesh, K)
f = compute_vector(mesh,f,u_lam)

u = np.linalg.solve(A,f)






def compute_error(mesh,u,u_fabric):
    cx = mesh.cell_centers.shape[1]
    cy = mesh.cell_centers.shape[0]
    u_fabric_vec = u.copy()
    volumes = u.copy()

    for i in range(cy):
        for j in range(cx):
            u_fabric_vec[mesh.meshToVec(i,j)] = u_fabric(mesh.cell_centers[i,j,0],mesh.cell_centers[i,j,1])
            volumes[mesh.meshToVec(i,j)] = mesh.volumes[i,j]

    return math.sqrt(np.square(u-u_fabric_vec).T@volumes)/(np.ones(volumes.shape).T@volumes)



mesh.plot_vector(u)
mesh.plot_funtion(u_lam)

print(compute_error(mesh,u,u_lam))
