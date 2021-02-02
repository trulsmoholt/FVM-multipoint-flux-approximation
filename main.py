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
nx = 10
ny = 15
x = sym.Symbol('x')
y = sym.Symbol('y')
u_fabric = (-x*y*(1-x)*(1-y))
f = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
f = sym.lambdify([x,y],f)

u_lam = sym.lambdify([x,y],u_fabric)




mesh = Mesh(nx,ny,lambda x,y: (0.9*y+0.1)*math.sqrt(x) + (0.9-0.9*y)*x**2)
mesh.plot()
A = compute_matrix(mesh, K)
f = compute_vector(mesh,f,u_lam)

u = np.linalg.solve(A,f)
mesh.plot_function(u)