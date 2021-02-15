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
import time as time


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

mesh = Mesh(8*4,8*4,random_perturbation(0.5/(8)))
mesh.plot_funtion(u_lam)

u_x = sym.diff(u_fabric,x)
u_y = sym.diff(u_fabric,y)

p1 = mesh.nodes[2*4,6*5]
d1 = np.array([float(u_x.subs(x,p1[0]).subs(y,p1[1])),float(u_y.subs(y,p1[1]).subs(x,p1[0]) )])
p2 = mesh.nodes[4*3,4*3]
d2 = np.array([float(u_x.subs(x,p2[0]).subs(y,p2[1])),float(u_y.subs(y,p2[1]).subs(x,p2[0]))])
p3 = mesh.nodes[4*6,4*3]
d3 = np.array([float(u_x.subs(x,p3[0]).subs(y,p3[1])),float(u_y.subs(y,p3[1]).subs(x,p3[0]))])
p4 = mesh.nodes[4*2,4*5]
d4 = np.array([float(u_x.subs(x,p4[0]).subs(y,p4[1])),float(u_y.subs(y,p4[1]).subs(x,p4[0]))])
XY = np.vstack((p1,p2,p3,p4))
UV = np.vstack((d1,d2,d3,d4))

print(XY)
print(UV)

X,Y = np.meshgrid(XY[0,:].T,XY[1,:].T)

plt.quiver(XY[:,0],XY[:,1],UV[:,0], UV[:,1])


plt.show()

