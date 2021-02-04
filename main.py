import numpy as np
from mesh import Mesh
from FVMO import compute_matrix, compute_vector
from differentiation import gradient, divergence
import sympy as sym
import math
import matplotlib.pyplot as plt

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



#T1 = lambda x,y: (0.9*y+0.1)*math.sqrt(x) + (0.9-0.9*y)*x**2
T = lambda x,y: x + 0.3*y

mesh = Mesh(4,4,T)

mesh.plot()
print('noden ',mesh.nodes[1,1])
print('har cellecenter ',mesh.cell_centers[1,1])
print('og midpunkter ',mesh.midpoints[1,1,:,:])
print('og normalvektorer ',mesh.normals[1,1,:,:])





def compute_error(mesh,u,u_fabric):
    cx = mesh.cell_centers.shape[1]
    cy = mesh.cell_centers.shape[0]
    u_fabric_vec = u.copy()
    volumes = u.copy()

    for i in range(cy):
        for j in range(cx):
            u_fabric_vec[mesh.meshToVec(i,j)] = u_fabric(mesh.cell_centers[i,j,0],mesh.cell_centers[i,j,1])
            volumes[mesh.meshToVec(i,j)] = mesh.volumes[i,j]
    mesh.plot_vector(u-u_fabric_vec,'error')
    err = math.sqrt(np.square(u-u_fabric_vec).T@volumes/(np.ones(volumes.shape).T@volumes))
    print(err)
    return err

def run_test(K,source,u_fabric,n,T):
    mesh = Mesh(n,n,T)
    mesh.plot()
    A = compute_matrix(mesh,K)
    f = compute_vector(mesh,source,u_fabric)
    u = np.linalg.solve(A,f)
    mesh.plot_vector(u,'computed solution')
    mesh.plot_funtion(u_fabric,'exact solution')
    return compute_error(mesh,u,u_fabric)

# result = np.zeros((4,2))
# for i in range(3,7):
#     result[i-3,0] = math.log(2**i,2)
#     result[i-3,1] = math.log(run_test(K,source,u_lam,2**i,T),2)

plt.plot(result[:,0],result[:,1],'o-')

plt.show()
