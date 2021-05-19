import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sym

from differentiation import gradient,divergence
from mesh import Mesh
from FEM import compute_matrix,compute_vector
K = np.array([[1,0],[0,1]])

x = sym.Symbol('x')
y = sym.Symbol('y')
u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)


#u_fabric = (x*y*(1-x)*(1-y))
source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
print(source)
source = sym.lambdify([x,y],source)
u_lam = sym.lambdify([x,y],u_fabric)



mesh = Mesh(5,5,lambda p: np.array([p[0],p[1]]))



A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
flux_matrix = {'x':np.zeros((mesh.elements.shape[0],mesh.num_unknowns)),'y': np.zeros((mesh.elements.shape[0],mesh.num_unknowns))}
A,fxx,fyy = compute_matrix(mesh,K,A,flux_matrix)
F = compute_vector(mesh,source,u_lam)

u = np.linalg.solve(A,F)
import matplotlib.tri as mtri
fx = fxx@u
fy = fyy@u
flux = np.stack((fx,fy),1)
flux = -flux
print(flux)
coordinates = np.reshape(mesh.cell_centers,(mesh.num_unknowns,2),order='C')
elements = mesh.elements.astype(int)

pos = np.zeros((mesh.elements.shape[0],2))
for i,e in enumerate(elements):
    pos[i,:] = (coordinates[e[0]]+coordinates[e[1]] + coordinates[e[2]])/3
fig = plt.figure(figsize=(10,5))

points = np.reshape(mesh.cell_centers,(mesh.cell_centers.shape[0]*mesh.cell_centers.shape[1],2))
tri = mtri.Triangulation(points[:,0], points[:,1], mesh.elements)
interpolator = mtri.LinearTriInterpolator(tri,u)

(dx,dy) = interpolator.gradient(pos[:,0],pos[:,1])


Nx = np.array([-1,0])
Ny = np.array([0,1])
Ndiag = np.array([-1,2])/np.linalg.norm(np.array([-1,2]))

D = np.stack((-dx,-dy),1)

V = np.zeros((6,2))
V[0,:] = Ndiag.dot(D[9,:])*Ndiag.T
V[1,:] = Ndiag.dot(D[8,:])*Ndiag.T
V[3,:] = Nx.dot(D[7,:])*Nx.T
V[2,:] = Nx.dot(D[8,:])*Nx.T
V[5,:] = Ny.dot(D[8,:])*Ny.T
V[4,:] = Ny.dot(D[15,:])*Ny.T
print(V)
pos1 = np.array([[0.5,0.25],[0.5,0.25],[0.375,0.25],[0.375,0.25],[0.5,0.31],[0.5,0.31]])
ix = np.array([0,2,3,4,5])

plt.quiver(pos1[ix,0],pos1[ix,1],V[ix,0],V[ix,1],zorder=10,color=['black','white','black','black','white'],label='normal flow density outside cell')
plt.quiver(pos1[1,0],pos1[1,1],V[1,0],V[1,1],zorder=10,color=['white'],label='normal flow density inside cell')
plt.legend()


plt.triplot(tri,color = 'red',linestyle = 'dashed',zorder = 5)
plt.tricontourf(tri,u,20)
plt.savefig('figs5/fem_flux.pdf')
plt.show()




x = np.linspace(0,6,50)
y = np.sin(x)

x_approx = np.linspace(0,6,5)
y_approx = np.sin(x_approx)

plt.plot(x,y,label='solution u')
plt.plot(x_approx,y_approx,label='linear interpolation of u')
plt.legend()
plt.savefig('figs5/interpolation.pdf')
plt.show()