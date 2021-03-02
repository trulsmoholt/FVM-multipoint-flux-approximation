import numpy as np
import sympy as sym
import math
import random
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from FVML import compute_matrix as compute_matrix_l
from FVMO import compute_matrix as compute_matrix_o
from TPFA import compute_matrix as compute_matrix_tpfa
from FVMO import compute_vector
from FEM import compute_matrix as compute_matrix_FEM
from FEM import compute_vector as compute_vector_FEM
from mesh import Mesh
from flux_error import compute_flux_error

from differentiation import divergence, gradient

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

def random_perturbation(h,aspect):
    return lambda p: np.array([random.uniform(0,h)*random.choice([-1,1]) + p[0] - 0.5*p[1],(1/aspect)*random.uniform(0,h)*random.choice([-1,1]) + p[1]])


x = sym.Symbol('x')
y = sym.Symbol('y')
K = np.array([[1,0],[0,1]])
u_fabric = sym.cos(y*math.pi)*sym.cosh(x*math.pi)
source = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
source = sym.lambdify([x,y],source)
u_lam = sym.lambdify([x,y],u_fabric)

nx = 8
ny = 8
T = lambda p: np.array([-0.5*p[1]+p[0],p[1]])
#T = lambda p: np.array([p[0],p[1]])

mesh = Mesh(nx,ny,T)
A = np.zeros((mesh.num_unknowns,mesh.num_unknowns))
A = compute_matrix_FEM(mesh,K,A)
f = compute_vector_FEM(mesh,source,u_lam)

# A = compute_matrix_l(mesh,K,A)
# f = compute_vector(mesh,source,u_lam)
# u = np.linalg.solve(A,f)
# u = np.reshape(u,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
# u = np.reshape(u,(mesh.num_unknowns,1),order='F')
# mesh.plot_vector(u)
# mesh.plot_funtion(u_lam)
# mesh = Mesh(6,6*5,T)
# mesh.plot()




start = 3
end = 5
aspect = 10
result_pressure = np.zeros((end-start,9))
result_flux = np.zeros((end-start,9))

for i in range(start,end):
    result_pressure[i-start,0] = i
    result_flux[i-start,0] = i
    mesh = Mesh(2**i,aspect*2**i,random_perturbation(1/(4*2**i),aspect))
    #mesh = Mesh(2**i,2**i,T)
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))

    #FVM-L
    num_unknowns = mesh.num_unknowns
    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    A,fx,fy = compute_matrix_l(mesh,K,A,flux_matrix)
    A = csr_matrix(A,dtype=float)
    f = compute_vector(mesh,source,u_lam)
    u = spsolve(A,f)
    fx = csr_matrix(fx,dtype=float)
    fy = csr_matrix(fy,dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    result_pressure[i-start,1] = math.log(l2err,2)
    result_pressure[i-start,2] = math.log(maxerr,2)
    result_flux[i-start,1] = math.log(l2err_flux,2)
    result_flux[i-start,2] = math.log(maxerr_flux,2)


    #FEM
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))
    A = compute_matrix_FEM(mesh,K,A)
    A = csr_matrix(A,dtype=float)
    f = compute_vector_FEM(mesh,source,u_lam)
    u = spsolve(A,f)
    u = np.reshape(u,(mesh.cell_centers.shape[0],mesh.cell_centers.shape[1]))
    u = np.ravel(u,order='F')

    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    print(l2err)
    result_pressure[i-start,7] = math.log(l2err,2)
    result_pressure[i-start,8] = math.log(maxerr,2)
    result_flux[i-start,7] = math.log(l2err_flux,2)
    result_flux[i-start,8] = math.log(maxerr_flux,2)

    #FVM-O
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))

    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    A,fx,fy = compute_matrix_o(mesh,K,A,flux_matrix)
    A = csr_matrix(A,dtype=float)
    f = compute_vector(mesh,source,u_lam)
    u = spsolve(A,f)
    mesh.plot_vector(u)
    fx = csr_matrix(fx,dtype=float)
    fy = csr_matrix(fy,dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    result_flux[i-start,3] = math.log(l2err_flux,2)
    result_flux[i-start,4] = math.log(maxerr_flux,2)
    result_pressure[i-start,3] = math.log(l2err,2)
    result_pressure[i-start,4] = math.log(maxerr,2)




    #TPFA
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))

    flux_matrix = {'x': lil_matrix((num_unknowns,num_unknowns)),'y':lil_matrix((num_unknowns,num_unknowns))}
    A,fx,fy = compute_matrix_tpfa(mesh,K,A,flux_matrix)
    A = csr_matrix(A,dtype=float)
    f = compute_vector(mesh,source,u_lam)
    u = spsolve(A,f)
    fx = csr_matrix(fx,dtype=float)
    fy = csr_matrix(fy,dtype=float)
    l2err_flux,maxerr_flux = compute_flux_error(fx.dot(u),fy.dot(u),u_fabric,mesh)
    l2err,maxerr = compute_error(mesh,u,u_lam)
    result_flux[i-start,5] = math.log(l2err_flux,2)
    result_flux[i-start,6] = math.log(maxerr_flux,2)
    result_pressure[i-start,5] = math.log(l2err,2)
    result_pressure[i-start,6] = math.log(maxerr,2)
    A = lil_matrix((mesh.num_unknowns,mesh.num_unknowns))




print(result_pressure)
print(result_flux)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('potential error')
p1, = ax1.plot(result_pressure[:,0],result_pressure[:,1],'--')
p3, = ax1.plot(result_pressure[:,0],result_pressure[:,3],'-.')
p3, = ax1.plot(result_pressure[:,0],result_pressure[:,5],'r:')
p3, = ax1.plot(result_pressure[:,0],result_pressure[:,7],'-')
ax1.set_title('$L_2$ error')
ax1.grid()
ax1.set(xlabel='$log_2 n$',ylabel='$log_2 e$')


p2, = ax2.plot(result_pressure[:,0],result_pressure[:,2],'--')
p2.set_label('L-method')
p4,=ax2.plot(result_pressure[:,0],result_pressure[:,4],'-.')
p4.set_label('O-method')
p4,=ax2.plot(result_pressure[:,0],result_pressure[:,6],'r:')
p4.set_label('TPFA-method')
p4,=ax2.plot(result_pressure[:,0],result_pressure[:,8],'-')
p4.set_label('FEM')
ax2.grid()
ax2.set_title('$max$ error')

ax2.set(xlabel='$log_2 n$',ylabel='$log_2 e$')
plt.legend(loc='lower center',bbox_to_anchor=(0.0, -0.3),ncol=4)
fig.subplots_adjust(bottom=0.20)
plt.savefig('figs2/pressure.pdf')

plt.show()

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
fig.suptitle('normal flow error')
p1, = ax1.plot(result_flux[:,0],result_flux[:,1],'--')
p3, = ax1.plot(result_flux[:,0],result_flux[:,3],'-.')
p3, = ax1.plot(result_flux[:,0],result_flux[:,5],'r:')
p3, = ax1.plot(result_flux[:,0],result_flux[:,7],'-')
ax1.set_title('$L_2$ error')
ax1.grid()
ax1.set(xlabel='$log_2 n$',ylabel='$log_2 e$')


p2, = ax2.plot(result_flux[:,0],result_flux[:,2],'--')
p2.set_label('L-method')
p4,=ax2.plot(result_flux[:,0],result_flux[:,4],'-.')
p4.set_label('O-method')
p4,=ax2.plot(result_flux[:,0],result_flux[:,6],'r:')
p4.set_label('TPFA-method')
p4,=ax2.plot(result_flux[:,0],result_flux[:,8],'-')
p4.set_label('FEM')
ax2.grid()
ax2.set_title('$max$ error')

ax2.set(xlabel='$log_2 n$',ylabel='$log_2 e$')
plt.legend(loc='lower center',bbox_to_anchor=(0.0, -0.3),ncol=4)
fig.subplots_adjust(bottom=0.20)
plt.savefig('figs2/flow.pdf')

plt.show()
