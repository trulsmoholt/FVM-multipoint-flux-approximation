import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from differentiation import gradient, divergence
import math

num_nodes = 5    
h = 1/(num_nodes-1)
bottom_left = (0,0)
top_right = (1,1)

x = sym.Symbol('x')
y = sym.Symbol('y')

K=np.array([[1,2],[2,1]])
u_fabric = (-x*y*(1-x)*(1-y))
f = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)


num_nodes_x = num_nodes
num_nodes_y = num_nodes
num_nodes = num_nodes_x*num_nodes_y
u_h = np.zeros(num_nodes)
nodes_x, nodes_y = np.meshgrid(np.linspace(bottom_left[0],top_right[0],num=num_nodes_x),np.linspace(bottom_left[1],top_right[1],num=num_nodes_y))
nodes = np.stack([nodes_x,nodes_y],axis=2)
u_fabric_vec = np.zeros(num_nodes)
u_fabric_nodes = np.zeros((num_nodes_y,num_nodes_x))
f_vec = np.zeros(num_nodes)

def meshToVec(j,i)->int:
    return i*num_nodes_y + j
def vecToMesh(h)->(int,int):
    return (h % num_nodes_y, math.floor(h/num_nodes_y))

a1=np.array([[0],[1]])
a2=np.array([[1],[0]])
V = 1


a = float(1/V*a1.T@K@a1)
b = float(1/V*a2.T@K@a2)
c = float(1/V*a1.T@K@a2)

A = np.array([[2*a  ,-c     ,0      ,c      ],
            [-c   ,2*b    , c     , 0     ],
            [0    ,c      , 2*a   , -c    ],
            [c    ,0      ,-c     ,2*b    ]])

B = np.array([[a+c  ,a-c    ,0      ,0      ],
            [0    ,b-c    ,b+c    ,0      ],
            [0    ,0      ,a+c    ,a-c    ],
            [b+c  ,0      ,0      ,b-c    ]])

C = np.array([[-a   ,0  ,0  ,-c ],
            [c    ,-b ,0  ,0  ],
            [0    ,c  ,a  ,0  ],
            [0    ,0  ,-c ,b  ]])

D = np.array([[-(a+c)   ,0      ,0      ,0      ],
            [0        ,-(b-c) ,0      ,0      ],
            [0        ,0      ,a+c    ,0      ],
            [0        ,0      ,0      ,b-c    ]])

T = C@np.linalg.inv(A)@B-D

def compute_flux_interface_east(j,i,A):
    current_cell = meshToVec(j,i)
    A[current_cell,meshToVec(j,i)] += T[0,0] + T[2,3]
    A[current_cell,meshToVec(j,i+1)] += T[0,1] + T[2,2]
    A[current_cell,meshToVec(j+1,i+1)] += T[0,2]
    A[current_cell,meshToVec(j+1,i)] += T[0,3]
    A[current_cell,meshToVec(j-1,i)] += T[2,0]
    A[current_cell,meshToVec(j-1,i+1)] += T[2,1]
    return
def compute_flux_interface_west(j,i,A):
    i = i - 1
    current_cell = meshToVec(j,i+1)
    A[current_cell,meshToVec(j,i)] += -T[0,0] + -T[2,3]
    A[current_cell,meshToVec(j,i+1)] += -T[0,1] - T[2,2]
    A[current_cell,meshToVec(j+1,i+1)] += -T[0,2]
    A[current_cell,meshToVec(j+1,i)] += -T[0,3]
    A[current_cell,meshToVec(j-1,i)] += -T[2,0]
    A[current_cell,meshToVec(j-1,i+1)] += -T[2,1]
    return

def compute_flux_interface_north(j,i,A):
    current_cell = meshToVec(j,i)
    A[current_cell,meshToVec(j,i)] += T[1,1] + T[3,0]
    A[current_cell,meshToVec(j+1,i)] += T[1,2] + T[3,3]
    A[current_cell,meshToVec(j+1,i-1)] += T[1,3]
    A[current_cell,meshToVec(j,i-1)] += T[1,0]
    A[current_cell,meshToVec(j,i+1)] += T[3,1]
    A[current_cell,meshToVec(j+1,i+1)] += T[3,2]
    return
def compute_flux_interface_south(j,i,A):
    j = j -1
    current_cell = meshToVec(j+1,i)
    A[current_cell,meshToVec(j,i)] += -T[1,1] - T[3,0]
    A[current_cell,meshToVec(j+1,i)] += -T[1,2] - T[3,3]
    A[current_cell,meshToVec(j+1,i-1)] += -T[1,3]
    A[current_cell,meshToVec(j,i-1)] += -T[1,0]
    A[current_cell,meshToVec(j,i+1)] += -T[3,1]
    A[current_cell,meshToVec(j+1,i+1)] += -T[3,2]
    return
def compute_flux_cell(j,i,A):
    compute_flux_interface_east(j,i,A)
    compute_flux_interface_north(j,i,A)
    compute_flux_interface_west(j,i,A)
    compute_flux_interface_south(j,i,A)
    return
def compute_source(j,i,f_vect,nodes):
    north_face = (nodes[j,i+1,0]-nodes[j,i-1,0])/2
    east_face = (nodes[j+1,i,1]-nodes[j-1,i,1])/2
    V = north_face*east_face
    x_d = nodes[j,i,0]
    y_d = nodes[j,i,1]
    f_vec[meshToVec(j,i)] = V*f.subs([(x,x_d),(y,y_d)])

def integrate(f,j,i):
    north_face = (nodes[j,i+1,0]-nodes[j,i-1,0])/2
    east_face = (nodes[j+1,i,1]-nodes[j-1,i,1])/2
    V = north_face*east_face
    x_d = nodes[j,i,0]
    y_d = nodes[j,i,1]

    #return V*(1/36)*(16*f(x_d,y_d)+4*f(x_d,y_d+north_face/2)+4*f(x_d,y_d-north_face/2)+4*f(x_d+east_face/2,y_d)+4*f(x_d-east_face/2,y_d)+f(x_d + east_face/2,y_d+north_face/2)+f(x_d + east_face/2,y_d-north_face/2)+f(x_d-east_face/2,y_d-east_face/2)+f(x_d - east_face/2,y_d + north_face/2))
    return V*(f(x_d,y_d))
    


#Matrix assembly
A = np.zeros((num_nodes,num_nodes))
f_vec = np.zeros(num_nodes)
f_lam = sym.lambdify([x,y],f)
u_lam = sym.lambdify([x,y],u_fabric)

for i in range(num_nodes_x):
    for j in range(num_nodes_y):
        vec_i = meshToVec(j,i)

        if (i==0) or (i==num_nodes_x-1) or (j==0) or (j==num_nodes_y-1):
            A[vec_i,vec_i] = 1
            f_vec[vec_i] = 0
            continue
        compute_flux_cell(j,i,A)  
        f_vec[meshToVec(j,i)] = integrate(f_lam,j,i)
        x_d = nodes[j,i,0]
        y_d = nodes[j,i,1]   
        u_fabric_vec[vec_i] = u_lam(x_d,y_d)
        u_fabric_nodes[j,i] = u_lam(x_d,y_d)
u = np.linalg.solve(A,f_vec)


print(A[7,:])


u_vec = np.linalg.solve(A,f_vec)
u_nodes = u_fabric_nodes.copy()
f_nodes = u_fabric_nodes.copy()
err_nodes = np.zeros(num_nodes)
err_nodes_max = 0
for i in range(num_nodes):
    u_nodes[vecToMesh(i)] = u_vec[i]
    f_nodes[vecToMesh(i)] = f_vec[i]
    err_nodes[i] = abs(u_fabric_vec[i]-u_vec[i])
err_nodes_max = np.max(err_nodes)
print('max error at nodes ',err_nodes_max)


L2_error = 0
V = (nodes[0,1,0]-nodes[0,0,0])**2
for i in range(num_nodes_x):
    for j in range(num_nodes_y):
        if (j==num_nodes_y-1) or (i==num_nodes_x-1):
            continue
        u_1 = u_vec[meshToVec(i,j)]
        u_2 = u_vec[meshToVec(i+1,j)]
        u_3 = u_vec[meshToVec(i+1,j+1)]
        u_4 = u_vec[meshToVec(i,j+1)]
        u = (u_1 + u_2 + u_3 + u_4)/4

        difference = (u_lam(nodes[j,i,0]+h/2,nodes[j,i,1]+h/2) - u)**2
        # print(difference)
        # print(nodes[j,i,0]+h/2)
        # print(nodes[j,i,1]+h/2)
        
        
        #print(u_lam(nodes[j,i,0]+(1/(2*num_nodes)),nodes[j,i,1]+(1/(2*num_nodes))))
        L2_error = L2_error + V*difference



print('L2 error ',math.sqrt(L2_error))
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot_surface(nodes[:,:,0],nodes[:,:,1],u_nodes,cmap='viridis', edgecolor='none')
ax.set_title('computed solution')
ax.set_zlim(0.00, -0.07)


ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(nodes[:,:,0],nodes[:,:,1],u_fabric_nodes,cmap='viridis', edgecolor='none')
ax.set_title('exact solution')
ax.set_zlim(0.00, -0.07)

plt.show()
