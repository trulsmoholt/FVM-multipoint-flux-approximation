import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from differentiation import gradient, divergence
import math

bottom_left = (0,0)
top_right = (1,1)

x = sym.Symbol('x')
y = sym.Symbol('y')

K=np.array([[1,0],[0,1]])
u_fabric = x*y*(1-x)*(1-y)
f = -divergence(gradient(u_fabric,[x,y]),[x,y],permability_tensor=K)
print(f)


num_nodes_x = 4
num_nodes_y = 4
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
print(a,b,c)
print(T)

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


#Matrix assembly
A = np.zeros((num_nodes,num_nodes))
f_vec = np.zeros(num_nodes)

for i in range(num_nodes_x):
    for j in range(num_nodes_y):
        vec_i = meshToVec(j,i)

        if (i==0) or (i==num_nodes_x-1) or (j==0) or (j==num_nodes_y-1):
            A[vec_i,vec_i] = 1
            f_vec[vec_i] = 0
            continue
        compute_flux_cell(j,i,A)  
        compute_source(j,i,f_vec,nodes)   
        x_d = nodes[j,i,0]
        y_d = nodes[j,i,1]   
        u_fabric_vec[vec_i] = u_fabric.subs([(x,x_d),(y,y_d)])
        u_fabric_nodes[j,i] = u_fabric.subs([(x,x_d),(y,y_d)])
u = np.linalg.solve(A,f_vec)
print(A)





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
print('max error at nodes',err_nodes_max)



fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot_surface(nodes[:,:,0],nodes[:,:,1],u_nodes,cmap='viridis', edgecolor='none')
ax.set_title('computed solution')
ax.set_zlim(0.00, 0.07)


ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(nodes[:,:,0],nodes[:,:,1],u_fabric_nodes,cmap='viridis', edgecolor='none')
ax.set_title('exact solution')
ax.set_zlim(0.00, 0.07)

plt.show()