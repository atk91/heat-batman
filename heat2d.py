#!/usr/bin/python

import numpy as np
import scipy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cgs
from scipy.sparse import csr_matrix

fig = plt.figure()

t_min = 0.0
t_max = 10.0
x_min = -10.0
x_max = 10.0
y_min = -10.0
y_max = 10.0

a = 1.0
c = 5.0
m = 400
n = 100

def ind(i, j):
    return i *  m + j

def abs(x):
    return np.fabs(x)

def sqrt(x):
    return np.sqrt(x)

def u_init(x, y):
    if x**2/(49*a**2)+y**2/(9*a**2)-1<=0 and (abs(x/a)>=4 and -(3*sqrt(33))/7<=y/a<=0 or abs(x/a)>=3 and y>=0 or \
    -3<=y/a<=0 and -4<=x/a<=4 and \
    (abs(x/a))/2+sqrt(1-(abs(abs(x/a)-2)-1)**2)-((3*sqrt(33)-7)*x**2)/(112*a**2)-y/a-3<=0 or y>=0 and \
    3.0/4.0<=abs(x/a)<=1.0 and -8*abs(x/a)-y/a+9>=0 or 1.0/2.0<=abs(x/a)<=3.0/4.0 and \
    3*abs(x/a)-y/a+3.0/4.0>=0 and y>=0 or abs(x/a)<=1.0/2.0 and y>=0 and 9.0/4.0-y/a>=0 or abs(x/a)>=1 \
    and y>=0 and -(abs(x/a))/2-3.0/7.0 * sqrt(10) * sqrt(4-(abs(x/a)-1)**2)-y/a+(6*sqrt(10))/7+3.0/2.0>=0):
        return 1.0
    else:
        return 0.0
        
def x_0(t):
    return 0.0

def y_0(t):
    return 0.0

x = np.linspace(x_min, x_max, m)
y = np.linspace(y_min, y_max, m)
t = np.linspace(t_min, t_max, n)

dx = (x_max - x_min)/(m - 1)
dy = (y_max - y_min)/(m - 1)
dt = (t_max - t_min)/(n - 1)

matr_size = m**2

L = csr_matrix((matr_size, matr_size))
right = np.zeros(matr_size)

u_prev = np.zeros(m * m)
u = np.zeros(m * m)

for i in range(m):
    for j in range(m):
        u_prev[(m - 1 - j) * m + i] = u_init(x_min + i * dx, y_min + j * dy)#u_init(x_min + i * dx, y_min + j * dy)
        u[(m - 1 - j) * m + i] = u_init(x_min + i * dx, y_min + j * dy)

for k in range(n):
    data = []
    row = []
    col = []
    L = csr_matrix((matr_size, matr_size))
    to_plot = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            to_plot[i][j] = u_prev[i * m + j]

    ax = fig.add_subplot(111)
    ax.set_title("Heat equation solution, t = %.2f" % (k * dt))
    plt.imshow(to_plot, vmax=1.0)
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.get_yaxis().set_ticklabels([])
    cax.get_xaxis().set_ticklabels([])
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')

    plt.savefig("images/%d.png" % k)
    plt.clf()
    for i in range(m):
        for j in range(m):
            # print "%d, %d", i, j
            str_num = i * m + j
            if i == 0 or i == m - 1:
                data.append(1.0)
                row.append(str_num)
                col.append(ind(i, j))
                # L[str_num][ind(i, j)] = 1.0
                right[str_num] = x_0(j * dx)
            elif j == 0 or j == m - 1:
                data.append(1.0)
                row.append(str_num)
                col.append(ind(i, j))
                #L[str_num][ind(i, j)] = 1.0
                right[str_num] = y_0(i * dy)
            else:
                data.append(c / (dx**2))
                row.append(str_num)
                col.append(ind(i - 1, j))
                
                data.append(c / (dx**2))
                row.append(str_num)
                col.append(ind(i, j - 1))
                
                data.append(- 4.0*c/(dx**2) - 1.0/dt)
                row.append(str_num)
                col.append(ind(i, j))
                
                data.append(c / (dx**2))
                row.append(str_num)
                col.append(ind(i + 1, j))
                
                data.append(c / (dx**2))
                row.append(str_num)
                col.append(ind(i, j + 1))
                
                right[str_num] = - u_prev[ind(i, j)] / dt
    L = csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(matr_size, matr_size))
    u, info = cgs(L, right, x0 = u_prev, tol=1e-10)
    #print "residual: %le" % la.norm(np.dot(L, u) - right)
    #print "norm u + u_prev = %le" % la.norm(u - u_prev)
    u_prev = u
    
