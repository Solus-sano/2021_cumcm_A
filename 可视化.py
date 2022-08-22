from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import *

def init_show():
    """初始可视化"""
    plt.figure()
    ax=plt.axes(projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    ax.set_xticks(list(range(-300,305,100)));ax.set_yticks(list(range(-300,305,100)));ax.set_zticks(list(range(-350,351,100)))
    plt.legend()

def trans_show(alpha,beta):#角度制
    """仿射变换后激活节点可视化"""
    alpha,beta=[i*pi/180 for i in [alpha,beta]]
    x0,y0,z0,x1,y1,z1=[[] for i in range(6)]
    n_l=np.array([cos(beta)*cos(alpha),cos(beta)*sin(alpha),sin(beta)]).reshape((-1,1))

    for r in zip(x,y,z):
        r=np.array(r).reshape((-1,1))
        d=np.linalg.norm(r)*sqrt(1-(np.dot(r.T,n_l)/np.linalg.norm(r))**2)
        if d >150 :
            x0.append(r[0]);y0.append(r[1]);z0.append(r[2])
        else:
            x1.append(r[0]);y1.append(r[1]);z1.append(r[2])

    print(len(x1))
    plt.figure()
    ax=plt.axes(projection='3d')
    ax.scatter(x0,y0,z0,label='static')
    ax.scatter(x1,y1,z1,label='activate')
    ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
    ax.set_xticks(list(range(-300,305,100)));ax.set_yticks(list(range(-300,305,100)));ax.set_zticks(list(range(-350,351,100)))
    plt.legend()



if __name__=='__main__':
    data = pd.read_csv(r"附件1.csv",encoding='gbk').iloc[:,1:].values
    x=data[:,0].reshape(-1,)
    y=data[:,1].reshape(-1,)
    z=data[:,2].reshape(-1,)
    init_show()
    trans_show(36.795,78.169)
    plt.show()