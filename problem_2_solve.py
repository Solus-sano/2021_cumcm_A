import pandas as pd
import numpy as np
from scipy import optimize
from math import *
import time

"""
先只考虑主索节点调节后，相邻节点之间的
距离可能会发生微小变化，变化幅度不超过 0.07%,
解出变化后节点坐标
计算约25分钟
"""

def Loss(x,y,z):
    """输出坐标与理想抛物面之间误差"""
    z_hat=(x**2+y**2)/560.76 - 300.79
    return (z-z_hat)**2


def obj(x):
    """优化目标函数"""
    ans=0
    for i in range(node_cnt):
        ans+=Loss(x[i],x[i+node_cnt],x[i+2*node_cnt])
    return ans

def get_idx():
    """获取需要激活节点的下标序号"""
    x0,y0,z0,x1,y1,z1=[[] for i in range(6)]
    n_l=np.array([cos(beta)*cos(alpha),cos(beta)*sin(alpha),sin(beta)]).reshape((-1,1))
    idx_lst=[]
    for idx,r in enumerate(zip(x_data,y_data,z_data)):
        r=np.array(r).reshape((-1,1))
        d=np.linalg.norm(r)*sqrt(1-(np.dot(r.T,n_l)/np.linalg.norm(r))**2)
        if d >150 :
            x0.append(r[0]);y0.append(r[1]);z0.append(r[2])
        else:
            x1.append(r[0]);y1.append(r[1]);z1.append(r[2])
            idx_lst.append(idx)
    print("总节点数：%d"%(len(x1)))
    return len(x1),idx_lst

if __name__=='__main__':
    """读取数据"""
    data2 = pd.read_csv(r"附件2.csv",encoding='gbk').values
    data1 = pd.read_csv(r"附件1.csv",encoding='gbk').values
    data3 = pd.read_csv(r"附件3.csv",encoding='gbk').values
    x_data=data1[:,1].reshape(-1,)
    y_data=data1[:,2].reshape(-1,)
    z_data=data1[:,3].reshape(-1,)
    alpha,beta=[36.795,78.169];alpha,beta=[i*pi/180 for i in [alpha,beta]]
    A=np.array([[cos(alpha),-sin(alpha),0],
                [sin(alpha),cos(alpha),0],
                [0,0,1]])
    B=np.array([[sin(beta),0,cos(beta)],
                [0,1,0],
                [-cos(beta),0,sin(beta)]])
    Tran_M=np.dot(A,B)#换系矩阵

    """节点名称与数组下标的转换"""
    idx_to_name={};name_to_idx={}
    for idx,name in enumerate(data1):
        idx_to_name[idx]=name[0]
        name_to_idx[name[0]]=idx
    

    """获取需要激活的节点"""
    node_cnt,active_node_idx_lst=get_idx()

    """选取激活节点"""
    #节点初始坐标
    x0=x_data[active_node_idx_lst];y0=y_data[active_node_idx_lst];z0=z_data[active_node_idx_lst]

    """换系"""
    for i in range(x0.shape[0]):
        r1=np.array([x0[i],y0[i],z0[i]]).reshape((-1,1))


        x0[i],y0[i],z0[i]=np.dot(np.linalg.inv(Tran_M),r1).reshape((-1,))

    
    #主索节点对
    edge_lst=[]
    for n1,n2,n3 in data3:
        if name_to_idx[n1] in active_node_idx_lst and name_to_idx[n2] in active_node_idx_lst and name_to_idx[n3] in active_node_idx_lst:
            edge_lst.append([active_node_idx_lst.index(name_to_idx[n1]),active_node_idx_lst.index(name_to_idx[n2])])
            edge_lst.append([active_node_idx_lst.index(name_to_idx[n2]),active_node_idx_lst.index(name_to_idx[n3])])
            edge_lst.append([active_node_idx_lst.index(name_to_idx[n3]),active_node_idx_lst.index(name_to_idx[n1])])
    # edge_lst去重后应该还能提升效率

    cons=[]
    """节点间距离变化不超过0.07%"""
    for i , j in edge_lst:
        l_ij0=sqrt((x0[i]-x0[j])**2+(y0[i]-y0[j])**2+(z0[i]-z0[j])**2)
        cons.append(
            {'type':'ineq',
            'fun':lambda x:sqrt((x[i]-x[j])**2+(x[i+1*node_cnt]-x[j+1*node_cnt])**2+(x[i+2*node_cnt]-x[j+2*node_cnt])**2)/l_ij0-1+0.0007
            })
        cons.append(
            {'type':'ineq',
            'fun':lambda x,:-sqrt((x[i]-x[j])**2+(x[i+1*node_cnt]-x[j+1*node_cnt])**2+(x[i+2*node_cnt]-x[j+2*node_cnt])**2)/l_ij0+1+0.0007
            })


    """初值"""
    x_init=np.array(list(x0)+
                    list(y0)+
                    list(z0))
    print("solving the problem...")
    t0=time.time()

    res=optimize.minimize(obj,x_init,constraints=cons)
    print(res.success)
    print(res.fun)
    print("用时：",time.time()-t0)
    
    ans=(res.x).reshape((3,node_cnt)).T
    for i in range(ans.shape[0]):
        r1=ans[i,:3].reshape((-1,1))

        ans[i,:3]=np.dot(Tran_M,r1).reshape((-1,))

    if res.success:
        pd.DataFrame(np.hstack((np.array(active_node_idx_lst).reshape((-1,1)),ans))).to_csv(r"调节后坐标.csv",header=['index','x1','y1','z1'],index=False)
