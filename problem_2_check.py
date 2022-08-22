from math import *
import pandas as pd
import numpy as np

"""
用problem_2_solve解出的坐标,利用下拉索不变
及促动器径向伸缩算出促动器上端点变化后坐标
验证可知伸缩量都不超过0.6
"""

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
    data2 = pd.read_csv(r"附件2.csv",encoding='gbk').values
    data1 = pd.read_csv(r"附件1.csv",encoding='gbk').values
    data3 = pd.read_csv(r"附件3.csv",encoding='gbk').values
    x_data=data1[:,1].reshape(-1,)
    y_data=data1[:,2].reshape(-1,)
    z_data=data1[:,3].reshape(-1,)
    alpha,beta=[36.795,78.169];alpha,beta=[i*pi/180 for i in [alpha,beta]]
    ans_data=pd.read_csv('调节后坐标.csv',encoding='gbk').values
    

    """获取需要激活的节点"""
    node_cnt,active_node_idx_lst=get_idx()
    """节点名称与数组下标的转换"""
    idx_to_name={};name_to_idx={}
    for idx,name in enumerate(data1):
        idx_to_name[idx]=name[0]
        name_to_idx[name[0]]=idx

    active_node_name_lst=[idx_to_name[idx] for idx in active_node_idx_lst]

    """选取激活节点"""
    #节点初始坐标
    x0=x_data[active_node_idx_lst];y0=y_data[active_node_idx_lst];z0=z_data[active_node_idx_lst]
    #节点变化后坐标
    x1,y1,z1=[ans_data[:,i].reshape((-1,)) for i in range(1,4)]
    #促动器下端点坐标
    u_down,v_down,w_down=[data2[active_node_idx_lst,i].reshape((-1,)) for i in range(1,4)]
    #促动器上端点初始坐标
    u0_up,v0_up,w0_up=[data2[active_node_idx_lst,i].reshape((-1,)) for i in range(4,7)]
    # 下拉索长度
    d0=[sqrt((x0[i]-u0_up[i])**2+(y0[i]-v0_up[i])**2+(z0[i]-w0_up[i])**2) for i in range(node_cnt)]
    #初始促动器顶端离原点距离
    D=[sqrt(u0_up[i]**2+v0_up[i]**2+w0_up[i]**2) for i in range(node_cnt)]

    delta=[]
    for i in range(len(x1)):
        a=u0_up[i]**2+v0_up[i]**2+w0_up[i]**2
        b=-2*(u0_up[i]*x1[i]+v0_up[i]*y1[i]+w0_up[i]*z1[i])
        c=x1[i]**2+y1[i]**2+z1[i]**2-d0[i]**2
        k=(-b+sqrt(b**2-4*a*c))/(2*a)
        delta.append(D[i]*(1-k))
    delta=np.array(delta)
    op=np.c_[np.array(active_node_name_lst).reshape((-1,1)),
            np.array(x1).reshape((-1,1)),
            np.array(y1).reshape((-1,1)),
            np.array(z1).reshape((-1,1)),
            delta.reshape((-1,1))]
    pd.DataFrame(op).to_csv("调节后坐标_伸缩量.csv",header=['节点编号','x坐标','y坐标','z坐标','伸缩量'],index=False,encoding='gbk')
    print("伸缩量最大值：%f"%(max(delta.max(),abs(delta.min()))))