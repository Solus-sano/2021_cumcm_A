from random import  random
import pandas as pd
import numpy as np
from math import *
from scipy.linalg import null_space
import matplotlib.pyplot as plt

"""
基于第 2 问的反射面调节方案，计算调节后馈源舱的接收比，即馈源舱有效区域接收到
的反射信号与 300 米口径内反射面的反射信号之比，并与基准反射球面的接收比作比较
"""

def get_idx():
    """获取口径300m内需要激活节点的下标序号"""
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

def cal(x1,y1,z1,x2,y2,z2,x3,y3,z3):
    """计算一块三角反射板的接受比"""
    r1=sqrt(x1**2+y1**2+z1**2)
    r2=sqrt(x2**2+y2**2+z2**2)
    r3=sqrt(x3**2+y3**2+z3**2)
    A=np.array([[x1-x2,y1-y2,z1-z2],
                [x2-x3,y2-y3,z2-z3],
                [x3-x1,y3-y1,z3-z1]])

    B=np.array([r1**2-r2**2,r2**2-r3**2,r3**2-r1**2])/2;B=B.T
    v0=(np.linalg.pinv(A).dot(B)).reshape((-1,))#外心法线：v0+k*v
    v=null_space(A).reshape((-1,1))
    a=v[0]**2+v[1]**2+v[2]**2
    b=2*(v[0]*v0[0]+v[1]*v0[1]+v[2]*v0[2]-x1*v[0]+y1*v[1]-z1*v[2])
    c=x1**2+y1**2+z1**2-300**2+v0[0]**2+v0[1]**2+v0[2]**2-2*(x1*v0[0]+y1*v0[1]+z1*v0[2])
    k1=(-b+sqrt(b**2-4*a*c))/a/2
    k2=(-b-sqrt(b**2-4*a*c))/a/2
    #反射板球心
    if np.linalg.norm(k1*v+v0)>np.linalg.norm(k2*v+v0):
        x0,y0,z0=[v[i]*k2+v0[i] for i in range(3)]
    else:
        x0,y0,z0=[v[i]*k1+v0[i] for i in range(3)]

    z_10=(1-0.466)*(-300)#馈源舱平面高度
    def opt(x,y,z):
        """求在xyz处反射光线在馈源舱平面焦点的xy坐标"""
        tao=np.array([x0-x,y0-y,z0-z]).reshape((-1,1))
        apha=np.array([0,0,1]).reshape((-1,1))
        beta=(apha-2*np.dot(apha.T,tao)*tao/(np.linalg.norm(tao)**2)).reshape((-1,))

        bias=(z_10-z)/beta[2]
        return x+bias*beta[0],y+bias*beta[1]

    x_10,y_10=opt(x1,y1,z1)
    x_20,y_20=opt(x2,y2,z2)
    x_30,y_30=opt(x3,y3,z3)
    r_max=max(0.5,sqrt(x_10**2+y_10**2),sqrt(x_20**2+y_20**2),sqrt(x_30**2+y_30**2))#蒙特卡洛取值范围

    plt.plot([x_10,x_20,x_30,x_10],[y_10,y_20,y_30,y_10])

    sum,sum1=0,0

    #蒙特卡洛
    for i in range(1000):
        x,y=(random()-0.5)*2*(r_max+1),(random()-0.5)*2*(r_max+1)
        R1=sqrt((x_10-x)**2+(y_10-y)**2)
        R2=sqrt((x_20-x)**2+(y_20-y)**2)
        R3=sqrt((x_30-x)**2+(y_30-y)**2)
        if R1*R2*R3==0:continue#避免随机点与三角形顶点重合
        t1=acos(((x_10-x)*(x_20-x)+(y_10-y)*(y_20-y))/(R1*R2))
        t2=acos(((x_20-x)*(x_30-x)+(y_20-y)*(y_30-y))/(R2*R3))
        t3=acos(((x_30-x)*(x_10-x)+(y_30-y)*(y_10-y))/(R3*R1))
        theta=t1+t2+t3
        if abs(theta-pi*2)<1e-6:#随机点是否在像三角形内
            sum+=1
            if (x**2+y**2)<=0.25:sum1+=1#随机点是否在馈源舱内
    return sum,sum1

def get_ratio(activate=False):
    """计算总接受比"""
    num=0
    m=0
    if activate:
        print("近似抛物面:",'\n')
        x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3=x1,y1,z1,x1,y1,z1,x1,y1,z1
    else:
        print("基准球面")
        x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3=x0,y0,z0,x0,y0,z0,x0,y0,z0
    for i,j,k in triangle_lst:
        tmp_sum,tmp_sum1=cal(x_1[i],y_1[i],z_1[i],x_2[j],y_2[j],z_2[j],x_3[k],y_3[k],z_3[k])

        if not tmp_sum==0:
            m+=tmp_sum1/tmp_sum
        if num%100==0:
            print("第%d块反射板: success"%(num))
        num+=1
    return m/len(triangle_lst)

if __name__=='__main__':
    """读取数据"""
    data2 = pd.read_csv(r"附件2.csv",encoding='gbk').values
    data1 = pd.read_csv(r"附件1.csv",encoding='gbk').values
    data3 = pd.read_csv(r"附件3.csv",encoding='gbk').values
    data_ans=pd.read_csv('调节后坐标_伸缩量.csv',encoding='gbk').values
    x_data=data1[:,1].reshape(-1,)
    y_data=data1[:,2].reshape(-1,)
    z_data=data1[:,3].reshape(-1,)
    alpha,beta=[36.795,78.169];alpha,beta=[i*pi/180 for i in [alpha,beta]]
    A0=np.array([[cos(alpha),-sin(alpha),0],
                [sin(alpha),cos(alpha),0],
                [0,0,1]])
    B0=np.array([[sin(beta),0,cos(beta)],
                [0,1,0],
                [-cos(beta),0,sin(beta)]])
    Tran_M=np.dot(A0,B0)

    """节点名称与数组下标的转换"""
    idx_to_name={};name_to_idx={}
    for idx,name in enumerate(data1):
        idx_to_name[idx]=name[0]
        name_to_idx[name[0]]=idx

    
    x0,y0,z0=[data1[:,i].reshape((-1,)) for i in range(1,4)]
    active_node_idx_lst_tmp,x1,y1,z1=[data_ans[:,i].reshape((-1,)) for i in range(4)]
    active_node_idx_lst=[i for i in active_node_idx_lst_tmp]

    """获取需要激活的节点"""
    node_cnt,active_node_idx_lst=get_idx()
    """选取激活节点"""
    #节点初始坐标
    x0=x_data[active_node_idx_lst];y0=y_data[active_node_idx_lst];z0=z_data[active_node_idx_lst]
    x1=data_ans[:,1];y1=data_ans[:,2];z1=data_ans[:,3]
    

    #三角形反射板节点
    triangle_lst=[]
    for n1,n2,n3 in data3:
        if name_to_idx[n1] in active_node_idx_lst and name_to_idx[n2] in active_node_idx_lst and name_to_idx[n3] in active_node_idx_lst:
            triangle_lst.append([active_node_idx_lst.index(name_to_idx[n1]),active_node_idx_lst.index(name_to_idx[n2]),active_node_idx_lst.index(name_to_idx[n3])])

    """换系"""
    for i in range(x0.shape[0]):
        r1=np.array([x0[i],y0[i],z0[i]]).reshape((-1,1))
        r2=np.array([x1[i],y1[i],z1[i]]).reshape((-1,1))


        x0[i],y0[i],z0[i]=np.dot(np.linalg.inv(Tran_M),r1).reshape((-1,))
        x1[i],y1[i],z1[i]=np.dot(np.linalg.inv(Tran_M),r2).reshape((-1,))

    
    print("反光板个数： %d"%(len(triangle_lst)))

    plt.figure()

    ans=get_ratio(activate=True)#非激活为基准球面，激活时为近似抛物面
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    
    print(f"接受比: {ans*100} %")
    x_tmp=np.linspace(-0.5, 0.5, 100)
    plt.plot(x_tmp, np.sqrt(0.25 - (x_tmp)**2),'k')
    plt.plot(x_tmp, -np.sqrt(0.25 - (x_tmp)**2),'k')
    plt.title('image plane')

    plt.show()