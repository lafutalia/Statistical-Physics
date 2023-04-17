import numpy as np
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
## sde_rk2_gen
def sde_rk2_gen(x0,t,f,noise):
    """
    :param f: 漂移项,请写成矢量
    :param x0:初始条件
    :param t:时间
    :param noise1:噪声项，矢量
    :return: 数值解
    """
    x=np.zeros([len(t),len(x0)])
    x[0]=x0
    for i in range(len(t)-1):
        delta_t=t[i+1]-t[i]
        u=noise(delta_t)
        F_x1=f(x[i],t)
        x_new=x[i]+F_x1*delta_t+u
        F_x2=f(x_new,t)
        F_x=(F_x2+F_x1)/2
        x[i+1]=x[i]+F_x*delta_t+u
    return x

## 求系综平均,by delayed and jobs


def gen_ensemble(gen,N=1000):
    """
    :param gen:生成一个所需要的值的生成器，可以是函数,gen没有输入值哦
    :param N:生成系综的数量
    :return: 一个列表，列表每一个是系综所得的值
    """
    ensemble=[]
    ensemble += Parallel(n_jobs=-1)(delayed(gen)(i) for i in range(N))
    return ensemble


def density_fun(x):
    """
    :param x:需要统计的量
    :return:获得的概率密度函数
    """
    return  gaussian_kde(x)

def Boltzman_distribution(N,f,x,T=1,k_B=1):
    """
    :N:归一化常数
    :T:温度
    :x:统计量
    :f:f(x)
    :return:获得的概率密度函数
    """
    return N*np.exp(-1/(k_B*T)*f(x))
def averge_ensemble(ensemble,type="none"):
    """
    :param ensemble: 获取系综内容,系综可能不包含完整的时间步，要等到稳定再取，该函数包含暂态
    :param type: 获取待平均的值
    :return: 返回系综平均值
    """
    N_ensemble=len(ensemble)  ## 获取系综数目，三维矩阵的列数 dim=1
    t_sample=ensemble[1].shape[0] ## 时间步长，三维矩阵的行数  dim=0
    d_sample=ensemble[1].shape[1] ## 待统计物理量的个数,三维矩阵的深度 dim=2
    X_martix=np.dstack([np.array([ensemble[i][:,j] for i in range(N_ensemble)]).T for j in range(d_sample)])
    X_square_martix=X_martix**2
    X_averge=np.mean(X_martix,axis=1)     ## 速度位置信息的均值
    X_square_average=np.mean(X_square_martix,axis=1)  ## 平方的均值
    averge_json={"x":X_averge,"x_square":X_square_average,"N_ensemble":N_ensemble}
    return averge_json


def average_on_ensemble(gen,N=1000):
    """
    :param gen:生成一个所需要的值的生成器，可以是函数,gen没有输入值哦
    :param N:生成系综的数量
    :return: 一个列表，列表每一个是系综所得的值
    """
    ensemble=[]
    ensemble += Parallel(n_jobs=-1)(delayed(gen)(i) for i in range(N))
    return ensemble


def time_cut(ensemble,N=1/5):
    """
    :param ensemble: 初始系综
    :param N: 截断距离
    :return: 截断后系综以及截断后时间索引
    """
    # index_cut=np.ceil(len(t)/5).astype(int)
    N_ensemble=len(ensemble)  ## 获取系综数目，三维矩阵的列数 dim=1
    t_sample=ensemble[1].shape[0] ## 时间步长，三维矩阵的行数  dim=0
    index_cut=np.ceil(t_sample*N).astype(int)
    d_sample=ensemble[1].shape[1] ## 待统计物理量的个数,三维矩阵的深度 dim=2
    ensemble_stable=[ensemble[i][index_cut:t_sample,:] for i in range(N_ensemble)]
    return ensemble_stable,[index_cut,t_sample]

def correction_fun_ensemble(ensemble):
    """
    :param ensemble: 传入系综
    :return:单个系综的关联函数，以及时间序列(并不是具体时间)
    """
    t_sample=ensemble.shape[0] ## 时间步长，三维矩阵的行数  dim=0
    d_sample=ensemble.shape[1] ## 待统计物理量的个数,三维矩阵的深度 dim=2
    X_martix=ensemble ## X的格式默认为:第一维度为时间步长，第三维度为时间步


    x_0=np.einsum('im,jn->ijmn',X_martix,X_martix)      ## 我们默认i<j即，上三角矩阵
    x_1=x_0.reshape(x_0.shape[0],x_0.shape[1],x_0.shape[2]*x_0.shape[3])   ## 时间组，自由重
    x_2=np.transpose(x_1,(2,0,1))
    x_3=x_2.reshape(x_0.shape[2]*x_0.shape[3],x_0.shape[0]*x_0.shape[1])   ##系综，自由重，时间重

    mask=np.triu(np.ones([x_0.shape[0],x_0.shape[1]]),k=0).reshape(-1)
    x_sample=x_3[:,mask>0]               ##系综，自由重，时间重+割


    t=np.arange(0,t_sample).reshape(-1,1)
    t_i_martix=t.repeat(len(t),axis=1)    ## i矩阵
    t_j_martix=t.T.repeat(len(t),axis=0)    ## j矩阵
    dt=(t_j_martix-t_i_martix).reshape(-1)
    dt_i_j=dt[mask>0]



    vector1=dt_i_j
    vector2 =x_sample

    unique_values, inverse_indices = np.unique(vector1, return_inverse=True)
    result = np.apply_along_axis(lambda x: np.bincount(inverse_indices, weights=x) / np.bincount(inverse_indices), axis=1, arr=vector2)

    return [result.T,unique_values]



def gen_ensemble_data_by_ensemble(gen,ensemble):
    """

    :param gen: 生成器，该生成器只能传入一个系综成员
    :param ensemble: 传入一个系综
    :return:
    """
    ensemble_data=[]
    ensemble_data += Parallel(n_jobs=-1)(delayed(gen)(ensemble[i]) for i in range(len(ensemble)))
    return ensemble_data


def averge_ensemble_A(gen,ensemble):
    """
    :param gen:生成器，给一个系综的值，会返回该系综的另一个统计量,gen的输入只能为X，输出值可以有很多列
    :param ensemble: 系综
    :return: 统计量A  的系综平均值
    """

    ensemble=gen_ensemble_data_by_ensemble(gen,ensemble)
    N_ensemble=len(ensemble)  ## 获取系综数目，三维矩阵的列数 dim=1
    t_sample=ensemble[1].shape[0] ## 时间步长，三维矩阵的行数  dim=0
    d_sample=ensemble[1].shape[1] ## 待统计物理量的个数,三维矩阵的深度 dim=2
    X_martix=np.dstack([np.array([ensemble[i][:,j] for i in range(N_ensemble)]).T for j in range(d_sample)])
    X_averge=np.mean(X_martix,axis=1)     ## 速度位置信息的均值
    averge_json={"x":X_averge,"N_ensemble":N_ensemble}
    return averge_json