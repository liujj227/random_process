import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os
# 配置图标中文字体
rcParams['font.sans-serif'] = ['SimHei'] 
rcParams['axes.unicode_minus'] = False   

def myplot(Delta_list,update,visit,save_path=None):

    k = np.arange(0, len(Delta_list))
    
    plt.plot(k, Delta_list, marker='o') # 绘制折线图
    # 添加 "visit" 和 "update" 的竖线
    for t in update:
        plt.axvline(x=t, color="green", linestyle="--", label="update" if t == update[0] else "")
    for t in visit:
        plt.axvline(x=t, color="red", linestyle=":", label="visit" if t == visit[0] else "")

    # 设置中文标题和标签
    plt.title('问题一  示例图-1') 
    plt.xlabel('时隙k')    
    plt.ylabel(r'同步年龄 $\Delta_k$')       
    plt.legend()
    plt.grid(True)
    if save_path:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(script_dir, save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_Delta(source_update,visit,k_max):
    Delta_list = []
    Delta_k = 0
    U_k = -1 # U_k 表示信息源在k时刻及以前最近一次发生状态变化的时刻
    V_k = -1 # V_k 表示无人机在k时刻及以前最近一次访问信息源的时刻
    for k in range(k_max):
        if k in source_update:
            U_k = k
        if k in visit:
            V_k = k

        if U_k <= V_k:
            Delta_k=0
        else:
            Delta_k+=1
        # print('当前时刻 {} | U_k {} | V_k {} | Delta_k {}'.format(k,U_k,V_k,Delta_k))
        Delta_list.append(Delta_k)
    return Delta_list

k_max = 30 # 本样例中总时长
source_update = [7,14,19] # 源更新时刻
visit = [5,10,26] # 无人机访问时隙

print('源更新发生时刻：',source_update)
print('无人机访问发生时刻：',visit)
Delta_list= get_Delta(source_update,visit,k_max)
save_path = 'Problem 1-1.png'
myplot(Delta_list,source_update,visit,save_path)