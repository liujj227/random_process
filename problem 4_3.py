from problem2 import DiscountedMDP 
from problem3 import AverageCostMDP
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
# 配置图标中文字体
rcParams['font.sans-serif'] = ['SimHei'] 
rcParams['axes.unicode_minus'] = False   

def run_variable(p,q,w):
    # 创建MDP实例
    alpha=0.9

    variable = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99,0.999] # 用于控制变量，观察各个参数对于结果的影响
    log_list = []
    # variable = [0.01]
    for v in variable:
        alpha = v

        mini_threshold_0 = 99999
        mini_threshold_1 = 99999

        print("-----------------------------begin-----------------------------")
        mdp = DiscountedMDP(p=p, q=q, w=w, alpha=alpha)

        # 运行值迭代算法
        V, policy = mdp.value_iteration()
        
        print("-------------------p={} q={} w={} alpha={}-----------------".format(p,q,w,alpha))
        # 输出最优策略
        for (t,s) in V.keys():
            print(f"State (t={t}, s={s}): {'Visit' if policy[(t,s)]==1 else 'Not Visit'}")
            if policy[(t,s)] == 1:
                if s == 0:
                    mini_threshold_0 = min(mini_threshold_0,t)
                elif s == 1:
                    mini_threshold_1 = min(mini_threshold_1,t)

        log_str0 = f"| alpha {alpha} | mini_threshold_0 {mini_threshold_0} | 'Visit' "
        log_str1 = f"| alpha {alpha} | mini_threshold_1 {mini_threshold_1} | 'Visit' "
        log_list.append(log_str0)
        log_list.append(log_str1)
        print("------------------------------end------------------------------")

    return log_list

def avgcost(p,q,w):
    print("-----------------------------begin-----------------------------")
    mdp = AverageCostMDP(p=p, q=q, w=w)

    # 运行值迭代算法
    V, policy = mdp.relative_value_iteration()
    log_list = []
    print("------------------------p={} q={} w={} -----------------".format(p,q,w))
    mini_threshold_0 = 99999
    mini_threshold_1 = 99999
    # 输出最优策略
    for (t,s) in V.keys():
        print(f"State (t={t}, s={s}): {'Visit' if policy[(t,s)]==1 else 'Not Visit'}")
        if policy[(t,s)] == 1:
            if s == 0:
                mini_threshold_0 = min(mini_threshold_0,t)
            elif s == 1:
                mini_threshold_1 = min(mini_threshold_1,t)

    log_str0 = f"| alpha {1} | mini_threshold_0 {mini_threshold_0} | 'Visit' "
    log_str1 = f"| alpha {1} | mini_threshold_1 {mini_threshold_1} | 'Visit' "
    log_list.append(log_str0)
    log_list.append(log_str1)
    return log_list

def compare_plot(p,q,w):
    folder_path = os.path.join(os.getcwd(), "avgcost_data")    
    file_path = os.path.join(folder_path, 'p {} q {} w {}.xlsx'.format(p,q,w))
    # 读取xlsx文件
    df = pd.read_excel(file_path)  
    # 创建空列表存储提取的数据
    var_values = []
    threshold_0_values = []
    threshold_1_values = []

    # 使用正则表达式提取数据
    for index, row in df.iterrows():
        # 提取var值
        var_match = re.search(r'alpha\s+([\d.]+)', str(row[0]))
        if var_match:
            var_value = float(var_match.group(1))
        
        # 提取threshold值
        threshold_match = re.search(r'mini_threshold_(\d+)\s+(\d+)', str(row[0]))
        if threshold_match:
            threshold_num = int(threshold_match.group(1))
            threshold_value = int(threshold_match.group(2))
        
            if threshold_num == 0:
                var_values.append(var_value)
                threshold_0_values.append(threshold_value)
            elif threshold_num == 1:
                threshold_1_values.append(threshold_value)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(var_values[:-1], threshold_0_values[:-1], 'b-o', label='Disc s_k = 0')
    print(var_values[:-1], threshold_0_values[:-1])
    plt.plot(var_values[-1], threshold_0_values[-1], 'ro', color='green', label='Avg_cost s_k = 0')
    plt.plot(var_values[:-1], threshold_1_values[:-1], 'r-o', label='Disc s_k = 1')
    plt.plot(var_values[-1], threshold_1_values[-1], 'ro', color='yellow', label='Avg_cost s_k = 1')

    # 设置图表属性
    plt.xlabel('变量参数值')
    plt.ylabel('阈值（第一个最优策略为Visit的t_k）')
    plt.title('超参数 p-{} q-{} w-{}'.format(p,q,w))
    plt.grid(True)
    plt.legend()

    # 保存图表
    script_dir = os.path.join(os.getcwd(), "avgcost_data")    
    save_path = os.path.join(script_dir, 'p {} q {} w {}.png'.format(p,q,w))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()
    print("picture have saved")

if __name__ == '__main__':
    p = 0.2
    q = 0.2
    w = 0.05
    
    # 运行折扣问题 alpha逐渐接近1
    log_list1 = run_variable(p,q,w)
    # 运行平均开销问题
    log_list2 = avgcost(p,q,w)
    log_list = log_list1 + log_list2

    # 保存结果
    folder_path = os.path.join(os.getcwd(), "avgcost_data")    
    file_path = os.path.join(folder_path, 'p {} q {} w {}.xlsx'.format(p,q,w))
    df = pd.DataFrame(log_list, columns=["Log"])
    df.to_excel(
        file_path,
        sheet_name="Sheet1",
        index=False,
    )

    # 画比较图
    compare_plot(p,q,w)