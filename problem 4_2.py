from problem2 import DiscountedMDP 
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
# 配置图标中文字体
rcParams['font.sans-serif'] = ['SimHei'] 
rcParams['axes.unicode_minus'] = False   


def compare_plot(ind):
    var_list = ['p','q','w','alpha']
    data_excel_list = ['compare_p.xlsx','compare_q.xlsx','compare_w.xlsx','compare_alpha.xlsx']
    save_path_list = ['compare_p.png','compare_q.png','compare_w.png','compare_alpha.png']
    folder_path = os.path.join(os.getcwd(), "discount_data")
    # 读取xlsx文件
    df = pd.read_excel(os.path.join(folder_path,data_excel_list[ind]))  

    # 创建空列表存储提取的数据
    var_values = []
    threshold_0_values = []
    threshold_1_values = []

    # 使用正则表达式提取数据
    for index, row in df.iterrows():
        # 提取var值
        var_match = re.search(r'Var\s+([\d.]+)', str(row[0]))
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
    plt.plot(var_values, threshold_0_values, 'b-o', label='s_k = 0')
    plt.plot(var_values, threshold_1_values, 'r-o', label='s_k = 1')

    # 设置图表属性
    plt.xlabel('变量参数值')
    plt.ylabel('阈值（第一个最优策略为Visit的t_k）')
    plt.title('控制变量 {}'.format(var_list[ind]))
    plt.grid(True)
    plt.legend()

    # 保存图表
    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(script_dir, 'discount_data',save_path_list[ind])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()
    print("picture have saved")

def run_variable(ind):
# 创建MDP实例
    p=0.1
    q=0.1
    w=0.1
    alpha=0.9

    variable = [0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.97,0.98,0.99] # 用于控制变量，观察各个参数对于结果的影响
    folder_path = os.path.join(os.getcwd(), "discount_data")
    file_path_list = ['compare_p.xlsx','compare_q.xlsx','compare_w.xlsx','compare_alpha.xlsx']
    log_list = []
    # variable = [0.01]
    for v in variable:
        if ind == 0:
            p = v
        elif ind == 1:
            q = v
        elif ind == 2:
            w = v
        elif ind == 3:
            alpha = v
        else:
            print("error ind")
            break

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

        log_str0 = f"| Var {v} | mini_threshold_0 {mini_threshold_0} | 'Visit' "
        log_str1 = f"| Var {v} | mini_threshold_1 {mini_threshold_1} | 'Visit' "
        log_list.append(log_str0)
        log_list.append(log_str1)
        print("------------------------------end------------------------------")

    # 保存结果
    file_path = os.path.join(folder_path, file_path_list[ind])
    df = pd.DataFrame(log_list, columns=["Log"])
    df.to_excel(
        file_path,
        sheet_name="Sheet1",
        index=False,
    )
    

if __name__ == '__main__':
    """
    本问题需要生成不同参数对于最优策略阈值的影响
    最终会生成折线图，位于discount_data文件夹中，如果要复现请提前创建文件夹
    同时修改 ind变量 选取需要控制的变量，0-p  1-q  2-w  3-alpha
    """
    ind = 2 

    # 运行控制变量
    run_variable(ind)
    

    # 画比较图
    compare_plot(ind)