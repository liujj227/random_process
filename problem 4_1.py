from problem2 import DiscountedMDP 
import pandas as pd
from collections import defaultdict
import numpy as np
import os


if __name__ == '__main__':
    # 创建MDP实例
    p=0.1
    q=0.1
    w=0.1
    alpha=0.9
        
    print("-----------------------------begin-----------------------------")
    mdp = DiscountedMDP(p=p, q=q, w=w, alpha=alpha)

    # 运行值迭代算法
    V, policy = mdp.value_iteration()
    log_list = []
    print("-------------------p={} q={} w={} alpha={}-----------------".format(p,q,w,alpha))
    # 输出最优策略
    for (t,s) in V.keys():
        print(f"State (t={t}, s={s}): {'Visit' if policy[(t,s)]==1 else 'Not Visit'}")

        log_str = f"| State (t_k {t} s_k {s}) | {'Visit' if policy[(t,s)]==1 else 'Not Visit'} "
        log_list.append(log_str)

        print("------------------------------end------------------------------")
        