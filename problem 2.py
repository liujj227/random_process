import pandas as pd
from collections import defaultdict
import numpy as np
import os

class DiscountedMDP:
    def __init__(self, p, q, w, alpha, C=1, init_max_k=50):
        """
        初始化MDP参数
        p: 从状态0转移到状态1的概率
        q: 从状态1转移到状态0的概率
        w: 同步年龄的权重
        alpha: 折扣因子
        C: 访问成本
        init_max_k: 初始最大时间间隔
        """
        self.p = p
        self.q = q
        self.w = w
        self.alpha = alpha
        self.C = C
        self.init_max_k = init_max_k

        # 状态空间: (t, s), t为时间间隔，s为上次访问时的源状态
        self.states = defaultdict(float)
        self.policy = defaultdict(int)
        for t in range(1, init_max_k+1):
            for s in [0, 1]:
                self.states[(t,s)] = 0
                self.policy[t] = 1

        # 动作空间: 0表示不访问，1表示访问
        self.actions = [0, 1]
        
    def get_transition_prob(self, n):
        """计算n步转移概率矩阵"""
        # 初始转移矩阵
        P = np.array([[1-self.p, self.p], 
                     [self.q, 1-self.q]])
        # 计算n步转移
        return np.linalg.matrix_power(P, n)
    
    def expected_AoS(self, t, s):
        """计算不访问时的期望同步年龄"""
        E_AoS = 0
        if s == 0: # 上一次访问时信息源状态为0
            for delta in range(t+1):
                # 分成两类
                if delta == 0:
                    prob = (1-self.p)**t
                else:
                    prob = (1-self.p)**(t-delta)*self.p
                E_AoS += delta * prob
        else: # 上一次访问时信息源状态为0
            for delta in range(t+1):
                # 分成三类
                if delta == 0:
                    prob = self.q*(1-self.p)**(t-1)
                elif delta < t:
                    prob = self.q*(1-self.p)**(t-delta-1)*self.p
                else:
                    prob = 1-self.q
                E_AoS += delta * prob
        return E_AoS

    def value_iteration(self, epsilon=1e-8):
        """值迭代算法"""
        max_k = self.init_max_k
        while True:
            state_old = self.states.copy() # 保存旧的状态-价值字典
            delta = 0 # 每一轮迭代最大差异值

            for state in self.states.keys(): # 遍历状态-价值字典
                t, s = state
                P_n = self.get_transition_prob(t) # 计算转移概率矩阵
                V_visit = self.C # 访问开销
                # 计算访问动作的值
                for s_next in [0, 1]:
                    V_visit += self.alpha * P_n[s, s_next] * state_old[(1, s_next)]
                
                # 计算不访问动作的值
                if t < max_k:
                    V_not_visit = self.w * self.expected_AoS(t+1, s) + self.alpha * state_old[(t+1, s)]
                else:
                    V_not_visit += 99999

                # 更新值函数和策略
                self.states[(t, s)] = min(V_visit, V_not_visit)
                self.policy[(t, s)] = 1 if V_visit < V_not_visit else 0

                delta = max(delta, abs(self.states[(t, s)] - state_old[(t, s)]))
            
            # 动态更新状态数，扩展或缩小阈值
            if delta < epsilon:
                if self.policy[(max_k-1,0)] == 1 and self.policy[(max_k-1,1)] == 1:
                    del self.states[(max_k,0)]
                    del self.states[(max_k,1)]
                    del self.policy[(max_k,0)]
                    del self.policy[(max_k,1)]
                    max_k = int(len(self.states)/2)
                elif self.policy[(max_k,0)] == 0 or self.policy[(max_k,1)] == 0:
                    # print(max_k,self.policy[(max_k,0)],self.policy[(max_k,1)])
                    self.states[(max_k+1,0)] = self.states[(max_k,0)]
                    self.states[(max_k+1,1)] = self.states[(max_k,1)]
                    self.policy[(max_k+1,0)] = 1
                    self.policy[(max_k+1,1)] = 1
                    max_k = int(len(self.states)/2)
                else:
                    break # 收敛

        return self.states, self.policy


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