import pandas as pd
from collections import defaultdict
import numpy as np
import os

class AverageCostMDP:
    def __init__(self, p, q, w, C=1, init_max_k=30):
        """
        初始化MDP参数
        p: 从状态0转移到状态1的概率
        q: 从状态1转移到状态0的概率
        w: 同步年龄的权重
        C: 访问成本
        init_max_k: 初始最大时间间隔
        """
        self.p = p
        self.q = q
        self.w = w
        self.C = C
        self.init_max_k = init_max_k
        self.pi_0, self.pi_1 = self.get_stationary(self.p,self.q)

        # 状态-价值字典: (t, s)-V, t为时间间隔，s为上次访问时的源状态，v为value
        self.states = defaultdict(float)
        self.policy = defaultdict(int)
        for t in range(1, self.init_max_k+1):
            for s in [0, 1]:
                self.states[(t,s)] = 0
                self.policy[t] = 1
        
        # 动作空间: 0表示不访问，1表示访问
        self.actions = [0, 1]
 
    def get_stationary(self,p,q):
        """计算稳态分布"""
        return q/(p+q), p/(p+q)
    
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
        if s == 0:
            for delta in range(t+1):
                if delta == 0:
                    prob = (1-self.p)**t
                else:
                    prob = (1-self.p)**(t-delta)*self.p
                E_AoS += delta * prob
        else:
            for delta in range(t+1):
                if delta == 0:
                    prob = self.q*(1-self.p)**(t-1)
                elif delta < t:
                    prob = self.q*(1-self.p)**(t-delta-1)*self.p
                else:
                    prob = 1-self.q
                E_AoS += delta * prob
        return E_AoS

    def relative_value_iteration(self, epsilon=1e-6,max_iter = 1000):
        """相对值迭代算法"""
        # 初始化值函数和平均开销
        lambda_k = 0  # 平均开销
        max_k = self.init_max_k
        iter = 0
        while True:
            state_old = self.states.copy()
            lambda_k_old = lambda_k
            delta = 0

            #  固定状态(1,0)和(1,1)，计算平均价值函数
            fix_state_value = []
            for state in [(1,0), (1,1)]:
                t, s = state
                P_n = self.get_transition_prob(t)
                # 计算访问动作的值
                V_visit = self.C
                for s_next in [0, 1]:
                    V_visit += P_n[s, s_next] * state_old[(1, s_next)]
                # 计算不访问动作的值
                V_not_visit = self.w * self.expected_AoS(t, s)
                V_not_visit += state_old[(t+1, s)]
                # 选择最小值作为新的值函数
                fix_state_value.append(min(V_visit, V_not_visit))
            lambda_k = self.pi_0*fix_state_value[0] + self.pi_1*fix_state_value[1]

            # 对每个状态进行更新
            for state in self.states.keys():
                t, s = state
                # 计算访问动作的值
                P_n = self.get_transition_prob(t)
                V_visit = self.C
                for s_next in [0, 1]:
                    V_visit += P_n[s, s_next] * state_old[(1, s_next)]
                
                # 计算不访问动作的值
                if t < max_k:
                    V_not_visit = self.w * self.expected_AoS(t+1, s) + state_old[(t+1, s)]
                else:
                    V_not_visit = 99999
                
                # 选择最小值作为新的值函数
                self.states[(t, s)] = min(V_visit, V_not_visit) - lambda_k
                self.policy[(t, s)] = 1 if V_visit < V_not_visit else 0
                delta = max(delta, abs(self.states[(t, s)] - state_old[(t, s)]))
            
            # 动态更新状态数，扩展或缩小阈值
            if (abs(lambda_k - lambda_k_old) < epsilon and delta < epsilon) or iter % max_iter == 0:
                if self.policy[(max_k-1,0)] == 1 and self.policy[(max_k-1,1)] == 1:
                    del self.states[(max_k,0)]
                    del self.states[(max_k,1)]
                    del self.policy[(max_k,0)]
                    del self.policy[(max_k,1)]
                elif self.policy[(max_k,0)] == 0 or self.policy[(max_k,1)] == 0:
                    self.states[(max_k+1,0)] = 0
                    self.states[(max_k+1,1)] = 0
                    self.policy[(max_k+1,0)] = 1
                    self.policy[(max_k+1,1)] = 1
                else:
                    break # 收敛
                max_k = int(len(self.states)/2)
            iter+=1

        return self.states, self.policy
    
if __name__ == '__main__':  
    # 创建MDP实例
    p=0.1
    q=0.1
    w=0.1

    print("-----------------------------begin-----------------------------")
    mdp = AverageCostMDP(p=p, q=q, w=w)

    # 运行值迭代算法
    V, policy = mdp.relative_value_iteration()
    log_list = []
    print("------------------------p={} q={} w={} -----------------".format(p,q,w))
    # 输出最优策略
    for (t,s) in V.keys():
        print(f"State (t={t}, s={s}): {'Visit' if policy[(t,s)]==1 else 'Not Visit'}")

        log_str = f"| State (t_k {t} s_k {s}) | {'Visit' if policy[(t,s)]==1 else 'Not Visit'} "
        log_list.append(log_str)
    
    print("------------------------------end------------------------------")