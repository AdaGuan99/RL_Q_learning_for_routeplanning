"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import math


class QLearningTable:
    # 初始化
    def __init__(self, actions, number, order, learning_rate=0.175, reward_decay=0.9, e_greedy=0.9, eta=0.4):
        self.number = number
        self.order = order
        self.actions = actions  # a list
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖赏衰减
        self.epsilon = e_greedy  # 贪婪度
        self.eta = eta
        self.over = 0  # 是否结束
        # 初始空的q_table
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # 下面是有初始化的读法
        '''if self.number == 1:
            self.q_table = pd.read_excel('map3_Q1final.xlsx', index_col=0, header=None)
        else:
            self.q_table = pd.read_excel('map3_Q2final.xlsx', index_col=0, header=None)'''
        self.q_table.columns = self.actions
        self.policy = pd.DataFrame.copy(self.q_table)
        self.policy.iloc[:, :] = 1 / len(self.actions) * np.ones(len(self.actions))

    # 选行为
    def choose_action(self, observation, episode):
        # 检验当前state是否在q_table中存在
        self.check_state_exist(observation)
        # action selection选择action
        # 随机数小于0.9，则按最优的exploit
        self.epsilon = 1 - 0.1 * math.exp(-0.006 * episode)
        if self.epsilon > 0.995:  # 设置到一定阈值就停止
            self.over = 1
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions,
                                      p=self.policy.loc[observation, :].ravel())  # 按照策略选择
        else:
            action = np.random.choice(self.actions)  # 随机探索
        return action

    # 学习参数更新：依据当前state，action，reward
    def learn(self, s, a, r, s_, meet_or_not):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal' and s != s_:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # 更新q表
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        # 存储Q表，不同命名
        if self.number == 1:
            self.q_table.to_excel('./Qtable/Q1_{}.xlsx'.format(self.order), header=None)
            self.policy.to_excel('./policy/policy1_{}.xlsx'.format(self.order), header=None)
        else:
            self.q_table.to_excel('./Qtable/Q2_{}.xlsx'.format(self.order), header=None)
            self.policy.to_excel('./policy/policy2_{}.xlsx'.format(self.order), header=None)

        # 更新策略函数
        state_actions = self.q_table.loc[s, :]
        max_action = state_actions[state_actions == np.max(state_actions)].index
        mu = np.zeros(len(self.actions))
        if a in max_action:
            mu[a] = 1
            self.eta = 0.4
            self.policy.loc[s, :] = (1 - self.eta) * self.policy.loc[s, :] + self.eta * mu
        else:
            if r == -30:
                self.eta = 0.8
            else:
                self.eta = 0.01
            mu = 1 / (len(self.actions) - 1) * np.ones(len(self.actions))
            mu[a] = 0
            self.policy.loc[s, :] = (1 - self.eta) * self.policy.loc[s, :] + self.eta * mu

    # 当前不知道到底多少种state，需要判断当前state是否经历过，若有，break，若无，则将该状态放入qtable
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        if state not in self.policy.index:
            self.policy = self.policy.append(
                pd.Series(
                    1 / len(self.actions) * np.ones(len(self.actions)),
                    index=self.q_table.columns,
                    name=state,
                )
            )






