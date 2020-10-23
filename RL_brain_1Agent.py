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
    def __init__(self, actions, learning_rate=0.175, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate     # 学习率
        self.gamma = reward_decay   # 奖赏衰减
        self.epsilon = e_greedy     # 贪婪度
        self.over = 0 #是否结束
        # 初始空的q_table
        self.q_table = pd.read_excel('Qempty.xlsx', index_col=0, header=None)
        self.q_table.columns = self.actions
        print(self.q_table)
        # self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 选行为
    def choose_action(self, observation, episode):
        # 检验当前state是否在q_table中存在
        self.check_state_exist(observation)

        # action selection选择action
        # 随机数小于0.9，则按最优的exploit
        #np.random.seed(episode)
        self.epsilon = 1-0.1*math.exp(-0.006*episode)
        #self.epsilon = 0.9 + 0.00020 * episode
        if self.epsilon > 0.995: #设置到一定阈值就停止
            self.over = 1
        #print(np.random.uniform())
        if np.random.uniform() < self.epsilon:
            # choose best action选择Q值最高的action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:   # explore
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # 学习参数更新：依据当前state，action，reward
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # 更新对应的state-action值：学习率*（真实值-预测值），传回判断误差
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #print(self.q_table)
        self.q_table.to_excel('Q2.xlsx', header=None)

    # 当前不知道到底多少种state，需要判断当前state是否经历过，若有，break，若无，则将该状态放入qtable
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


