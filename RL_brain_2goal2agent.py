"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import math

class QLearningTable:
    # 初始化1
    def __init__(self, actions, learning_rate=0.175, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate     # 学习率
        self.gamma = reward_decay   # 奖赏衰减
        self.epsilon = e_greedy     # 贪婪度
        self.over = 0 #是否结束

        # 初始空的q_table
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        # 待初始
        # self.q_table = pd.read_excel('Q3test910.xlsx', index_col=0, header=None)
        # self.q_table.columns = self.actions
        # self.q_table = self.q_table.append(
        #    pd.Series(
        #        [0] * len(self.actions),
        #        index=self.q_table.columns,
        #        name=list("agent1", "agent2")
        #    )
        #)
        print(self.q_table)

    # 选行为
    def choose_action(self, ob1, ob2, episode):
        # 检验当前state是否在q_table中存在
        self.check_state_exist(ob1, ob2)

        # action selection选择action
        # 随机数小于0.9，则按最优的exploit

        self.epsilon = 1-0.1*math.exp(-0.006*episode)
        if self.epsilon > 0.995: #设置到一定阈值就停止
            self.over = 1

        if np.random.uniform() < self.epsilon:
            # choose best action选择Q值最高的action
            state_action = self.q_table.loc[str((ob1, ob2)), :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:   # explore
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # 学习参数更新：依据当前state，action，reward
    def learn(self, s1, s2, a, r, s_1, s_2):
        self.check_state_exist(s_1, s_2)
        q_predict = self.q_table.loc[str((s1, s2)), a]

        if s_1 != 'terminal' or s_2 != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[str((s_1, s_2)), :].max()
            # next state is not final terminal
        else:
            q_target = r
            # next state is final terminal
        # 更新对应的state-action值：学习率*（真实值-预测值），传回判断误差
        self.q_table.loc[str((s1, s2)), a] += self.lr * (q_target - q_predict)  # update
        # print(self.q_table)
        self.q_table.to_excel('D:\\AdaGuan大四上\\SRTP_MID\\RESULT\\8Qtable_small.xlsx', header=None)

    # 当前不知道到底多少种state，需要判断当前state是否经历过，若有，break，若无，则将该状态放入qtable
    def check_state_exist(self, state1, state2):
        if str((state1, state2)) not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=str((state1, state2))
                )
            )



