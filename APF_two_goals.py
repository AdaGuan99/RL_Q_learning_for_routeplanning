"""
This part of code initializes Q-table with artificial potential field.
"""
from typing import List

import pandas as pd
import numpy as np
import math

UNIT = 40  # pixels每个单元的大小
MAZE_H = 12  # grid height垂直方向网格数
MAZE_W = 12  # grid width水平方向网格数
actions: List[str] = ['u', 'd', 'l', 'r']


class APF:
    def __init__(self, k_att, k_rep, d0, discount, goal1, goal2, obstruction):
        self.k_att = k_att  # 引力系数
        self.k_rep = k_rep  # 斥力系数
        self.d0 = d0  # 斥力作用阈值
        self.gamma = discount  # 奖赏折扣系数，设为0.9
        self.goal_range = []
        self.goal_range.append(goal1)
        self.goal_range.append(goal2)
        self.goal1 = ((goal1[0] + goal1[2]) / 2, (goal1[1] + goal1[3]) / 2)
        self.goal2 = ((goal2[0] + goal2[2]) / 2, (goal2[1] + goal2[3]) / 2)
        self.obs_range = obstruction
        self.obs = []
        for c in obstruction:
            self.obs.append(((c[0] + c[2]) / 2, (c[1] + c[3]) / 2))

        self.states = []
        for i in range(0, MAZE_W * UNIT, UNIT):
            for j in range(0, MAZE_H * UNIT, UNIT):
                if (i, j, i + UNIT, j + UNIT) == self.goal_range[0]:
                    self.states.append("terminal1")
                elif (i, j, i + UNIT, j + UNIT) == self.goal_range[1]:
                    self.states.append("terminal2")
                else:
                    # self.states.append((format(i, ".1f"), format(j, ".1f"), format(i + UNIT, ".1f"),format(j + UNIT, ".1f")))
                    self.states.append((i, j, i + UNIT, j + UNIT))

        self.Q_Table = pd.DataFrame(columns=actions, index=self.states)
        # self.Q_Table.to_excel('Q0.xlsx', header=None)

    def attraction(self):
        att_data = pd.DataFrame(index=self.states, columns=['value'])
        for i in self.states:
            if (i != "terminal1" ) & (i !="terminal2"):
                x = (int(i[0]) + int(i[2])) / 2
                y = (int(i[1]) + int(i[3])) / 2
                U_att1 = 0.5 * self.k_att * math.hypot(x - self.goal1[0], y - self.goal1[1])
                U_att2 = 0.5 * self.k_att * math.hypot(x - self.goal2[0], y - self.goal2[1])
                att_data.loc[i] = U_att1 + U_att2
                print(i)
                print(U_att1)
                print(U_att2)
            elif i == "terminal1":
                x = self.goal1[0]
                y = self.goal1[1]
                U_att1 = 0.5 * self.k_att * math.hypot(x - self.goal1[0], y - self.goal1[1])
                U_att2 = 0.5 * self.k_att * math.hypot(x - self.goal2[0], y - self.goal2[1])
                att_data.loc[i] = U_att1 + U_att2
            else:
                x = self.goal2[0]
                y = self.goal2[1]
                U_att1 = 0.5 * self.k_att * math.hypot(x - self.goal1[0], y - self.goal1[1])
                U_att2 = 0.5 * self.k_att * math.hypot(x - self.goal2[0], y - self.goal2[1])
                att_data.loc[i] = U_att1 + U_att2

        return att_data

    def repulsion(self):
        rep_data = pd.DataFrame(index=self.states, columns=['value'])
        for i in self.states:
            rep = 0
            if i == 'terminal1':
                x = self.goal1[0]
                y = self.goal1[1]
            elif i == 'terminal2':
                x = self.goal2[0]
                y = self.goal2[1]
            else:
                x = (int(i[0]) + int(i[2])) / 2
                y = (int(i[1]) + int(i[3])) / 2
            for j in self.obs:
                dist_obs = math.hypot(x - j[0], y - j[1])
                if 0 < dist_obs <= self.d0:
                    rep += 0.5 * self.k_rep * ((1 / dist_obs - 1 / self.d0) ** 2)
            rep_data.loc[i] = rep
        return rep_data

    def standardization(self):
        standardize = pd.DataFrame(index=self.states, columns=['value'])
        U = pd.DataFrame(index=self.states, columns=['value'])
        attraction = self.attraction()
        repulsion = self.repulsion()
        for s in self.states:
            U.loc[[s]] = attraction.loc[[s]] + repulsion.loc[[s]]
        Umax = U.max()
        Umin = U.min()
        for s in self.states:
            standardize.loc[[s]] = (Umax - U.loc[[s]]) / (Umax - Umin)
        return standardize

    def InitQTable(self):
        reward = pd.DataFrame(index=self.states, columns=['value'])
        standard = self.standardization()
        flag = 0
        for s in self.states:
            if (s == 'terminal1') | (s == 'terminal2'):
                reward.loc[[s]] = 500
            elif s in self.obs_range:
                reward.loc[[s]] = -30
            else:
                reward.loc[[s]] = -1
        for s in self.states:
            if s == 'terminal1':
                s = self.goal_range[0]
                print(s)
                flag = 1
            elif s == 'terminal2':
                s = self.goal_range[1]
                print(s)
                flag = 2
            move = {'u': (s[0], s[1] - UNIT, s[2], s[3] - UNIT)
            if s[1] > 0 else np.nan,
                    'd': (s[0], s[1] + UNIT, s[2], s[3] + UNIT)
                    if s[3] < MAZE_H * UNIT else np.nan,
                    'l': (s[0] - UNIT, s[1], s[2] - UNIT, s[3])
                    if s[0] > 0 else np.nan,
                    'r': (s[0] + UNIT, s[1], s[2] + UNIT, s[3])
                    if s[2] < MAZE_W * UNIT else np.nan}
            for j in actions:
                next_state = move[j]
                if flag == 1:
                    s = 'terminal1'
                    flag = 0
                if flag == 2:
                    s = 'terminal2'
                    flag = 0
                if next_state == self.goal_range[0]:
                    next_state = 'terminal1'
                if next_state == self.goal_range[1]:
                    next_state = 'terminal2'
                if next_state is np.nan:
                    # self.Q_Table.loc[[s], [j]] = -np.Inf
                    self.Q_Table.loc[[s], [j]] = -1000
                else:
                    Q = reward.loc[[next_state]] + self.gamma * standard.loc[[next_state]]
                    Q.columns = [j]
                    Q.index = [s]
                    self.Q_Table.loc[[s], [j]] = Q
        # return self.Q_Table.loc['terminal']
        return self.Q_Table

if __name__ == '__main__':
    term1 = 0
    term2 = 0
    # k_att = [0.01 * 20, 0.01 * 10, 0.01 * 8, 0.01 * 6, 0.01 * 4, 0.01 * 2, 0.01, 0.01 * 0.5, 0.01 * 0.25, 0.01 / 6,
    #         0.01 / 8, 0.01 / 10, 0.01 / 20]
    k_att = [0.0001 + 0.0001 * i for i in range(50)]

    k = [10 + 50000 * i for i in range(10)]

    # d0 = [5 + 5 * i for i in range(1000)]
    # d0 = [200 * 20, 200 * 10, 200 * 8, 200 * 6, 200 * 4, 200 * 2, 200,  200 * 0.5, 200 * 0.25, 200 / 6, 200 / 8,
    #     200 / 10, 200 / 20]
    obs = []
    origin = np.array([0, 0])

    maze = pd.read_csv('map3.csv')  # 不同的迷宫文件
    # maze = pd.read_csv('map2.csv')  # 不同的迷宫文件
    # maze = pd.read_csv('maze3.csv')  # 不同的迷宫文件
    goal1 = (80, 80, 120, 120)
    goal2 = (360, 80, 400, 120)

    obs_num = sum(maze.genre == 1)
    print(obs_num)
    for i in range(obs_num):
        hell_center = origin + np.array([UNIT * (maze.iloc[i - 1, 0] - 1) + 20,
                                         UNIT * (maze.iloc[i - 1, 1] - 1) + 20])
        hell = np.array([hell_center[0] - 20, hell_center[1] - 20, hell_center[0] + 20, hell_center[1] + 20])
        hell = tuple(hell)
        obs.append(hell)

    # result1 = []
    # for term1 in range(len(k)):
    #    apf = APF(0.01, k[term1], 20, 0.9, goal, obs)
    #    q = apf.InitQTable()
    #    sums = q.loc['terminal', 'd'] + q.loc['terminal', 'l'] + q.loc['terminal', 'r']
    #    print(q.loc['terminal', :])
    #    result1.append(sums)
    # print(result1)

    # result1 = pd.DataFrame(result1)
    # result1.to_excel('1katt.xlsx', header=None)
    apf = APF(0.01, 100000.0, 200, 0.9, goal1, goal2, obs)
    Q_table = apf.InitQTable()
    # print(Q_table['(80,0,120,40)'])
    Q_table.to_excel('Qinit3two.xlsx', header=None)
    print(Q_table.loc['terminal1', :])
    print(Q_table.loc['terminal2', :])
    '''

    result2 = []
    for term2 in range(13):
        apf = APF(0.01, 100000.0, d0[term2], 0.9, goal, obs)
        q = apf.InitQTable()
        sum = q.iloc[3] + q.iloc[2] + q.iloc[1]
        result2.append(sum)
    print(result2)
    result2 = pd.DataFrame(result2)
    result2.to_excel('1qd0.xlsx', header=None)




    apf = APF(0.01, 100000.0, 200.0, 0.9, (320,0,360,40), [(40,0,80,40),(40,40,80,80),(40,40,40,80),(280,40,320,80),
                                                           (280,80,320,120),(320,80,360,120),(360,80,400,120),(0,120,40,160),
                                                           (40,120,80,160),(120,160,160,200),(160,120,200,160),(200,120,240,160),
                                                           (40,160,80,200),(320,160,360,200),(360,160,400,200),(0,200,40,240),
                                                           (40,200,80,240),(120,200,160,240),(160,200,200,240),(320,200,360,240),
                                                           (80,280,320,120),(120,280,160,320),(240,280,280,320),(280,280,320,320),
                                                           (360,280,400,320),(360,320,400,360),(0,360,40,400),(40,360,80,400),
                                                           (80,360,120,400)])
    Q_table = apf.InitQTable()
    #print(Q_table['(80,0,120,40)'])
    Q_table.to_excel('Qinit3.xlsx', header=None)

    '''
