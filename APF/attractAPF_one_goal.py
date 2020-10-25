"""
This part of code initializes Q-table with artificial potential field.
A modification of APF, with only attraction field involved.
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
    def __init__(self, k_att, k_rep, d0, discount, goal, obstruction):
        self.k_att = k_att  # 引力系数
        self.k_rep = k_rep  # 斥力系数
        self.d0 = d0  # 斥力作用阈值
        self.gamma = discount  # 奖赏折扣系数，设为0.9
        self.goal_range = goal
        self.goal = ((goal[0] + goal[2]) / 2, (goal[1] + goal[3]) / 2)
        self.obs_range = obstruction
        self.obs = []
        for c in obstruction:
            self.obs.append(((c[0] + c[2]) / 2, (c[1] + c[3]) / 2))

        self.states = []
        for i in range(0, MAZE_W * UNIT, UNIT):
            for j in range(0, MAZE_H * UNIT, UNIT):
                if (i, j, i + UNIT, j + UNIT) == self.goal_range:
                    self.states.append("terminal")
                else:
                    # self.states.append((format(i, ".1f"), format(j, ".1f"), format(i + UNIT, ".1f"),format(j + UNIT, ".1f")))
                    self.states.append((i, j, i + UNIT, j + UNIT))

        self.Q_Table = pd.DataFrame(columns=actions, index=self.states)
        self.Q_Table.to_excel('Q0.xlsx', header=None)

    def attraction(self):
        att_data = pd.DataFrame(index=self.states, columns=['value'])
        for i in self.states:
            if i != 'terminal':
                x = (int(i[0]) + int(i[2])) / 2
                y = (int(i[1]) + int(i[3])) / 2
                U_att = 0.5 * self.k_att * math.hypot(x - self.goal[0], y - self.goal[1])
                att_data.loc[i] = U_att
            else:
                x = self.goal[0]
                y = self.goal[1]
                U_att = 0.5 * self.k_att * math.hypot(x - self.goal[0], y - self.goal[1])
                att_data.loc[i] = U_att
        return att_data

    """
        def repulsion(self):
            rep_data = pd.DataFrame(index=self.states, columns=['value'])
            for i in self.states:
                rep = 0
                if i == 'terminal':
                    x = self.goal[0]
                    y = self.goal[1]
                else:
                    x = (int(i[0]) + int(i[2])) / 2
                    y = (int(i[1]) + int(i[3])) / 2
                for j in self.obs:
                    dist_obs = math.hypot(x - j[0], y - j[1])
                    if 0 < dist_obs <= self.d0:
                        rep += 0.5 * self.k_rep * ((1 / dist_obs - 1 / self.d0) ** 2)
                rep_data.loc[i] = rep
            return rep_data
    """
    
    def standardization(self):
        standardize = pd.DataFrame(index=self.states, columns=['value'])
        U = pd.DataFrame(index=self.states, columns=['value'])
        attraction = self.attraction()
        # repulsion = self.repulsion()
        for s in self.states:
            U.loc[[s]] = attraction.loc[[s]]
           # U.loc[[s]] = attraction.loc[[s]] + repulsion.loc[[s]]
        Umax = U.max()
        Umin = U.min()
        for s in self.states:
            standardize.loc[[s]] = (Umax - U.loc[[s]]) / (Umax - Umin)
        return standardize

    def InitQTable(self):
        reward = pd.DataFrame(index=self.states, columns=['value'])
        standard = self.standardization()
        flag = 0
        # 奖赏与单一智能体保持一致：+500,-30,-1
        for s in self.states:
            if s == 'terminal':
                reward.loc[[s]] = 500
            elif s in self.obs_range:
                reward.loc[[s]] = -30
            else:
                reward.loc[[s]] = -1
        for s in self.states:
            if s == 'terminal':
                s = self.goal_range
                flag = 1
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
                    s = 'terminal'
                    flag = 0
                if next_state == self.goal_range:
                    next_state = 'terminal'
                if next_state is np.nan:
                    # 不可达的动作Q设置为-1000，其实无所谓，因为实际上不会查到这个Q值。
                    self.Q_Table.loc[[s], [j]] = -1000
                else:
                    Q = reward.loc[[next_state]] + self.gamma * standard.loc[[next_state]]
                    Q.columns = [j]
                    Q.index = [s]
                    self.Q_Table.loc[[s], [j]] = Q
        # return self.Q_Table.loc['terminal']
        return self.Q_Table

        #     elif len(self.actions) == 8:
        #         # actions=['up','down','left','right','upleft','upright','downleft','downright']
        #         for i in range(1, self.dim ** 2 + 1):
        #             move = {'up': i - self.dim if i > self.dim else np.nan,
        #                     'down': i + self.dim if i <= self.dim * (self.dim - 1) else np.nan,
        #                     'left': i - 1 if (i - 1) % self.dim != 0 and i != 1 else np.nan,
        #                     'right': i + 1 if i % self.dim != 0 else np.nan,
        #                     'upleft': i - self.dim - 1 if i > self.dim and (i - 1) % self.dim != 0 else np.nan,
        #                     'upright': i - self.dim + 1 if i > self.dim and i % self.dim != 0 else np.nan,
        #                     'downleft': i + self.dim - 1 if i <= self.dim * (self.dim - 1) and (
        #                             i - 1) % self.dim != 0 else np.nan,
        #                     'downright': i + self.dim + 1 if i <= self.dim * (
        #                     self.dim - 1) and i % self.dim != 0 else np.nan}
        #             for j in self.actions:
        #                 next_state = move[j]
        #                 if next_state is np.nan:
        #                     self.QTable.loc[i, j] = -np.Inf
        #                 else:
        #                     self.QTable.loc[i, j] = reward[next_state] + self.gamma * self.standardization()[next_state - 1]
        # return self.QTable


if __name__ == '__main__':
    term1 = 0
    term2 = 0
    # k_att = [0.01 * 20, 0.01 * 10, 0.01 * 8, 0.01 * 6, 0.01 * 4, 0.01 * 2, 0.01, 0.01 * 0.5, 0.01 * 0.25, 0.01 / 6,
    #         0.01 / 8, 0.01 / 10, 0.01 / 20]
    k_att = [0.01 * i for i in range(5, 10000, 50)]
    d0 = [10 * i for i in range(50)]
    #d0 = [200 * 20, 200 * 10, 200 * 8, 200 * 6, 200 * 4, 200 * 2, 200,  200 * 0.5, 200 * 0.25, 200 / 6, 200 / 8,
    #     200 / 10, 200 / 20]
    obs = []
    origin = np.array([0, 0])

    maze = pd.read_csv('maze1.csv')  # 不同的迷宫文件
    goal = (80+40, 0+40, 120+40, 40+40)
    # maze = pd.read_csv('maze2.csv')  # 不同的迷宫文件
    # goal = (280+40, 0+40, 320+40, 40+40)
    # maze = pd.read_csv('maze3.csv')  # 不同的迷宫文件
    # goal = (320+40, 0+40, 360+40, 40+40)

    obs_num = sum(maze.genre == 1)
    for i in range(obs_num):
        hell_center = origin + np.array([UNIT * (maze.iloc[i - 1, 0] - 1) + 20,
                                         UNIT * (maze.iloc[i - 1, 1] - 1) + 20])
        hell = np.array([hell_center[0] - 20, hell_center[1] - 20, hell_center[0] + 20, hell_center[1] + 20])
        hell = tuple(hell)
        obs.append(hell)

    # result1 = []
    # for term1 in range(len(d0)):
    #    apf = APF(0.01, 100000.0, d0[term1], 0.9, goal, obs)
    #    q = apf.InitQTable()
    #    sum = q.iloc[3] + q.iloc[2] + q.iloc[1]
    #    result1.append(sum)
    # print(result1)

    # result1 = pd.DataFrame(result1)
    # result1.to_excel('1qk_att.xlsx', header=None)
    apf = APF(0.01, 100000.0, 20, 0.9, goal, obs)
    Q_table = apf.InitQTable()
    # print(Q_table['(80,0,120,40)'])
    Q_table.to_excel('newQinit1.xlsx', header=None)

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
