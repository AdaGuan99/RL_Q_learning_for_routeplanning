"""
两智能体，两目标【分别到达两个目标点才成功，撞到障碍物退格继续】
暂时没有搭配APF_two_goals实现初始化
配套地图：mapX系列

reward
思路一
到达目标点   +500
到达一般障碍物或运动中的另一目标点 -100
到达已被另一目标点到达的目标点 -20 [目标点处随后小惩罚]
多走一步 -10

思路二
到达目标点   +500
到达一般障碍物或运动中的另一目标点 -100
到达已被另一目标点到达的目标点 -100 [目标点处随后大惩罚]
多走一步 -10

注意：
取消了seed
"""

import numpy as np
import time
import sys
import tkinter as tk
import random as rd
import pandas as pd

UNIT = 40  # pixels每个单元的大小
# small_map
MAZE_W = 8  # grid width水平方向网格数
MAZE_H = 8  # grid height垂直方向网格数
# map
# MAZE_W = 12  # grid width水平方向网格数
# MAZE_H = 12  # grid height垂直方向网格数

# reward
# 思路一
#r_goal = +500   # 到达目标点   +500
#r_col = -100    # 到达一般障碍物或运动中的另一目标点 -100
#r_repeat_goal = -20     # 到达已被另一目标点到达的目标点 -20 [目标点处随后小惩罚]
#r_move = -10    # 多走一步 -10

# 思路二
r_goal = +500   # 到达目标点   +500
r_col = -100    # 到达一般障碍物或运动中的另一目标点 -100
r_repeat_goal = -100     # 到达已被另一目标点到达的目标点 -100 [目标点处随后大惩罚]
r_move = -10    # 多走一步 -10

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['uu', 'ud', 'ur', 'ul', 'du', 'dd', 'dr', 'dl', 'ru', 'rd', 'rr', 'rl', 'lu', 'ld', 'lr', 'll']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.flag = 0 # 统计成功到达的agent个数
        self.firstsuccessagent = -1
        self.firstsuccessgoal = -1

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 固定原点，其实默认（0，0），后面也没用
        origin = np.array([0, 0])

        # 创建障碍物
        self.hells = []
        self.maze = pd.read_csv('../small_map1.csv')  # 不同的迷宫文件
        # self.maze = pd.read_csv('map3.csv')  # 不同的迷宫文件
        self.hell_number = sum(self.maze.genre == 1)
        for i in range(1, self.hell_number+1):
            hell_center = origin+np.array([UNIT * (self.maze.iloc[i-1, 0]-1) + 20,
                                           UNIT * (self.maze.iloc[i-1, 1]-1) + 20])
            self.hell = self.canvas.create_rectangle(
                hell_center[0] - 20, hell_center[1] - 20, hell_center[0] + 20,
                hell_center[1] + 20, fill='grey')
            self.hells.append(self.canvas.coords(self.hell))

        # 左侧出发点
        origins1 = np.array([20 + UNIT * (self.maze.iloc[self.hell_number, 0] - 1),
                            20 + UNIT * (self.maze.iloc[self.hell_number, 1] - 1)])
        origins1 = self.canvas.create_rectangle(
            origins1[0] - 20, origins1[1] - 20,
            origins1[0] + 20, origins1[1] + 20,
            fill='LightBlue')

        # 右侧出发点
        origins2 = np.array([20 + UNIT * (self.maze.iloc[self.hell_number + 1, 0] - 1),
                                20 + UNIT * (self.maze.iloc[self.hell_number + 1, 1] - 1)])
        origins2 = self.canvas.create_rectangle(
            origins2[0] - 20, origins2[1] - 20,
            origins2[0] + 20, origins2[1] + 20,
            fill='LightBlue')

        # 目标点
        oval_center1 = np.array([UNIT * (self.maze.iloc[self.hell_number+2, 0]-1) + 20,
                                UNIT * (self.maze.iloc[self.hell_number+2, 1]-1) + 20])
        '''while (oval_center[0] == origin[0]) & (oval_center[1] == origin[1]):
            oval_center = np.array([UNIT * rd.randint(1, MAZE_H - 1) + 20, UNIT * rd.randint(1, MAZE_W - 1) + 20])
        '''

        self.oval1 = self.canvas.create_oval(
            oval_center1[0] - 20, oval_center1[1] - 20,
            oval_center1[0] + 20, oval_center1[1] + 20,
            fill='yellow')

        oval_center2 = np.array([UNIT * (self.maze.iloc[self.hell_number+3, 0]-1) + 20,
                                UNIT * (self.maze.iloc[self.hell_number+3, 1]-1) + 20])
        '''while (oval_center[0] == origin[0]) & (oval_center[1] == origin[1]):
            oval_center = np.array([UNIT * rd.randint(1, MAZE_H - 1) + 20, UNIT * rd.randint(1, MAZE_W - 1) + 20])
        '''

        self.oval2 = self.canvas.create_oval(
            oval_center2[0] - 20, oval_center2[1] - 20,
            oval_center2[0] + 20, oval_center2[1] + 20,
            fill='yellow')

        # create red rect
        self.rect = []

        self.rect1 = self.canvas.create_rectangle(
            origin[0]+20-20 , origin[1]+20-20 ,
            origin[0]+20+20, origin[1]+20+20,
            fill='red')
        self.rect.append(self.rect1)

        self.rect2 = self.canvas.create_rectangle(
            origin[0] + 20 - 20, origin[1] + 20 - 20,
            origin[0] + 20 + 20, origin[1] + 20 + 20,
            fill='red')
        self.rect.append(self.rect2)

        # pack all
        self.canvas.pack()

    def reset(self, episode):
        self.update()
        time.sleep(0.001)
        self.canvas.delete(self.rect1)
        self.canvas.delete(self.rect2)

        origin = np.array([0, 0])
        # 左侧agent
        origins1 = np.array([20 + UNIT * (self.maze.iloc[self.hell_number, 0] - 1),
                            20 + UNIT * (self.maze.iloc[self.hell_number, 1] - 1)])

        self.rect1 = self.canvas.create_rectangle(
            origins1[0] - 20, origins1[1] - 20,
            origins1[0] + 20, origins1[1] + 20,
            fill='red')

        # 右侧agent
        origins2 = np.array([20 + UNIT * (self.maze.iloc[self.hell_number + 1, 0] - 1),
                             20 + UNIT * (self.maze.iloc[self.hell_number + 1, 1] - 1)])
        self.rect2 = self.canvas.create_rectangle(
            origins2[0] - 20, origins2[1] - 20,
            origins2[0] + 20, origins2[1] + 20,
            fill='pink')

        self.rect = []
        self.rectcoords = []
        self.rect.append(self.rect1)
        self.rectcoords.append(self.canvas.coords(self.rect1))
        self.rect.append(self.rect2)
        self.rectcoords.append(self.canvas.coords(self.rect2))

        self.flag = 0  # 统计成功到达的agent个数
        self.firstsuccessagent = -1
        self.firstsuccessgoal = -1

        # return observation
        return self.rectcoords

    def step(self, action):
        s1 = self.canvas.coords(self.rect[0])
        s2 = self.canvas.coords(self.rect[1])
        base_action1 = np.array([0, 0])
        base_action2 = np.array([0, 0])

        if action == 0:  # up+up
            if s1[1] >= UNIT:
                base_action1[1] -= UNIT
            if s2[1] >= UNIT:
                base_action2[1] -= UNIT
        elif action == 1:  # up+down
            if s1[1] >= UNIT:
                base_action1[1] -= UNIT
            if s2[1] <= (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action == 2:  # up+right
            if s1[1] >= UNIT:
                base_action1[1] -= UNIT
            if s2[0] <= (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action == 3:  # up+left
            if s1[1] >= UNIT:
                base_action1[1] -= UNIT
            if s2[0] >= UNIT:
                base_action2[0] -= UNIT
        elif action == 4:  # down+up
            if s1[1] <= (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
            if s2[1] >= UNIT:
                base_action2[1] -= UNIT
        elif action == 5:  # down+down
            if s1[1] <= (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
            if s2[1] <= (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action == 6:  # down+right
            if s1[1] <= (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
            if s2[0] <= (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action == 7:  # down+left
            if s1[1] <= (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
            if s2[0] >= UNIT:
                base_action2[0] -= UNIT
        elif action == 8:  # right+up
            if s1[0] <= (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT
            if s2[1] >= UNIT:
                base_action2[1] -= UNIT
        elif action == 9:  # right+down
            if s1[0] <= (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT
            if s2[1] <= (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action == 10:  # right+right
            if s1[0] <= (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT
            if s2[0] <= (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action == 11:  # right+left
            if s1[0] <= (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT
            if s2[0] >= UNIT:
                base_action2[0] -= UNIT
        elif action == 12:  # left+up
            if s1[0] >= UNIT:
                base_action1[0] -= UNIT
            if s2[1] >= UNIT:
                base_action2[1] -= UNIT
        elif action == 13:  # left+down
            if s1[0] >= UNIT:
                base_action1[0] -= UNIT
            if s2[1] <= (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action == 14:  # left+right
            if s1[0] >= UNIT:
                base_action1[0] -= UNIT
            if s2[0] <= (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action == 15:  # left+left
            if s1[0] >= UNIT:
                base_action1[0] -= UNIT
            if s2[0] >= UNIT:
                base_action2[0] -= UNIT

        if self.firstsuccessagent == 1:  # agent1已到达一个目标点
            base_action1 = np.array([0, 0])
        elif self.firstsuccessagent == 2:  # agent2已到达一个目标点
            base_action2 = np.array([0, 0])
        self.canvas.move(self.rect1, base_action1[0], base_action1[1])  # move agent1
        self.canvas.move(self.rect2, base_action2[0], base_action2[1])  # move agent1

        s_1 = self.canvas.coords(self.rect[0])  # next state
        s_2 = self.canvas.coords(self.rect[1])  # next state

        reward1 = 0
        reward2 = 0

        # reward function奖赏函数
        # agent1
        temp = s1[:]
        if self.firstsuccessagent == 1:
            # agent1已到达目标点，不会再获得奖励
            reward1 = 0
            s_1 = s1
            if self.flag == 2:
                done = 1
            else:
                done = 0
            s_1 = "terminal"
        else:
            if s_1 in self.hells:
                # 走到无底洞，获得-100
                reward1 = r_col
                done = -1
                print("agent1遇到障碍物，-100，返回上一个状态")
                s_1 = temp[:]
                self.canvas.move(self.rect1, -base_action1[0], -base_action1[1])
                self.update()

            # agent1尚未到达过目标点
            elif s_1 == self.canvas.coords(self.oval1):
                # agent1走到goal1，此时需判断goal1是否已经被到达
                if self.flag == 0:
                    reward1 = r_goal
                    # 说明本次是第一次有agent到达goal
                    self.firstsuccessagent = 1
                    self.firstsuccessgoal = 1
                    self.flag = 1
                    s_1 = 'terminal'
                    print("goal1被agent1首次到达，+500！")
                elif self.flag == 1:
                    # 之前agent2已经到达某目标点
                    if self.firstsuccessgoal == 1:
                        # 此时agent1相当于撞到了障碍物
                        # 走到无底洞，获得-20，终止
                        reward1 = r_repeat_goal
                        done = -1
                        print("agent1遇到目标点处的agent2，-20，返回上一个状态")
                        s_1 = temp[:]
                        self.canvas.move(self.rect1, -base_action1[0], -base_action1[1])
                        self.update()
                    else:
                        # 完成任务！
                        reward1 = r_goal
                        self.flag = 2
                        done = 1
                        s_1 = 'terminal'
                        print("2agent到达2goal，+500，完成任务！")

            elif s_1 == self.canvas.coords(self.oval2):
                # agent1走到goal2，此时需判断goal2是否已经被到达
                if self.flag == 0:
                    reward1 = r_goal
                    # 说明本次是第一次有agent到达goal
                    self.firstsuccessagent = 1
                    self.firstsuccessgoal = 2
                    self.flag = 1
                    s_2 = 'terminal'
                    print("goal2被agent1首次到达，+500！")
                elif self.flag == 1:
                    # 之前agent2已经到达某目标点
                    if self.firstsuccessgoal == 2:
                        # 此时agent1相当于撞到了障碍物
                        # 走到无底洞，获得-20，终止
                        reward1 = r_repeat_goal
                        done = -1
                        print("agent1遇到目标点处的agent2，-20，返回上一个状态")
                        s_1 = temp[:]
                        self.canvas.move(self.rect1, -base_action1[0], -base_action1[1])
                        self.update()
                    else:
                        # 完成任务！
                        reward1 = r_goal
                        self.flag = 2
                        done = 1
                        s_1 = 'terminal'
                        print("2agent到达2goal，+500，完成任务！")

            else:
                # 普通位置，-1
                reward1 = r_move
                done = 0


        # agent2
        temp = s2[:]
        if self.firstsuccessagent == 2:
            # agent2已到达目标点，不会再获得奖励
            reward2 = 0
            s_2 = s2
            if self.flag == 2:
                done = 1
            else:
                done = 0
            s_2 = "terminal"
        else:
            # agent2尚未到达过目标点
            if (s_2 == s_1) | ((s_1 == s2) & (s_2 == s1)):
                # 1 2相撞
                if self.firstsuccessagent == 1:
                    reward2 = r_repeat_goal
                    done = -1
                    print("agent2遇到目标点处的agent1，-20，返回上一个状态")
                    s_2 = temp[:]
                    self.canvas.move(self.rect2, -base_action2[0], -base_action2[1])
                    self.update()
                else:
                    reward2 = r_col
                    done = -1
                    print("agent1和agent2运动中相撞，-100，返回上一个状态")
                    s_2 = temp[:]
                    self.canvas.move(self.rect2, -base_action2[0], -base_action2[1])
                    self.update()

            elif s_2 == self.canvas.coords(self.oval1):
                # agent2走到goal1，此时需判断goal1是否已经被到达
                if self.flag == 0:
                    reward2 = r_goal
                    # 说明本次是第一次有agent到达goal
                    self.firstsuccessagent = 2
                    self.firstsuccessgoal = 1
                    self.flag = 1
                    s_2 = 'terminal'
                    print("goal1被agent2首次到达，+500！")
                elif self.flag == 1:
                    # 之前agent1已经到达某目标点
                    if self.firstsuccessgoal == 1:
                        # 此时agent2相当于撞到了障碍物
                        # 走到无底洞，获得-20，终止
                        reward2 = r_repeat_goal
                        done = -1
                        print("agent2遇到目标点处的agent1，-20，返回上一个状态")
                        s_2 = temp[:]
                        self.canvas.move(self.rect2, -base_action2[0], -base_action2[1])
                        self.update()
                    else:
                        # 完成任务！
                        reward2 = r_goal
                        self.flag = 2
                        done = 1
                        s_2 = 'terminal'
                        print("2agent到达2goal，+500，完成任务！")

            elif s_2 == self.canvas.coords(self.oval2):
                # agent2走到goal2，此时需判断goal2是否已经被到达
                if self.flag == 0:
                    reward2 = r_goal
                    # 说明本次是第一次有agent到达goal
                    self.firstsuccessagent = 2
                    self.firstsuccessgoal = 2
                    self.flag = 1
                    print("goal2被agent2首次到达，+500！")
                    s_2 = 'terminal'
                elif self.flag == 1:
                    # 之前agent1已经到达某目标点
                    if self.firstsuccessgoal == 2:
                        # 此时agent2相当于撞到了障碍物
                        # 走到无底洞，获得-20，终止
                        reward2 = r_repeat_goal
                        done = -1
                        print("agent2遇到目标点处的agent1，-20，返回上一个状态")
                        s_2 = temp[:]
                        self.canvas.move(self.rect2, -base_action2[0], -base_action2[1])
                        self.update()
                    else:
                        # 完成任务！
                        reward2 = r_goal
                        self.flag = 2
                        done = 1
                        s_2 = 'terminal'
                        print("2agent到达2goal，+500，完成任务！")

            elif s_2 in self.hells:
                # 走到无底洞，获得-30，终止
                reward2 = r_col
                done = -1
                print("agent2遇到障碍物，-100，返回上一个状态")
                s_2 = temp[:]
                self.canvas.move(self.rect2, -base_action2[0], -base_action2[1])
                self.update()

            else:
                # 普通位置，-10
                reward2 = r_move
                done = 0

        return s_1, s_2, reward1, reward2, done

    def render(self):
        time.sleep(0.001)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
