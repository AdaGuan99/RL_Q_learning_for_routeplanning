"""
maze_env.py
"""

import numpy as np
import time
import sys
import tkinter as tk
import random as rd
import pandas as pd

UNIT = 40  # pixels每个单元的大小
MAZE_W = 12  # grid width水平方向网格数
MAZE_H = 12  # grid height垂直方向网格数



class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

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
        self.maze = pd.read_csv('maze2.csv')  # 不同的迷宫文件
        self.hell_number = sum(self.maze.genre == 1)
        for i in range(1, self.hell_number+1):
            hell_center = origin+np.array([UNIT * (self.maze.iloc[i-1, 0]-1) + 20,
                                           UNIT * (self.maze.iloc[i-1, 1]-1) + 20])
            self.hell = self.canvas.create_rectangle(
                hell_center[0] - 20, hell_center[1] - 20, hell_center[0] + 20,
                hell_center[1] + 20, fill='grey')
            self.hells.append(self.canvas.coords(self.hell))

        # 创建可能的出发点
        for index in range(1,4):
            origins = np.array([20 + UNIT * (self.maze.iloc[self.hell_number + index - 1, 0] - 1),
                               20 + UNIT * (self.maze.iloc[self.hell_number + index - 1, 1] - 1)])
            origins = self.canvas.create_rectangle(
                origins[0] - 20, origins[1] - 20,
                origins[0] + 20, origins[1] + 20,
                fill='LightBlue')


        #create oval

        oval_center = np.array([UNIT * (self.maze.iloc[self.hell_number+3, 0]-1) + 20,
                                UNIT * (self.maze.iloc[self.hell_number+3, 1]-1) + 20])
        '''while (oval_center[0] == origin[0]) & (oval_center[1] == origin[1]):
            oval_center = np.array([UNIT * rd.randint(1, MAZE_H - 1) + 20, UNIT * rd.randint(1, MAZE_W - 1) + 20])
        '''

        self.oval = self.canvas.create_oval(
            oval_center[0] - 20, oval_center[1] - 20,
            oval_center[0] + 20, oval_center[1] + 20,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0]+20-20 , origin[1]+20-20 ,
            origin[0]+20+20, origin[1]+20+20,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self,episode):
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect)

        # origin = np.array([UNIT * rd.randint(1, MAZE_H - 1) + 20, UNIT * rd.randint(1, MAZE_W - 1) + 20])

        # 随机生成原点
      #  rd.seed(episode)
        index = rd.choice([1,2,3])
        origin = np.array([20 + UNIT * (self.maze.iloc[self.hell_number+index-1, 0]-1),
                           20 + UNIT * (self.maze.iloc[self.hell_number+index-1, 1]-1)])
        print("起点位于：{} {}".format(origin[0], origin[1]))

        self.rect = self.canvas.create_rectangle(
            origin[0] - 20, origin[1] - 20,
            origin[0] + 20, origin[1] + 20,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function奖赏函数
        if s_ == self.canvas.coords(self.oval):
            # 走到目标点，获得+500
            reward = 500
            done = 1
            s_ = 'terminal'
        elif s_ in self.hells:
            # 走到无底洞，获得-30
            reward = -30
            done = -1
            s_ = 'terminal'
        else:
            # 普通位置，-1
            reward = -1
            done = 0

        return s_, reward, done

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
