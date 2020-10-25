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
        self.ovals = []
        self.maze = pd.read_csv('map2.csv')  # 不同的迷宫文件
        self.hell_number = sum(self.maze.genre == 1)
        self.origin_number = sum(self.maze.genre == 2)
        self.oval_number = sum(self.maze.genre == 3)
        for index in self.maze[self.maze.genre == 1].index:
            hell_center = origin + np.array([UNIT * (self.maze.iloc[index, 0] - 1) + 20,
                                             UNIT * (self.maze.iloc[index, 1] - 1) + 20])
            self.hell = self.canvas.create_rectangle(
                hell_center[0] - 20, hell_center[1] - 20, hell_center[0] + 20,
                hell_center[1] + 20, fill='grey')
            self.hells.append(self.canvas.coords(self.hell))

        # 创建可能的出发点
        for index in self.maze[self.maze.genre == 2].index:
            origin_center = origin + np.array([20 + UNIT * (self.maze.iloc[index, 0] - 1),
                                               20 + UNIT * (self.maze.iloc[index, 1] - 1)])
            origins = self.canvas.create_rectangle(
                origin_center[0] - 20, origin_center[1] - 20,
                origin_center[0] + 20, origin_center[1] + 20,
                fill='LightBlue')

        # create oval
        for index in self.maze[self.maze.genre == 3].index:
            oval_center = np.array([UNIT * (self.maze.iloc[index, 0] - 1) + 20,
                                    UNIT * (self.maze.iloc[index, 1] - 1) + 20])
            self.oval = self.canvas.create_oval(
                oval_center[0] - 20, oval_center[1] - 20,
                oval_center[0] + 20, oval_center[1] + 20,
                fill='yellow')
            self.ovals.append(self.canvas.coords(self.oval))

        # create red rect
        self.rect1 = self.canvas.create_rectangle(
            origin[0] + 20 - 20, origin[1] + 20 - 20,
            origin[0] + 20 + 20, origin[1] + 20 + 20,
            fill='red')
        self.rect2 = self.canvas.create_rectangle(
            origin[0] + 20 - 20, origin[1] + 20 - 20,
            origin[0] + 20 + 20, origin[1] + 20 + 20,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect1)
        self.canvas.delete(self.rect2)

        index = self.maze[self.maze.genre == 2].index
        origin1 = np.array([20 + UNIT * (self.maze.iloc[index[0], 0] - 1),
                            20 + UNIT * (self.maze.iloc[index[0], 1] - 1)])
        origin2 = np.array([20 + UNIT * (self.maze.iloc[index[1], 0] - 1),
                            20 + UNIT * (self.maze.iloc[index[1], 1] - 1)])

        self.rect1 = self.canvas.create_rectangle(
            origin1[0] - 20, origin1[1] - 20,
            origin1[0] + 20, origin1[1] + 20,
            fill='purple')
        self.rect2 = self.canvas.create_rectangle(
            origin2[0] - 20, origin2[1] - 20,
            origin2[0] + 20, origin2[1] + 20,
            fill='pink')
        # return observation
        return self.canvas.coords(self.rect1), self.canvas.coords(self.rect2)

    def step(self, action, number):
        if number == 1:
            s = self.canvas.coords(self.rect1)
            # newhell = total_state[1]
            newhell = self.canvas.coords(self.rect2)
        else:
            s = self.canvas.coords(self.rect2)
            # newhell = total_state[0]
            newhell = self.canvas.coords(self.rect1)
        temp = s[:]
        base_action = np.array([0, 0])  # 记录x,y两个方向的运动
        if action == 'u':  # up
            if s[1] >= UNIT:
                base_action[1] -= UNIT
        elif action == 'd':  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'l':  # left
            if s[0] >= UNIT:
                base_action[0] -= UNIT
        elif action == 'r':  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT

        if number == 1:
            self.canvas.move(self.rect1, base_action[0], base_action[1])  # move agent
            s_ = self.canvas.coords(self.rect1)  # next state
        else:
            self.canvas.move(self.rect2, base_action[0], base_action[1])  # move agent
            s_ = self.canvas.coords(self.rect2)  # next state

        meet_or_not = 0
        # reward function奖赏函数
        if s_ in self.ovals and s_ != newhell:
            # 走到目标点，获得+500
            reward = 500
            done = 1
            # s_ = 'terminal'
        elif s_ in self.hells or s_ == newhell:
            # 走到无底洞，获得-30
            if s_ == newhell:
                meet_or_not = 1
                print(number, "遇到对方，返回上一个状态")
            else:
                print(number, "遇到障碍物，返回上一个状态")
            reward = -30
            done = -1
            s_ = temp[:]
            self.render()
            if number == 1:
                self.canvas.move(self.rect1, -base_action[0], -base_action[1])
            else:
                self.canvas.move(self.rect2, -base_action[0], -base_action[1])
        else:
            # 普通位置，-1
            reward = -1
            done = 0
            print(number, '正常走一步')

        return s_, reward, done

    def render(self):
        time.sleep(0.01)
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
