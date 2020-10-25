"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -20].
Yellow bin circle:      paradise    [reward = +10].
All other states:       ground      [reward = -1].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from b1maze_env import Maze
from b1RL_brain import QLearningTable
import matplotlib.pyplot as plt
import pandas as pd
import time

summary = []


def update(order):
    global summary
    r_step = []
    r_step_succeed = []  # 记录成功所需步数
    succeed_or_not = []  # 记录每次是否成功
    times_total = 0  # 记录总次数
    times_succeed = 0  # 记录成功次数
    rate_succeed = []  # 记录成功率
    succeed_step_sum = 0  # 记录总的成功步数
    r_step_succeed_cummean = []  # 每次的平均成功步数
    total_stop_times = []  # 成功但是遇到过障碍

    for episode in range(1, 600):  # 进行500个回合

        if RL1.over == 1 and RL2.over == 1:  # 判断是否结束，其实后面对epsilon的限制自动在500
            break
        # initial observation
        observation1, observation2 = env.reset()  # （1,1）
        step = 0
        stop_times = 0
        done1 = 0
        done2 = 0

        while True:
            env.render()   # fresh env 把环境刷新一次，获得初始位置
            # 其实也把一些更新放到函数step里面了，因为两个是分开走的，每次都要更新
            step += 1
            # RL choose action based on observation
            if done1 != 1 and done2 != 1:
                action1 = RL1.choose_action(str(observation1), episode)
                action2 = RL2.choose_action(str(observation2), episode)

                # RL take action and get next observation and reward
                observation1_, reward1, done1, meet_or_not1 = env.step(action1, 1)
                env.render()  # 更新
                observation2_, reward2, done2, meet_or_not2 = env.step(action2, 2)

                # RL learn from this transition
                RL1.learn(str(observation1), action1, reward1, str(observation1_), meet_or_not1)
                RL2.learn(str(observation2), action2, reward2, str(observation2_), meet_or_not2)

                # swap observation
                observation1 = observation1_
                observation2 = observation2_

            elif done2 != 1:
                action2 = RL2.choose_action(str(observation2), episode)
                observation2_, reward2, done2, meet_or_not2 = env.step(action2, 2)
                RL2.learn(str(observation2), action2, reward2, str(observation2_), meet_or_not2)
                observation2 = observation2_

            else:
                action1 = RL1.choose_action(str(observation1), episode)
                observation1_, reward1, done1, meet_or_not1 = env.step(action1, 1)
                RL1.learn(str(observation1), action1, reward1, str(observation1_), meet_or_not1)
                observation1 = observation1_

            # break while loop when end of this episode
            if done1 == 1 and done2 == 1:
                times_total = times_total + 1
                if stop_times == 0:
                    print("第{}幕：{}步，完美成功！".format(episode, step))
                    times_succeed = times_succeed + 1
                    succeed_or_not.append(1)
                    r_step_succeed.append(step)
                    succeed_step_sum = step + succeed_step_sum
                    r_step_succeed_cummean.append(succeed_step_sum / times_succeed)
                else:
                    print("第{}幕：{}步，最终成功！".format(episode, step))
                    total_stop_times.append(stop_times)
                    succeed_or_not.append(0)
                break
            elif done1 == -1 or done2 == -1 and step < 100:
                stop_times += 1
            elif step >= 100:
                if done1 == 1:
                    print("第{}幕：{}步，1成功，2失败！！".format(episode, step))
                elif done2 == 1:
                    print("第{}幕：{}步，2成功，1失败！！".format(episode, step))
                else:
                    print("第{}幕：{}步，失败！！".format(episode, step))
                times_total = times_total + 1
                succeed_or_not.append(0)
                break
        r_step.append(step)
        rate_succeed.append(times_succeed / times_total)

    # end of game
    env.destroy()
    print('game over')

    # 结果处理
    result = {'步数': r_step, '是否完美成功': succeed_or_not}
    succeed_or_not = pd.DataFrame(succeed_or_not)
    result = pd.DataFrame(result)
    result1 = times_succeed / times_total  # 成功率
    result2 = r_step  # 各幕步数
    result3 = succeed_or_not.rolling(10).mean()  # 近十次成功率
    result = pd.concat([result, succeed_or_not.rolling(10).mean()], axis=1)
    result.to_excel('./result/results{}.xlsx'.format(order))
    print("最终成功率：{}".format(result1))
    summary.append(result1)

    plt.ion()  # 为了使plt.plot是unblocking，可以继续执行关闭窗口动作，进行下一次循环
    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(result3)
    plt.axhline(y=0.8, color='r', linestyle='-')
    plt.title('最近十次成功率')
    plt.savefig("./figure/十次成功率{}".format(order))

    plt.figure(2)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(r_step, linewidth=1)
    plt.axhline(y=5, color='r', linestyle='-')
    # plt.axhline(y=11, color='r', linestyle='-')
    plt.title('各幕步数')
    plt.savefig("./figure/各幕步数{}".format(order))
    plt.show()
    time.sleep(1)  # 显示一下

    plt.close(1)
    plt.close(2)  # 关闭图窗


if __name__ == "__main__":
    total = []
    for i in range(1, 2):
        env = Maze()
        RL1 = QLearningTable(actions=list(range(env.n_actions)), number=1, order=i)
        RL2 = QLearningTable(actions=list(range(env.n_actions)), number=2, order=i)
        env.after(100, update, i)  # 这个地方，如何获取输出值
        env.mainloop()
    summary = pd.DataFrame(summary)
    summary.to_excel('./result/成功率汇总.xlsx')
