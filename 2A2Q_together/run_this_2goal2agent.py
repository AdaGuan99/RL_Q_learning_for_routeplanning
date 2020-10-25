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
# from maze_env_2goal_1agent import Maze
from maze_env_2goal2agent_new import Maze
from RL_brain_2goal2agent import QLearningTable
import matplotlib.pyplot as plt
import pandas as pd


def update():
    r_step = []
    r_step_succeed = []  # 记录成功所需步数
    succeed_or_not = []  # 记录每次是否成功
    times_total = 0  # 记录总次数
    times_succeed = 0  # 记录成功次数
    rate_succeed = []  # 记录成功率
    succeed_step_sum = 0  # 记录总的成功步数
    r_step_succeed_cummean = []  # 每次的平均成功步数
    total_stop_times = []  # 成功但是遇到过障碍

    for episode in range(500):  # 进行五百个回合

        if RL.over == 1:  # 判断是否结束
            break
        # initial observation
        # reset 返回self.canvas.coords(self.rect)
        observation = env.reset(episode)  # （1,1）
        print("observation = ")
        print(observation)
        ob1 = observation[0]
        print(ob1)
        ob2 = observation[1]
        print(ob2)

        step = 0  # 修改了一下，不知道是不是对的
        stop_times = 0

        while True:
            # fresh env 把环境刷新一次，获得初始位置
            env.render()
            step += 1
            # RL choose action based on observation
            action = RL.choose_action(ob1, ob2, episode)

            # RL take action and get next observation and reward
            ob_1, ob_2, reward1, reward2, done = env.step(action)
            reward = reward1 + reward2

            # RL learn from this transition
            RL.learn(ob1, ob2, action, reward, ob_1, ob_2)

            # swap observation
            ob1 = ob_1
            ob2 = ob_2

            # break while loop when end of this episode

            if done == 1:
                times_total = times_total + 1
                if stop_times == 0:
                    print("第{}幕：{}步，完美成功！".format(episode, step))
                    times_succeed = times_succeed + 1
                    succeed_or_not.append(1)
                    r_step_succeed.append(step)
                    succeed_step_sum = step + succeed_step_sum
                    r_step_succeed_cummean.append(succeed_step_sum / times_succeed)
                else:
                    print("第{}幕：{}步，曾撞障碍物，但最终成功！".format(episode, step))
                    total_stop_times.append(stop_times)
                    succeed_or_not.append(0)
                break
            elif done == -1 and step <= 100:
                stop_times += 1
            elif step > 100:
                # 超过最大步数，强制终止
                print("第{}幕：{}步，失败！！".format(episode, step))
                times_total = times_total + 1
                succeed_or_not.append(0)
                break
        r_step.append(step)
        rate_succeed.append(times_succeed / times_total)

    # 结果处理
    # 计算结果：result1成功率，result2各幕步数，result3近十次成功率

    result = {'步数': r_step, '是否完美成功': succeed_or_not}
    succeed_or_not = pd.DataFrame(succeed_or_not)
    result = pd.DataFrame(result)
    result1 = times_succeed/times_total       # 成功率
    result2 = r_step        # 各幕步数
    result3 = succeed_or_not.rolling(10).mean()     # 近十次成功率
    result = pd.concat([result, succeed_or_not.rolling(10).mean()], axis=1)
    result.to_excel('D:\\AdaGuan大四上\\SRTP_MID\\RESULT\\8small_results.xlsx')
    print("最终成功率：{}".format(result1))


    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(result3)
    plt.axhline(y=0.8, color='r', linestyle='-')
    plt.title('最近十次成功率')


    plt.figure(2)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(r_step, linewidth=1)
    plt.axhline(y=10, color='r', linestyle='-')
    plt.axhline(y=5, color='r', linestyle='-')
    plt.title('各幕步数')
    plt.show()

    # end of game
    print('game over')
    print(r_step)
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    print(RL.q_table)
    env.after(100, update)
    env.mainloop()
