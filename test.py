import gym
from gym import envs
import numpy as np
import base
import time
import matplotlib.pyplot as plt

env_list = envs.registry
print('There are {0} envs in gym'.format(len(env_list)))
for env_id in env_list.keys():
    print(env_id)


# 生成环境
env = base.Env(seed=999)
agent = base.Agent()
agent.load_param()
# 环境初始化
state = env.reset()
plt.figure()
# 循环交互
while True:
    # 从动作空间随机获取一个动作
    action = agent.select_action(state)
    # agent与环境进行一步交互
    state, reward, done, terminated, truncated, img = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
    print(env.env.observation_space)
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.subplot(1,3,2)

    plt.subplot(1,3,3)

    plt.show()
    # print('state = {0}; reward = {1}'.format(state, reward))
    # 判断当前episode 是否完成
    if terminated or truncated:
        print(terminated)
        print(truncated)
        break
    # time.sleep(1)
# 环境结束
env.env.close()
