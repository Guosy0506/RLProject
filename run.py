import gym
from gym import envs
import time
env_list = envs.registry
env_ids = [env_item for env_item in env_list]
print('There are {0} envs in gym'.format(len(env_ids)))
print(env_ids)

# 生成环境
env = gym.make('CarRacing-v2', render_mode="human")
# 环境初始化
state = env.reset()
# 循环交互
while True:
    # 从动作空间随机获取一个动作
    action = env.action_space.sample()
    # agent与环境进行一步交互
    state, reward, terminated, truncated, info = env.step(action)
    print('state = {0}; reward = {1}'.format(state, reward))
    # 判断当前episode 是否完成
    if terminated or truncated:
        print('done')
        break
    # time.sleep(1)
# 环境结束
env.close()
