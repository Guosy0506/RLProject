from env import CarRacingEnv
from SAC_baseline import SAC
import argparse
import torch
import numpy as np
import os

parser = argparse.ArgumentParser(description='Train a SAC agent for the CarRacing-v2')
parser.add_argument('--action-repeat', type=int, default=4, metavar='N', help='repeat action in N frames (default: 4)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=1024, metavar='N', help='random seed (default: 1024)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if __name__ == '__main__':
    env = CarRacingEnv(args, device, render=True)
    agent = SAC(env.action_dim, env.max_action)
    print("test")
    date = '0316'
    # index_list = [62,84,107,130,154,176,198,221,242,265,288,309,331,352,374,396,418,441]  # 0312
    # 155,214,275,334,394,450,505,562,
    # index_list = [622,677,734,793,857,920,977,1035,1089,1149,1211]
    # index_list = [1149]
    index_list = [52,62,73,83,94,104,115,125,135,146,157,167,177,188,199,209,219,230,240]
    times = 1  # Perform three evaluations and calculate the average
    for index in index_list:
        model_path = './param/{}/SAC_{}.pkl'.format(date,index)
        agent.load_param(model_path)
        print("load param in Epi_{}".format(index))
        evaluate_reward = 0
        for i in range(times):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.choose_action(state, deterministic=False)  # We use the deterministic policy during the evaluating
                state_, reward, dead, finished, timeout = env.step(action)
                episode_reward += reward
                if dead or finished or timeout:
                    done = True
                state = state_
            print("episode {} reward is {}".format(i,episode_reward))
            evaluate_reward += episode_reward
        # print("average reward of index {} is {}".format(index, evaluate_reward))
    env.env.close()
    # time.sleep(1)
    # 环境结束


