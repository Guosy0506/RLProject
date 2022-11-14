import argparse
import os

import numpy as np
import torch

from env import CarRacingEnv
from agents import PPO_Agent
from utils import DrawLine

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--valid', action='store_true', help='train the net')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

VALID_EPI = 10
TRAIN_EPI = 100000

if __name__ == "__main__":
    # different mode is different in the function: select_action
    mode = 'VALIDATION' if args.valid else 'TRAIN'
    agent = PPO_Agent(img_stack=args.img_stack,
                      gamma=args.gamma,
                      device=device,
                      mode=mode)
    env = CarRacingEnv(is_render=args.render,
                       img_stack=args.img_stack,
                       action_repeat=args.action_repeat)

    # test args param
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")
        print("Vis is true")
    if args.render:
        print("render is true")
    if args.valid:
        print("{}".format(mode))

    training_records = []
    running_score = 0
    state = env.reset()
    Episode = VALID_EPI if args.valid else TRAIN_EPI
    for i_ep in range(Episode):
        score = 0
        state = env.reset()
        if args.valid:  # validation mode
            while True:
                action, _ = agent.select_action(state)
                state_, reward, die, truncated = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                score += reward
                state = state_
                if die:
                    print("Episode is terminated")
                    break
                if truncated:
                    print("Episode is truncated")
                    break
            print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        else:  # training mode
            for t in range(1000):
                action, a_logp = agent.select_action(state)
                state_, reward, die, truncated = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                if agent.store_memory((state, action, a_logp, reward, state_)):
                    print('updating')
                    agent.update()
                score += reward
                state = state_
                if die or truncated:
                    break
            running_score = running_score * 0.99 + score * 0.01

            if i_ep % args.log_interval == 0:
                if args.vis:
                    draw_reward(xdata=i_ep, ydata=running_score)
                print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
                agent.save_param()
            if running_score > env.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
                break
    env.env.close()
