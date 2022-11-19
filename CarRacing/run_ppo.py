import argparse

import numpy as np
import torch

from env import CarRacingEnv
from agents import PPO_Agent
from utils.util import DrawLine

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--changemap', action='store_true', help='change the map to train or validation')
parser.add_argument('--valid', action='store_true', help='if true, valid the net')
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

    training_records = []
    running_score = 0
    Episode = VALID_EPI if args.valid else TRAIN_EPI
    seed = torch.randint(0, 100000, (1,)).item()
    for i_ep in range(Episode):
        score = 0
        if args.changemap:
            seed = torch.randint(0, 100000, (1,)).item()
        state = env.reset(seed=seed)
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
            print('Ep {}\tScore: {:.2f}\tSeed {}'.format(i_ep, score, seed))
        else:  # training mode
            for t in range(1000):
                action, a_logp = agent.select_action(state)
                state_, reward, die, truncated = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                ##  in the first 20 flames, gym is loading the Scenes and the state_ is NOT suitable as an input to the network
                if t > 20.0/args.action_repeat:
                    if agent.store_memory((state, action, a_logp, reward, state_)):
                        agent.update()  # print("params updating")
                        agent.save_param(i_ep)  # print("save net params in param/ppo_{i_ep}")
                    score += reward
                state = state_
                if die or truncated:
                    break
            running_score = running_score * 0.99 + score * 0.01
            training_records.append([i_ep, score, running_score, seed])
            if i_ep % args.log_interval == 0:
                np.save('training_records.npy', training_records)
                if args.vis:
                    draw_reward(xdata=i_ep, ydata=running_score)
                print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            if running_score > env.reward_threshold:
                print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
                break
    env.env.close()
