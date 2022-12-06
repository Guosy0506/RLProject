import argparse

import numpy as np
import torch

from env import CarRacingEnv
from agents import PPO_Agent
from util import DrawLine

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of actor")
parser.add_argument("--max_train_steps", type=int, default=int(1e7), help=" Maximum number of training steps")
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument("--use_lr_decay", type=bool, default=True, help="automatic decay learning rate")
parser.add_argument('--use_changing_map', type=bool, default=True, help='whether the map is changing')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--train', action='store_true', help='if true, train the net')
parser.add_argument('--transfer_learning', action='store_true', help='if true, transfer_learning')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

VALID_EPI = 100
TRAIN_EPI = 100000

if __name__ == "__main__":
    # different mode is different in the function: select_action
    agent = PPO_Agent(args, device=device)
    env = CarRacingEnv(args)
    if args.transfer_learning:
        agent.load_param('../ModifyPPOinv2/param/ppo_3870.pkl')
    # test args param
    if args.vis:
        draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")
        print("Vis is true")
    if args.render:
        print("render is true")

    training_records = []
    running_score = 0
    Episode = TRAIN_EPI if args.train else VALID_EPI
    seed = torch.randint(0, 100000, (1,)).item()
    for i_ep in range(Episode):
        score = 0
        if args.use_changing_map:
            seed = torch.randint(0, 100000, (1,)).item()
        state = env.reset(seed=seed)
        if args.train:  # training mode
            for t in range(1000):
                action, a_logp = agent.select_action(state)
                state_, reward, die, truncated = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                ##  in the first 20 flames, gym is loading the Scenes and the state_ is NOT suitable as an input to the network
                if t > 20.0/args.action_repeat:
                    if agent.store_memory((state, action, a_logp, reward, state_)):
                        print("param updating")
                        agent.update(running_score, env.reward_threshold)
                        agent.save_param(i_ep)  # print("update and save net params in param/ppo_{i_ep}")
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
        else:   # validation mode
            while True:
                action, _ = agent.select_action(state)
                state_, reward, die, truncated = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                score += reward
                state = state_
                if die:
                    break
                if truncated:
                    break
            print('Ep {}\tScore: {:.2f}\tSeed {}'.format(i_ep, score, seed))
    env.env.close()
