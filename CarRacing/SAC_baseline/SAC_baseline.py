import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from env import CarRacingEnv
import argparse
import time
from time import strftime, gmtime

parser = argparse.ArgumentParser(description='Train a SAC agent for the CarRacing-v2')
parser.add_argument('--action-repeat', type=int, default=4, metavar='N', help='repeat action in N frames (default: 4)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=1024, metavar='N', help='random seed (default: 1024)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

CNN_OUTPUT = 64

def seed_torch(seed=1024):
    torch.manual_seed(seed)  # 为CPU中设置种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        # self.cnn = nn.Sequential(  # input shape (4, 96, 96)
        #     nn.Conv2d(4, 8, kernel_size=4, stride=2),
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
        #     nn.ReLU(),  # activation
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
        #     nn.ReLU(),  # activation
        # )  # output shape (256, 1, 1)
        self.cnn = nn.Sequential(  #input shape (4,96,96)
            nn.Conv2d(4, 8, kernel_size=8, stride=4),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=5, stride=3),   # (8,23,23)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 7, 7)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # (32, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (64, 1, 1)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = self.cnn(x)
        x = x.view(-1, CNN_OUTPUT)
        return x


class Actor(nn.Module):
    def __init__(self, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.cnn = nn.Sequential(  #input shape (4,96,96)
            nn.Conv2d(4, 8, kernel_size=8, stride=4),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=5, stride=3),   # (8,23,23)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 7, 7)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # (32, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (64, 1, 1)
        self.cnn_output = CNN_OUTPUT  # 最后用超参数替代
        self.max_action = max_action
        self.l1 = nn.Linear(self.cnn_output, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = self.cnn(x)
        x_cnn = x.view(-1, self.cnn_output)
        x = F.relu(self.l1(x_cnn))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = torch.from_numpy(self.max_action) * torch.tanh(a)
        a = a * torch.tensor([1., 0.5, 0.5]) + torch.tensor([0., 0.5, 0.5])
        # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi, x_cnn


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.cnn_output = CNN_OUTPUT
        # Q1
        self.l1 = nn.Linear(self.cnn_output + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(self.cnn_output + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, s, a):
        s = s.view(-1, self.cnn_output)
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class ReplayBuffer(object):
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, 4, 96, 96))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, 4, 96, 96))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class SAC(object):
    def __init__(self, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 100  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(action_dim, self.hidden_width, max_action)
        self.critic = Critic(action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a, _, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.numpy().flatten()

    def save_param(self, epi):
        _dir = "./param/{}_{}.pkl".format("SAC", epi)
        _state = {'actor': self.actor.state_dict(),
                  'actor_optim': self.actor_optimizer.state_dict(),
                  'critic': self.critic.state_dict(),
                  'critic_optim': self.critic_optimizer.state_dict()}
        torch.save(_state, _dir)
        print("save params in {}".format(_dir))

    def load_param(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optim'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optim'])

    def learn(self, replay_buffer):
        batch_s, batch_a, batch_r, batch_s_next, batch_dw = replay_buffer.sample(self.batch_size)  # Sample a batch

        with torch.no_grad():
            batch_a_, log_pi_, batch_s_next_cnn = self.actor(batch_s_next)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_next_cnn, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)
            a, log_pi, batch_s_cnn = self.actor(batch_s)  # *********

        # Compute current Q
        # a, log_pi, batch_s_cnn = self.actor(batch_s)  # **********
        current_Q1, current_Q2 = self.critic(batch_s_cnn, batch_a)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # # Freeze critic networks so you don't waste computational effort
        # for params in self.critic.parameters():
        #     params.requires_grad = False

        # Compute actor loss
        Q1, Q2 = self.critic(batch_s_cnn, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # # Unfreeze critic networks
        # for params in self.critic.parameters():
        #     params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, reward, dead, finished, timeout = env.step(a)
            episode_reward += reward
            if dead or finished or timeout:
                done = True
            s = s_
        evaluate_reward += episode_reward
    env.env.close()
    return int(evaluate_reward / times)


if __name__ == '__main__':
    seed_torch(args.seed)
    env = CarRacingEnv(args, device)
    env.env.action_space.seed(args.seed)
    state_dim = 4*96*96
    action_dim = env.action_dim
    max_action = env.max_action
    replaybuffer_size = int(1e4)
    max_episode_steps = env.max_episode_steps  # Maximum number of steps per episode
    work_path = os.getcwd()
    start_time = time.time()
    print_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print('Working path: {}'.format(work_path))
    print("Begin at {}".format(print_time))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = SAC(action_dim, max_action)
    replay_buffer = ReplayBuffer(replaybuffer_size, state_dim, action_dim)
    # Build a tensorboard
    writer = SummaryWriter(log_dir='./runs/SAC_baseline')

    max_train_steps = 1e5  # Maximum number of training steps
    sample_steps = replaybuffer_size  # Take the random actions in the beginning for the better exploration
    evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    episode = 0  # Record the number of current episode
    running_score = 0  # Record the running score
    training_records = []  # Record something important during the training
    # when steps < sample_steps, only store the random action and state,
    # when steps > sample_steps, begin training network
    train_mode = False

    while total_steps < max_train_steps:
        seed = torch.randint(0, 100000, (1,)).item()
        state = env.reset(seed=seed)
        epi_steps = 0
        done = False
        epi_score = 0
        episode += 1
        if total_steps >= sample_steps:
            train_mode = True
        while not done:
            if not train_mode:  # Take the random actions in the beginning for the better exploration
                a = env.env.action_space.sample()
            else:
                a = agent.choose_action(state)
                # print('action is {}'.format(a))
            state_, reward, dead, finished, timeout = env.step(a)
            epi_steps = env.timestep
            # Set env = gym.make.unwrapped, and set max_episode_steps = 1000,now we have 3 conditions:
            # dead / finished(truncated) / timeout(reach the max_episode_steps).
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            dw = True if dead or finished else False
            if dead or finished or timeout:
                done = True
            # ##  in the first 20 flames, gym is loading the Scenes and the state_ is NOT suitable as an input to the network
            # if epi_steps > 20.0:
            replay_buffer.store(state, a, reward, state_, dw)  # Store the transition
            total_steps += 1
            epi_score += reward
            if train_mode:
                agent.learn(replay_buffer)
            state = state_

            # Evaluate the policy every 'evaluate_freq' steps
            if train_mode and (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                env_evaluate = CarRacingEnv(args, device, render=True)
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)

                end_time = time.time()
                runtime = end_time - start_time
                runtime = strftime("%d-%H:%M:%S", gmtime(runtime))

                print("evaluate_num:{}\tevaluate_reward:{:.2f}\tRunTime:{}".format(evaluate_num, evaluate_reward,
                                                                                   runtime))
                writer.add_scalar('step_rewards_CarRacingV2', evaluate_reward, global_step=total_steps)
                agent.save_param(episode)  # print("save net params in param/SAC_{}
                # Save the rewards
                if evaluate_num % 10 == 0:
                    np.save('./data_train/SAC_evaluate_rewards.npy', np.array(evaluate_rewards))
                    np.save('./data_train/SAC_training_records.npy', training_records)
                    print('save records')

        running_score = running_score * 0.99 + epi_score * 0.01
        training_records.append([episode, epi_score, running_score, seed])
        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {}".format(running_score))
            break
    env.env.close()
