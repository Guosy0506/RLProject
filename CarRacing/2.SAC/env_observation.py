import time

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class Observation(nn.Module):
    def __init__(self, img_stack, device):
        super(Observation, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 84, 84)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 41, 41)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=2, stride=2),  # (16, 20, 20)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=2, stride=2),  # (32, 10, 10)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.output_dim = 256
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.load_state_dict(torch.load("ppo_6727.pkl", map_location=device))

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        return x


class CarRacingEnv(object):
    """
    Environment wrapper for CarRacing
    """

    # def __init__(self, is_render, img_stack, action_repeat):
    def __init__(self, args, device, render=False):
        if render:
            self.env = gym.make('CarRacing-v2', render_mode='human').unwrapped
        else:
            self.env = gym.make('CarRacing-v2').unwrapped
        self.reward_threshold = self.env.spec.reward_threshold  # 900
        self.img_stack = args.img_stack
        self.action_repeat = args.action_repeat
        self.device = device
        self.obs = Observation(self.img_stack, self.device)

        self.state_dim = self.obs.output_dim
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high
        self.max_episode_steps = 1000
        self.timestep = None

        self.counter = None
        self.av_r = None
        self.stack = None

    def reset(self, seed=None):
        self.counter = 0
        self.timestep = 0
        self.av_r = self.reward_memory()
        # condition
        self.dead = False
        self.finished = False
        self.timeout = False
        img_rgb, _ = self.env.reset(seed=seed)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        state = torch.from_numpy(np.array(self.stack)).float().unsqueeze(0)
        obs = self.obs(state).detach().numpy()
        return obs

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, terminated, truncated, _ = self.env.step(action)
            # don't penalize "die state"
            if terminated:
                reward += 100
            # grass penalty
            on_grass = np.mean(img_rgb[64:78, 42:54, 1])  # channel 1 has the most difference
            if on_grass > 160:
                reward -= 0.06
                ##  to test.py the values of on_grass, un-comment the codes below:
                # plt.imshow(img_rgb[64:78, 42:54, 1])
                # plt.title("{}".format(on_grass))
                # plt.pause(0.2)
            speed_reward = (action[1] - action[2]) * 0.1
            reward += max(speed_reward, 0)
            self.timestep += 1
            total_reward += reward
            # if no reward recently, end the episode
            if truncated:
                self.finished = True
                total_reward += 100
                break
            if self.timestep == self.max_episode_steps:
                self.timeout = True
                total_reward += 50
                break
            if self.av_r(reward) <= -0.1 or terminated:
                self.dead = True
                total_reward -= 100
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        state = torch.from_numpy(np.array(self.stack)).float().unsqueeze(0)
        obs = self.obs(state).detach().numpy()
        return obs, total_reward, self.dead, self.finished, self.timeout

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray[0:84, 6:90]  # in python,0:84 means index from 0 to 83

    @staticmethod
    def reward_memory():
        # only calculate ave_reward for last {length} steps, if smaller
        # than -0.1,the episode is died
        count = 0
        length = 70
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
