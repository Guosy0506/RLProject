import time

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # self.reward_threshold = self.env.spec.reward_threshold  # 900
        self.reward_threshold = 600
        self.img_stack = args.img_stack
        self.action_repeat = args.action_repeat
        self.device = device

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
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, terminated, truncated, _ = self.env.step(action)
            # # don't penalize "die state"
            # if terminated:
            #     reward += 100
            # grass penalty
            on_grass = np.mean(img_rgb[64:78, 42:54, 1])  # channel 1 has the most difference
            if on_grass > 150:
                reward -= 200.0
            # speed_reward = (action[1] - action[2]) * 0.05
            # reward += max(speed_reward, 0)
            self.timestep += 1
            total_reward += reward
            # if no reward recently, end the episode
            if truncated:
                self.finished = True
                break
            if self.timestep == self.max_episode_steps:
                self.timeout = True
                break
            # 以下判断条件需配合单步奖励的最低分进行实时调整，否则扣分过多，会直接判定当圈状态为dead
            if self.av_r(reward) <= -10 or terminated:
                self.dead = True
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, self.dead, self.finished, self.timeout

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        # return gray[0:84, 6:90]  # in python,0:84 means index from 0 to 83
        return gray

    @staticmethod
    def reward_memory():
        # only calculate ave_reward for last {length} steps, if smaller
        # than -0.1,the episode is died
        count = 0
        length = 50
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
