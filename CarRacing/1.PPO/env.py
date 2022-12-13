import time

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt


class CarRacingEnv(object):
    """
    Environment wrapper for CarRacing
    """

    # def __init__(self, is_render, img_stack, action_repeat):
    def __init__(self, args):
        if args.render:
            self.env = gym.make('CarRacing-v2', render_mode='human')
        else:
            self.env = gym.make('CarRacing-v2')
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = args.img_stack
        self.action_repeat = args.action_repeat
        self.counter = None
        self.av_r = None
        self.stack = None

    def reset(self, seed=None):
        self.counter = 0
        self.av_r = self.reward_memory()
        img_rgb, _ = self.env.reset(seed=seed)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        die = False
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
            total_reward += reward
            # if no reward recently, end the episode
            if self.av_r(reward) <= -0.1 or terminated:
                die = True
            if die or truncated:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, die, truncated

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
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
