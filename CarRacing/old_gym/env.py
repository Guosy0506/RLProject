import gym
import numpy as np


class CarRacingEnv(object):
    """
    Environment wrapper for CarRacing
    """

    def __init__(self,
                 isrender,
                 seed,
                 img_stack,
                 action_repeat
                 ):
        if isrender:
            self.env = gym.make('CarRacing-v2', render_mode='human')
        else:
            self.env = gym.make('CarRacing-v2')
        self.seed = seed
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.counter = None
        self.av_r = None
        self.stack = None

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        img_rgb, _ = self.env.reset(seed=self.seed)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack  # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, terminated, truncated, _ = self.env.step(action)
            # don't penalize "die state"
            if truncated:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or terminated or truncated:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, terminated, truncated

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
