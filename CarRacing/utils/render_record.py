import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.nn as nn
import imageio.v2 as imageio


class Env(object):
    """
    Test environment wrapper for CarRacing
    """

    def __init__(self, seed):
        # self.env = gym.make('CarRacing-v2', render_mode="human")
        self.env = gym.make('CarRacing-v2', render_mode="human")
        self.seed = seed
        self.av_r = None
        self.timestep = None

    def reset(self):
        self.timestep = 0
        self.av_r = self.reward_memory()
        img_rgb, _ = self.env.reset(seed=self.seed)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * 4
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        die = False
        for i in range(8):
            img_rgb, reward, terminated, truncated, _ = self.env.step(action)
            on_grass = np.mean(img_rgb[64:78, 42:54, 1])  # channel 1 has the most difference
            if terminated:
                reward += 100
            if on_grass > 160:
                reward -= 0.06
            if self.av_r(reward) <= -0.1 or terminated:
                die = True
            total_reward += reward
            self.timestep += 1
            if die or truncated:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == 4
        return np.array(self.stack), total_reward, die, truncated, img_rgb, self.timestep

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

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray.astype('float32') / 128 - 1
        return gray[0:84, 6:90]


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        # self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
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
        self.cnn_base = nn.Sequential(  # input shape (4, 84, 84)
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
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
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(torch.device("cpu"))

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(torch.device("cpu")).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)
        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self, index, path):
        _path = '{}/param/ppo_{}.pkl'.format(path, index)
        try:
            f = open(_path)
            f.close()
        except IOError:
            return False
        self.net.load_state_dict(torch.load(_path, map_location=torch.device("cpu")))
        return True


def create_gif(image_list, gif_name, duration=1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    path = "D:/git/project/CarRacing/ModifyPPOinv2"
    record = np.load("{}/training_records.npy".format(path))
    i_ep = record[:, 0]
    score = record[:, 1]
    running_score = record[:, 2]
    seeds = record[:, 3]
    # index_list = [1820 1840 1890 1940 1960 3480 3490 3510 3520 3540 3660
    #               3700 4120 5210 5230 5250 8460 8480 8500 8510 8570 8630] #PPOinv2
    # index_list = [3013, 3030, 3047, 3063, 3081, 3098, 3114, 3166, 3182, 3199, 3216, 3251, 3285, 3339,
    #               3357, 3431, 3490, 3530, 3548, 3637, 3690, 3707, 3725, 3798, 3817, 3852, 3870] # ModifyPPOinv2
    index_list = [3870]
    is_save_gif = False
    reward_threshold = 650
    agent = Agent()
    for index in i_ep:
        if index < 3889:
            continue
        index = index.astype(int).item()
    # for index in index_list:
        seed = seeds[index].astype(int).item()
        print("Epi_{}, seed is {}".format(index, seed))
        if agent.load_param(index, path):
            print("load param in Epi_{}".format(index))
        env = Env(seed=seed)
        image_list = []
        # 环境初始化
        state = env.reset()
        total_reward = 0
        total_step = 0
        # 循环交互
        while True:
            # 从动作空间随机获取一个动作
            action = agent.select_action(state)
            # agent与环境进行一步交互
            state, reward, died, truncated, image, timestep = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            total_reward += reward
            total_step = timestep
            image_list.append(image)
            # 判断当前episode 是否完成
            if died:
                print("Episode is FAIL! Reward is {:.2f}, STEP is {}".format(total_reward, total_step))
                break
            if truncated:
                print("Episode is SUCCESS! Reward is {:.2f}, STEP is {}".format(total_reward, total_step))
                break
        if is_save_gif and total_reward >= reward_threshold:
            fps = 50
            gif_name = '{}/gif/{}.gif'.format(path, index)
            imageio.mimwrite(gif_name, image_list, fps=fps)  # 覆写当前文件
    env.env.close()


if __name__ == "__main__":
    main()
