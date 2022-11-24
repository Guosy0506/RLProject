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
        self.env = gym.make('CarRacing-v2', render_mode="human").unwrapped
        self.seed = seed

    def reset(self):
        img_rgb, _ = self.env.reset(seed=self.seed)
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * 4
        return np.array(self.stack)

    def step(self, action):
        img_rgb, _, terminated, truncated, _ = self.env.step(action)
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == 4
        return np.array(self.stack), terminated, truncated, img_rgb

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray.astype('float32') / 128 - 1
        return gray


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
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

    def load_param(self, index):
        self.net.load_state_dict(torch.load('D:/git/project/CarRacing/PPOinv2/param/ppo_{}.pkl'.format(index),
                                            map_location=torch.device("cpu")))


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
    record = np.load("D:/git/project/CarRacing/PPOinv2/training_records.npy")
    i_ep = record[:, 0]
    score = record[:, 1]
    running_score = record[:, 2]
    seeds = record[:, 3]
    # 1820 (1840 1890) 1940 1960 3480(better) 3490 3510 3520 (3540) (3660(缺少横向维度数据) 3700) 4120 5210 5230 5250
    # 8460 8480(best) 8500 8510(G) 8570 8630
    is_save_gif = False
    index_list = [1820, 1840, 1890, 1940, 1960, 3480, 3490, 3510, 3520, 3540, 3660,
                  3700, 4120, 5210, 5230, 5250, 8460, 8480, 8500, 8510, 8570, 8630]
    index_list = [3480]
    for index in index_list:
        print(index)
        seed = seeds[index].astype(int).item()
        print("seed is {}".format(seed))
        print(i_ep[index].astype(int).item())
        env = Env(seed=seed)
        agent = Agent()
        agent.load_param(index)
        image_list = []
        # 环境初始化
        state = env.reset()
        # 循环交互
        while True:
            # 从动作空间随机获取一个动作
            action = agent.select_action(state)
            # agent与环境进行一步交互
            state, terminated, truncated, image = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            image_list.append(image)
            # 判断当前episode 是否完成
            if terminated:
                print("Episode is terminated!")
                break
            if truncated:
                print("Episode is truncated!")
                break
        env.env.close()
        if is_save_gif:
            fps = 50
            gif_name = 'D:/git/project/CarRacing/PPOinv2/gif/{}.gif'.format(index)
            imageio.mimwrite(gif_name, image_list, fps=fps)


if __name__ == "__main__":
    main()
