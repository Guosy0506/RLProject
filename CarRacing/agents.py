import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, img_stack):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
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
        x = x.view(-1, 256)  # reshape()
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class PPO_Agent(object):
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    # 参数声明，传参初始化
    img_stack = None
    gamma = None
    device = None
    transition = None

    def __init__(self, img_stack, gamma, device, mode='TRAIN'):
        self.img_stack = img_stack
        self.gamma = gamma
        self.device = device
        self.transition = np.dtype([('s', np.float64, (img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                                    ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96))])
        self.training_step = 0
        self.net = Net(img_stack).double().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0
        self.mode = mode
        print("Agent mode is {}.".format(mode))
        if self.mode == "VALIDATION":
            self.load_param()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        action = None
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        if self.mode == 'TRAIN':
            dist = Beta(alpha, beta)
            action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1)
            a_logp = a_logp.item()
        if self.mode == 'VALIDATION':
            action = alpha / (alpha + beta)
            a_logp = []
        assert action is not None, 'mode is wrong!'
        action = action.squeeze().cpu().numpy()
        return action, a_logp

    def save_param(self, epi):
        _dir = "param/{}_{}.pkl".format("ppo", epi)
        torch.save(self.net.state_dict(), _dir)
        print("update and save params in {}".format(_dir))

    def load_param(self):
        self.net.load_state_dict(torch.load('param/ppo_net_origin.pkl', map_location=self.device))

    def store_memory(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        print("params updating")
        self.training_step += 1
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
