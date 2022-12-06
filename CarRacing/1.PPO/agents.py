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

    def __init__(self, args):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 84, 84)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
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
    buffer_capacity, batch_size = 1500, 128

    # 参数声明，传参初始化
    img_stack = None
    gamma = None
    device = None
    transition = None

    # def __init__(self, img_stack, gamma, device, mode='TRAIN'):
    def __init__(self, args, device):
        self.img_stack = args.img_stack
        self.gamma = args.gamma
        self.device = device
        self.transition = np.dtype([('s', np.float64, (args.img_stack, 84, 84)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                                    ('r', np.float64), ('s_', np.float64, (args.img_stack, 84, 84))])
        self.training_step = 0
        self.net = Net(args).double().to(self.device)
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0
        self.istrain = args.train
        self.lr = args.lr  # Learning rate
        self.use_lr_decay = args.use_lr_decay
        self.max_train_steps = args.max_train_steps
        if self.istrain:
            print("Agent mode is TRAIN.")
        else:
            self.load_param()
            print("Agent mode is VALIDATION.")
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        action = None
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        if self.istrain:
            dist = Beta(alpha, beta)
            action = dist.sample()
            a_logp = dist.log_prob(action).sum(dim=1)
            a_logp = a_logp.item()
        else:
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
        self.net.load_state_dict(torch.load('../ModifyPPOinv2/param/ppo_3870.pkl', map_location=self.device))

    def store_memory(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self, running_score, reward_threshold):
        if self.device == "cuda":
            s = torch.cuda.DoubleTensor(self.buffer['s'], device=self.device)
            a = torch.cuda.DoubleTensor(self.buffer['a'], device=self.device)
            r = torch.cuda.DoubleTensor(self.buffer['r'], device=self.device).view(-1, 1)
            s_ = torch.cuda.DoubleTensor(self.buffer['s_'], device=self.device)
            old_a_logp = torch.cuda.DoubleTensor(self.buffer['a_logp'], device=self.device).view(-1, 1)
        else:
            s = torch.tensor(self.buffer['s'], dtype=torch.double, device=self.device)
            a = torch.tensor(self.buffer['a'], dtype=torch.double, device=self.device)
            r = torch.tensor(self.buffer['r'], dtype=torch.double, device=self.device).view(-1,1)
            s_ = torch.tensor(self.buffer['s_'], dtype=torch.double, device=self.device)
            old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double, device=self.device)

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

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(running_score, reward_threshold)

    def lr_decay(self, cumulative_reward, reward_threshold):
        lr_now = self.lr * (1 - cumulative_reward / reward_threshold)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

