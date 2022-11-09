"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
EPISODES = 3000
MEMORY_CAPACITY = 6000
IsValid = True  # If True, play the game once; Make sure IsReuseModel is True
IsTrain = False  # If True, train the model
IsSAVE = False  # If True, save the model in ${SAVE_PATH}
IsReuseModel = True  # If True, load the model in ${SAVE_PATH}
SAVE_PATH = "1e-5.pth"

# env.unwrapped的作用是解除限制，此时CartPole场景不再受时间限制，可以超过200步
env = gym.make('CartPole-v1').unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(128, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.checkpoint = []
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_his = []

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.loss_his.append(loss.data.numpy())
        self.optimizer.step()

    def save_model(self):
        torch.save({
            'EVAL_NET': self.eval_net.state_dict(),
            'TARGET_NET': self.target_net.state_dict(),
            'OPTIMIZER_STATE_DICT': self.optimizer.state_dict(),
            'MEMORY': self.memory,
            'MEMORY_COUNTER': self.memory_counter,
            'LOSS_HISTORY': self.loss_his
        }, SAVE_PATH)

    def load_model(self, path):
        self.checkpoint = torch.load(path)
        self.eval_net.load_state_dict(self.checkpoint['EVAL_NET'])
        self.target_net.load_state_dict(self.checkpoint['TARGET_NET'])
        self.optimizer.load_state_dict(self.checkpoint['OPTIMIZER_STATE_DICT'])
        self.memory = self.checkpoint["MEMORY"]
        self.memory_counter = self.checkpoint['MEMORY_COUNTER']
        self.loss_his = self.checkpoint['LOSS_HISTORY']


def cul_reward(next_state, reward):
    # modify the reward
    x, x_dot, theta, theta_dot = next_state
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.2
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    r = r1 + r2
    reward += r
    return reward


def main():
    dqn = DQN()
    episodes = EPISODES
    t = 0
    reward = []
    mean_reward = []
    if IsReuseModel:
        dqn.load_model(SAVE_PATH)
        episodes = 50
    if IsValid:
        print("\nValidation")
        s = env.reset()
        done = False
        ep_r = 0
        while not done:
            env.render()
            a = dqn.choose_action(s)
            # take action
            s_, r, done, info = env.step(a)
            # modify the reward
            ep_r = cul_reward(s_, ep_r)
            s = s_
        print(ep_r)
    if IsTrain:
        print("Start training...")
        for i_episode in range(episodes):
            s = env.reset()
            ep_r = 0
            while True:
                env.render()
                a = dqn.choose_action(s)
                # take action
                s_, r, done, info = env.step(a)
                # modify the reward
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.2
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

                dqn.store_transition(s, a, r, s_)

                ep_r += r
                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()
                    if done:
                        print('Ep: ', i_episode,
                              '| Ep_r: ', round(ep_r, 2))
                        reward.append(ep_r)
                        mean_reward.append(np.mean(reward))
                        break
                if done:
                    break
                s = s_
            if len(dqn.loss_his) > 1 and dqn.loss_his[-1] < 1e-5:
                print('Finish in Ep: ', i_episode)
                break
        if IsSAVE:
            dqn.save_model()
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(np.arange(len(dqn.loss_his)), dqn.loss_his)
        ax[0].set_ylabel('Cost')
        ax[0].set_xlabel('training steps')
        ax[1].plot(mean_reward)
        plt.show()


if __name__ == '__main__':
    main()
