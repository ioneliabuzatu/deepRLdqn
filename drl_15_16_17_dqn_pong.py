import gym
import gym.spaces

import time
import numpy as np
import collections
import torch
import torch.nn as nn  # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package

from psutil import virtual_memory
import warnings

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import collections
import datetime

import os
import random
from collections import deque, namedtuple

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from psutil import virtual_memory

import config
import wandb
from utils import make_atari
from utils import plot
from utils import set_seed
from utils import to_numpy
from utils import wrap_deepmind
from utils import wrap_pytorch
from utils import soft_update

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device, f"\nRAM: {virtual_memory().total / (1024. ** 3)}")

Experience = collections.namedtuple('Experience', ['state', 'action', 'reward', 'done', 'new_state'])


class DQNNetwork_atari(nn.Module):

    def __init__(self, num_actions=4, features_maps=32, activation=nn.ReLU()):
        super().__init__()

        self.num_actions = num_actions
        self.activation = activation

        self.cnn_layer_1 = nn.Conv2d(4, features_maps, 8, stride=4)
        self.cnn_layer_2 = nn.Conv2d(features_maps, features_maps * 2, 4, stride=2)
        self.cnn_layer_3 = nn.Conv2d(features_maps * 2, features_maps * 2, 3, stride=1)

        self.fc_1 = nn.Linear(3136, 512)
        self.fc_2 = nn.Linear(512, self.num_actions)

        nn.init.xavier_normal(self.cnn_layer_1.weight)
        nn.init.xavier_normal(self.cnn_layer_2.weight)
        nn.init.xavier_normal(self.cnn_layer_3.weight)
        nn.init.xavier_normal(self.fc_1.weight)
        nn.init.xavier_normal(self.fc_2.weight)

    def forward(self, x):
        # output forward should always be q values for all actions
        x = torch.tensor(x, dtype=torch.float).to(device)
        if len(x.size()) == 3:
            x = x.unsqueeze(dim=0)
        x = self.cnn_layer_1(x)
        x = self.activation(x)
        x = self.cnn_layer_2(x)
        x = self.activation(x)
        x = self.cnn_layer_3(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.fc_2(x)

        return x


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0):

        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = self.state
            q_vals_v = net(state_a)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


print(">>>Training starts at ", datetime.datetime.now())

MEAN_REWARD_BOUND = 19.0

gamma = 0.99
batch_size = 128
replay_size = 30000
learning_rate = 1e-4
sync_target_frames = 1000
replay_start_size = 10000

epsilon = 1.0
eps_decay = .999985
eps_min = 0.02
tau = 1e-3

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
num_actions = env.action_space.n
state_space = env.observation_space.shape

net = DQNNetwork_atari(num_actions=num_actions).to(device)
target_net = DQNNetwork_atari(num_actions=num_actions).to(device)

buffer = ExperienceReplay(replay_size)
agent = Agent(env, buffer)

if config.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
else:
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=config.momentum)
total_rewards = []
frame_idx = 0

best_mean_reward = None
num_episodes = 900

epsilon_decay = 10 ** 5
epsilon_ub = 1.0
epsilon_lb = 0.02

wandb.run = config.tensorboard.run
timesteps  = 0
# for i in range(num_episodes):
while True:
    frame_idx += 1
    epsilon = max(epsilon * eps_decay, eps_min)
    # epsilon = max(epsilon_lb, epsilon_ub - timesteps / epsilon_decay)

    reward = agent.play_step(net, epsilon)

    # timesteps += 1

    if reward is not None:
        total_rewards.append(reward)

        mean_reward = np.mean(total_rewards[-100:])

        print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (
            frame_idx, len(total_rewards), mean_reward, epsilon))

        wandb.log({"reward":mean_reward}, step=int(len(total_rewards)))

        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), "checkpoint_dqn_good")
            best_mean_reward = mean_reward
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f" % (best_mean_reward))

        if mean_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break

    if len(buffer) < replay_start_size:
        continue

    batch = buffer.sample(batch_size)
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    next_state_values = target_net(next_states_v).max(1)[0]

    next_state_values[done_mask] = 0.0

    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v

    loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss_t.backward()
    optimizer.step()

    soft_update(net, target_net, tau)

    if len(total_rewards) % 50 == 0:
        torch.save({"dqn_state_model": net.state_dict()}, "./checkpoint_model")
        wandb.save(f"./checkpoint_model")
    #     torch.save({
    #         "dqn_state_model": net.state_dict(),
    #         "dqn_target_state_model": target_net.state_dict(),
    #         "epsilon": epsilon,
    #         "buffer": buffer,
    #         "optimizer": optimizer,
    #     },
    #         f"checkpoint")


print(">>>Training ends at ", datetime.datetime.now())
