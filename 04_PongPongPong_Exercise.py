import datetime
import warnings
from collections import deque
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from psutil import virtual_memory

import config
import wandb
from utils import make_atari
from utils import set_seed
from utils import soft_update
from utils import wrap_deepmind
from utils import wrap_pytorch

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device, f"\nRAM: {virtual_memory().total / (1024. ** 3)}")

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


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


class ReplayBuffer:
    def __init__(self, memory_len):
        self.memory_len = memory_len
        self.buffer = deque(maxlen=memory_len)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) > self.memory_len:
            self.remove()
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states_mb, mb_a, mb_reward, next_states_mb, mb_done = zip(*[self.buffer[idx] for idx in indices])

        states_mb = np.array(states_mb)
        mb_a = np.array(mb_a)
        mb_reward = np.array(mb_reward, dtype=np.float32)
        next_states_mb = np.array(next_states_mb)
        mb_done = np.array(mb_done, dtype=np.uint8)

        states_mb = torch.tensor(states_mb).to(device)
        mb_a = torch.tensor(mb_a).to(device)
        mb_reward = torch.tensor(mb_reward).to(device)
        next_states_mb = torch.tensor(next_states_mb).to(device)
        mb_done_mask = torch.ByteTensor(mb_done).to(device)

        return states_mb, mb_a, mb_reward, next_states_mb, mb_done_mask

    def remove(self):
        self.buffer.popleft()


wandb.run = config.tensorboard.run
print(">>>Training starts at ", datetime.datetime.now())
num_episodes = 120
buffer_size = 10000
epsilon = 1.0
minibatch_size = 32
gamma = 0.99
eval_episode = 100
num_eval = 10
tau = 1e-3
learning_rate = 1e-4
update_after = 10000
epsilon_decay = 10 ** 5
epsilon_ub = 1.0
epsilon_lb = 0.02

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
num_actions = env.action_space.n
state_space = env.observation_space.shape

set_seed(env, 0)

dqn = DQNNetwork_atari(num_actions=num_actions).to(device)
dqn_target = DQNNetwork_atari(num_actions=num_actions).to(device)
buffer = ReplayBuffer(buffer_size)

returns = []
returns_50 = deque(maxlen=50)
losses = []
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
mse = torch.nn.MSELoss()

best_mean_reward = -30

time_steps = 0
state = env.reset()
for i in range(num_episodes):
    ret = 0.0
    done = False
    while not done:

        epsilon = max(epsilon_lb, epsilon_ub - time_steps / epsilon_decay)

        if np.random.choice([0, 1], p=[1 - epsilon, epsilon]) == 1:
            action = env.action_space.sample()
        else:
            state_a = state
            q_vals_v = dqn(state_a)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, r, done, _ = env.step(action)
        ret += r

        buffer.add(Transition(state, action, r, new_state, done))

        state = new_state
        time_steps += 1

        # update policy using temporal difference
        if len(buffer) > minibatch_size and len(buffer) >= update_after:
            optimizer.zero_grad()

            states_mb, mb_a, mb_reward, next_states_mb, mb_done = buffer.sample_batch(minibatch_size)

            state_action_values = dqn(states_mb).gather(1, mb_a.unsqueeze(-1)).squeeze(-1)

            next_state_values = dqn_target(next_states_mb).max(1)[0]

            next_state_values[mb_done] = 0.0

            next_state_values = next_state_values.detach()

            expected_state_action_values = mb_reward + next_state_values * gamma

            loss = mse(state_action_values, expected_state_action_values)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print(loss.item())
            wandb.log({"loss": loss.item()}, step=time_steps)

            soft_update(dqn, dqn_target, tau)

        if done:
            env.reset()
            returns.append(ret)

            mean_reward = np.mean(returns[-50:])
            games = len(returns)

            print(f"played {games} games, mean reward {mean_reward:.3f}, epsilon {epsilon:.3f}")

            wandb.log({"reward": mean_reward, "step_reward": i})

            if (best_mean_reward - mean_reward) > 1:
                torch.save(dqn.state_dict(), f"checkpoint_dqn_best_{mean_reward}")
                best_mean_reward = mean_reward
                print(f"Best mean reward updated {best_mean_reward:.3f} at game {games}")

print(">>>Training ends at ", datetime.datetime.now())
