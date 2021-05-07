#!/usr/bin/env python
# coding: utf-8

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

os.environ["PATH"] += os.pathsep + "/usr/bin/xdpyinfo"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device, f"\nRAM: {virtual_memory().total / (1024. ** 3)}")

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# TODO: Implement Prioritized Replay Buffer
class ReplayBuffer():
    def __init__(self, num_actions, memory_len=10000):
        self.memory_len = memory_len
        self.transition = []
        self.num_actions = num_actions

    def add(self, state, action, reward, next_state, done):
        if self.length() > self.memory_len:
            self.remove()
        self.transition.append(Transition(state, action, reward, next_state, done))

    def sample_batch(self, batch_size=32):
        minibatch = random.sample(self.transition, batch_size)

        states_mb, a_, reward_mb, next_states_mb, done_mb = map(np.array, zip(*minibatch))

        mb_reward = torch.from_numpy(reward_mb).to(device)
        mb_done = torch.from_numpy(done_mb.astype(int)).to(device)

        a_mb = np.zeros((a_.size, self.num_actions))
        a_mb[np.arange(a_.size), a_] = 1
        mb_a = torch.from_numpy(a_mb).to(device)

        return states_mb, mb_a, mb_reward, next_states_mb, mb_done

    def length(self):
        return len(self.transition)

    def remove(self):
        self.transition.pop(0)


class DQNNetwork_atari(nn.Module):

    def __init__(self, num_actions=4, features_maps=32, activation=nn.ReLU()):
        super().__init__()

        self.num_actions = num_actions
        self.activation = activation

        self.cnn_layer_1 = nn.Conv2d(4, features_maps, 8, stride=4)
        self.cnn_layer_2 = nn.Conv2d(features_maps, features_maps * 2, 4, stride=2)
        self.cnn_layer_3 = nn.Conv2d(features_maps * 2, features_maps * 2, 3, stride=1)

        feature_map_for_linear_layer = calculate_output_features(
            [self.cnn_layer_1, self.cnn_layer_2, self.cnn_layer_3], 84
        )

        # self.fc_1 = nn.Linear(feature_map_for_linear_layer, 600)
        self.fc_1 = nn.Linear(3136, 512)
        self.fc_2 = nn.Linear(512, self.num_actions)

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


def calculate_output_features(layers: list, input_width: int):
    """ Initially the feature_map_size is the width of the input """
    for layer in layers:
        padding = layer.padding[0]
        stride = layer.stride[0]
        kernel = layer.kernel_size[0]
        feature_map_size = (input_width - kernel + 2 * padding) / stride + 1
    return int(feature_map_size)


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """
    target_model.weights = tau * local_model.weights + (1 - tau) * target_model.weights
    # for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #     target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def main():
    checkpoint_path = "checkpoint"

    wandb.run = config.tensorboard.run

    num_episodes = 700  # number of episodes to run the algorithm
    buffer_size = 10 ** 4  # size of the buffer to use
    epsilon = 1.0  # initial probablity of selecting random action a, annealed over time
    minibatch_size = 64  # size of the minibatch sampled
    gamma = 0.99  # discount factor
    eval_episode = 100
    num_eval = 10
    tau = 1e-3  # hyperparameter for updating the target network
    learning_rate = 1e-4
    update_after = 1000  # update after num time steps
    # epsilon_decay = 10 ** 5
    epsilon_decay = .999985
    epsilon_ub = 1.0
    epsilon_lb = 0.02

    eps_start = 1.0
    eps_min = 0.02
    sync_target_frames = 1000

    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    num_actions = env.action_space.n
    state_space = env.observation_space.shape

    set_seed(env, 0)

    dqn = DQNNetwork_atari(num_actions=num_actions).to(device)
    dqn_target = DQNNetwork_atari(num_actions=num_actions).to(device)
    buffer = ReplayBuffer(num_actions=num_actions, memory_len=buffer_size)

    # try:
    #     load_checkpoint = torch.load(checkpoint_path)
    #     dqn.load_state_dict(load_checkpoint["dqn_state_model"])
    #     dqn_target.load_state_dict(load_checkpoint["dqn_target_state_model"])
    #     # epsilon_ub = load_checkpoint["epsilon"]
    #     buffer = load_checkpoint["buffer"]
    # except FileNotFoundError:
    #     print("No checkpoint found")

    # Train the agent using DQN for Pong
    returns = []
    returns_50 = deque(maxlen=50)
    losses = []
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss()

    state = env.reset()
    timesteps = 0
    dones = 0
    for i in range(num_episodes):
        ret = 0
        done = False
        while not done:
            # Decay epsilon :
            # epsilon = max(epsilon_lb, epsilon_ub - timesteps / epsilon_decay)
            epsilon = max(epsilon * epsilon_decay, eps_min)
            # action selection
            if np.random.choice([0, 1], p=[1 - epsilon, epsilon]) == 1:
                a = np.random.randint(low=0, high=num_actions, size=1)[0]
            else:
                net_out = dqn(state).detach().cpu().numpy()
                a = np.argmax(net_out)
            next_state, r, done, info = env.step(a)
            ret = ret + r

            buffer.add(state, a, r, next_state, done)

            state = next_state
            timesteps = timesteps + 1

            # update policy using temporal difference
            if buffer.length() > minibatch_size and buffer.length() > update_after:
                optimizer.zero_grad()

                states_mb, mb_a, mb_reward, next_states_mb, mb_done = buffer.sample_batch()

                next_q = torch.max(dqn_target(next_states_mb), 1)[0]
                targets = mb_reward + (gamma * next_q * (1 - mb_done))

                predictions = dqn(states_mb)

                loss = mse(predictions.float(), targets.unsqueeze(-1).float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                wandb.log({"loss": loss.item()}, step=timesteps)

                # soft_update(dqn, dqn_target, tau)
                if timesteps % sync_target_frames == 0:
                    dqn_target.load_state_dict(dqn.state_dict())
            if done:
                state = env.reset()
                print(f"Episode: {i} {ret} epsilon={epsilon:3f}")
                wandb.log({"Reward": ret}, step=timesteps)
                wandb.log({"step_reward": i, "epsilon": epsilon})
                dones += 1
                break
        returns.append(ret)
        returns_50.append(ret)
        if i % 20 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(returns_50)))
        if i % 20 == 0:
            if not os.path.exists("./checkpoints/"):
                os.makedirs("./checkpoints/")
            torch.save({
                "dqn_state_model": dqn.state_dict(),
                "dqn_target_state_model": dqn_target.state_dict(),
                "epsilon": epsilon,
                "buffer": buffer,
            },
                f"checkpoint")

    plot(timesteps, returns, losses)

    state = env.reset()

    torch.onnx.export(dqn,  # model
                      torch.tensor(state, dtype=torch.float),  # example model input
                      'submission_onnx.onnx',  # name of the submission
                      export_params=True,  # save trained parameters
                      opset_version=10,
                      do_constant_folding=True)

    # test your onnx model using runtime - we are going to use this to test your model
    # on the server.

    ort_session = onnxruntime.InferenceSession("submission_onnx.onnx")

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.tensor(state, dtype=torch.float))}
    ort_outs = ort_session.run(None, ort_inputs)

    # check if output matches
    ort_outs, dqn(state)

    # test on the env using onnx model
    # We are going to use this block of code for testing your model on the server.
    # Make sure your model works in this block of code, before submitting.
    return_ = []
    for i in range(50):
        state = env.reset()
        ret = 0
        done = False
        while not done:
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch.tensor(state, dtype=torch.float))}
            q_actions = ort_session.run(None, ort_inputs)
            action = np.argmax(q_actions)
            state, r, done, _ = env.step(action)
            ret += r
        return_.append(ret)

    print("Average Return:", np.mean(return_))


main()
