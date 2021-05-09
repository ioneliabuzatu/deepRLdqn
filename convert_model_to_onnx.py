import torch
import torch.nn as nn

from utils import make_atari
from utils import wrap_deepmind
from utils import wrap_pytorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)
num_actions = env.action_space.n
state = env.reset()
dqn = DQNNetwork_atari(num_actions=num_actions).to(device)
load_checkpoint = torch.load("../checkpoint_dqn_best")
dqn.load_state_dict(load_checkpoint)
torch.onnx.export(dqn,
                  torch.tensor(state, dtype=torch.float),
                  'submission_onnx.onnx',
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True)
