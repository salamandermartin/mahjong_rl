import torch.nn as nn
import torch.optim as optim

class MahjongAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MahjongAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)