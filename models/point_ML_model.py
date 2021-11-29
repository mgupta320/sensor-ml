import torch
import torch.nn as nn


class PointModel(nn.Module):
    def __init__(self, num_hidden):
        super(PointModel, self).__init__()
        self.num_hidden = num_hidden
        self.model = nn.Sequential(
            nn.Linear(6, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 25)
        )

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        return x


