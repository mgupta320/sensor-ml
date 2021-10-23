import torch
import torch.nn as nn


class PointModel(nn.Module):
    def __init__(self, num_hidden):
        super(PointModel, self).__init__()
        self.num_hidden = num_hidden
        if num_hidden == 0:
            self.model = nn.Sequential(
                nn.Linear(6, 25),
                nn.ReLU()
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(6, num_hidden),
                nn.Linear(num_hidden, num_hidden),
                nn.Linear(num_hidden, 25),
                nn.ReLU()
            )

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        y_predicted = nn.functional.log_softmax(x, dim=1)
        return y_predicted


