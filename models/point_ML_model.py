import torch
import torch.nn as nn


class PointModel(nn.Module):
    def __init__(self, n_hidden):
        super(PointModel, self).__init__()
        self.n_hidden = n_hidden
        if n_hidden == 0:
            self.linear = nn.Linear(6, 25)
        else:
            self.linear1 = nn.Linear(6, n_hidden)
            self.linear2 = nn.Linear(n_hidden, 25)

    def forward(self, x):
        if self.n_hidden == 0:
            x = self.linear(x)
        else:
            x = self.linear1(x)
            x = self.linear2(x)
        x = torch.sigmoid(x)
        y_predicted = nn.functional.log_softmax(x, dim=1)
        return y_predicted


