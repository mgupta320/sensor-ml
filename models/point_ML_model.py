import torch
import torch.nn as nn


class PointModel(nn.Module):
    def __init__(self, num_hidden):
        super(PointModel, self).__init__()
        self.num_hidden = num_hidden
        if num_hidden == 0:
            self.linear = nn.Linear(6, 25)
        else:
            self.linear1 = nn.Linear(6, num_hidden)
            self.linear2 = nn.Linear(num_hidden, 25)

    def forward(self, x):
        x = x.float()
        if self.num_hidden == 0:
            x = self.linear(x)
        else:
            x = self.linear1(x)
            x = self.linear2(x)
        x = torch.sigmoid(x)
        y_predicted = nn.functional.log_softmax(x, dim=1)
        return y_predicted


