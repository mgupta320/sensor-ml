import torch
import torch.nn as nn


class TCNModel(nn.Module):
    def __init__(self, kernel_size, num_hidden):
        super(TCNModel, self).__init__()
        self.n_hidden = num_hidden
        if num_hidden == 0:
            self.conv = nn.Conv1d(6, 25, kernel_size=kernel_size)
        else:
            self.conv = nn.Conv1d(6, num_hidden, kernel_size=kernel_size)
            self.linear = nn.Linear(num_hidden, 25)

    def forward(self, x):
        x = self.conv(x)
        if self.num_hidden != 0:
            x = self.linear(x)
        x = torch.sigmoid(x)
        y_pred = nn.functional.log_softmax(x, dim=1)
        return y_pred


