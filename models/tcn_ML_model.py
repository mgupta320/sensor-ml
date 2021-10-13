import torch
import torch.nn as nn


class TCNModel(nn.Module):
    def __init__(self, kernel_size, time_steps, num_hidden, output_channels):
        super(TCNModel, self).__init__()
        self.num_hidden = num_hidden
        self.time_steps = time_steps
        self.conv = nn.Conv1d(6, output_channels, kernel_size=kernel_size)
        num_flattened = (time_steps - (kernel_size - 1)) * output_channels
        if num_hidden == 0:
            self.linear = nn.Linear(num_flattened, 25)
        else:
            self.linear1 = nn.Linear(num_flattened, num_hidden)
            self.linear2 = nn.Linear(num_hidden, 25)

    def forward(self, x):
        x = x.float()
        x = self.conv(x)
        x = torch.flatten(x, 1)
        if self.num_hidden == 0:
            x = self.linear(x)
        else:
            x = self.linear1(x)
            x = self.linear2(x)
        x = torch.sigmoid(x)
        y_predicted = nn.functional.log_softmax(x, dim=1)
        return y_predicted


