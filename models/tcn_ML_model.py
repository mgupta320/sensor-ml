import torch
import torch.nn as nn


class TCNModel(nn.Module):
    def __init__(self, kernel_size, time_steps, num_hidden, output_channels):
        super(TCNModel, self).__init__()
        self.num_hidden = num_hidden
        self.time_steps = time_steps
        num_flattened = (time_steps - (kernel_size - 1)) * output_channels
        if num_hidden == 0:
            self.model = nn.Sequential(
                nn.Conv1d(6, output_channels, kernel_size=kernel_size),
                nn.Flatten(),
                nn.Linear(num_flattened, 25),
                nn.ReLU()
            )
        else:
            self.model = nn.Sequential(
                nn.Conv1d(6, output_channels, kernel_size=kernel_size),
                nn.Flatten(),
                nn.Linear(num_flattened, num_hidden),
                nn.Linear(num_hidden, num_hidden),
                nn.Linear(num_hidden, 25),
                nn.ReLU()
            )

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        y_predicted = nn.functional.log_softmax(x, dim=1)
        return y_predicted


