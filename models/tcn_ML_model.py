import torch
import torch.nn as nn


class TCNModel(nn.Module):
    def __init__(self, kernel_size, time_steps, num_hidden, output_channels, input_size=6):
        super(TCNModel, self).__init__()
        self.num_hidden = num_hidden
        self.time_steps = time_steps
        num_flattened = (time_steps - (kernel_size - 1)) * output_channels
        self.model = nn.Sequential(
            nn.Conv1d(input_size, output_channels, kernel_size=kernel_size),
            nn.Flatten(),
            nn.Linear(num_flattened, num_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(num_hidden, 5)
        )
        self.classification = nn.Sequential(
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.model(x)
        output = self.classification(x)
        return output


