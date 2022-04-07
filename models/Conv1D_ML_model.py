import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Conv1D_Model(nn.Module):
    def __init__(self, kernel_size, time_steps, output_channels=6, input_size=6, num_conv_layers=1, num_outputs=5):
        super(Conv1D_Model, self).__init__()
        num_flattened = time_steps * output_channels
        num_hidden = num_flattened + 1
        self.num_hidden = num_hidden
        self.time_steps = time_steps
        self.layers = []
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_flattened, num_outputs, bias=True),
            nn.Softmax(dim=-1)
        )
        for i in range(num_conv_layers):
            if i == 0:
                in_channels = input_size
            else:
                in_channels = output_channels
            self.layers.append(nn.Sequential(
                weight_norm(nn.Conv1d(in_channels, output_channels, kernel_size=kernel_size, padding='same')),
                nn.ReLU(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classification(x)
        return x

    def reset_params(self):
        for sequential in self.layers:
            for layer in sequential:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for layer in self.classification:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


