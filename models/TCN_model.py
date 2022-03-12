import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math


class LeftPad(nn.Module):
    def __init__(self, padding):
        super(LeftPad, self).__init__()
        self.padding = padding

    def forward(self, x):
        return x[:, :, :-self.padding].contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, drop):
        super(ResidualBlock, self).__init__()
        layers = []
        for i in range(2):
            if i == 1:
                in_channels = out_channels
            layers.append(weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)))
            layers.append(LeftPad(padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
        if in_channels != out_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x


class TCN_Model(nn.Module):
    def __init__(self, kernel_size, time_steps, num_in_features, num_outputs, filter_channels, drop=0.2, dilation_base=2):
        super(TCN_Model, self).__init__()
        self.layers = []
        min_layers_calc = (time_steps - 1) * (dilation_base - 1) / (2 * (kernel_size - 1)) + 1
        num_layers = math.ceil(math.log(min_layers_calc, dilation_base))
        for i in range(num_layers):
            if i == 0:
                input_channels = num_in_features
            else:
                input_channels = filter_channels
            output_channels = filter_channels
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            self.layers.append(ResidualBlock(input_channels, output_channels, kernel_size, dilation, padding, drop))
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_steps * filter_channels, num_outputs, bias=True),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classification(x)
        return x

    def reset_params(self):
        for block in self.layers:
            for layer in block.block:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for layer in self.classification:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
