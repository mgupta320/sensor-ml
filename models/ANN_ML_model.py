import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class PointModel(nn.Module):
    def __init__(self, num_hidden_nodes, num_hidden_layers=1, input_size=6, num_outputs=5):
        super(PointModel, self).__init__()
        self.num_hidden_nodes = num_hidden_nodes
        self.layers = []
        self.num_hidden_layers = num_hidden_layers
        for i in range(num_hidden_layers):
            if i == 0:
                in_size = input_size
            else:
                in_size = num_hidden_nodes
            out_size = num_hidden_nodes
            self.layers.append(nn.Sequential(
                weight_norm(nn.Linear(in_features=in_size, out_features=out_size, bias=True)),
                nn.ReLU()
                ))

        self.classification = nn.Sequential(
            nn.Linear(num_hidden_nodes, num_outputs),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.classification(x)
        return output

    def reset_params(self):
        self.eval()
        for sequential in self.layers:
            for layer in sequential:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for layer in self.classification:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


