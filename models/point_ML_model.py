import torch
import torch.nn as nn


class PointModel(nn.Module):
    def __init__(self, num_hidden_nodes, num_hidden_layers=1, input_size=6):
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
                nn.Linear(in_features=in_size, out_features=out_size, bias=True),
                nn.ReLU()
                ))

        self.classification = nn.Sequential(
            nn.Linear(num_hidden_nodes, 5),
            nn.Softmax(dim=1)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.classification(x)
        return output


