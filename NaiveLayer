import torch
import torch.nn as nn
from itertools import combinations_with_replacement

class PolynomialLayer(nn.Module):
    def __init__(self, input_dim, degree, num_neurons):
        super(PolynomialLayer, self).__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.num_outputs = len(list(combinations_with_replacement(range(input_dim), degree))) + input_dim
        self.weights = nn.Parameter(torch.randn(num_neurons, self.num_outputs))
        self.bias = nn.Parameter(torch.randn(num_neurons))

    def forward(self, x):
        # x is expected to be of size (batch_size, input_dim)
        assert x.shape[1] == self.input_dim, "Input should have {} features".format(self.input_dim)
        output = torch.empty(x.shape[0], self.num_outputs, device=x.device)
        output[:, :self.input_dim] = x
        idx = self.input_dim
        for degree in range(2, self.degree + 1):
            for combination in combinations_with_replacement(range(self.input_dim), degree):
                prod = x[:, combination[0]]
                for i in combination[1:]:
                    prod *= x[:, i]
                output[:, idx] = prod
                idx += 1
        return (output.unsqueeze(1) * self.weights.unsqueeze(0)).sum(dim=-1) + self.bias  # size: (batch_size, num_neurons)
