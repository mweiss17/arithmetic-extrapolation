import math
import torch
import torch.nn as nn
from torch.nn import Parameter

class LayerNorm(nn.Module):
    """
    Layer Normalization based on Ba & al.:
    'Layer Normalization'
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, input_size: int, learnable: bool = True, epsilon: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.input_size = input_size
        self.learnable = learnable
        self.alpha = torch.empty(1, input_size).fill_(0)
        self.beta = torch.empty(1, input_size).fill_(0)
        self.epsilon = epsilon
        # Wrap as parameters if necessary
        if learnable:
            self.alpha = Parameter(self.alpha)
            self.beta = Parameter(self.beta)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        x = x.view(x.size(0), -1)
        x = (x - torch.mean(x, 1).unsqueeze(1)) / torch.sqrt(torch.var(x, 1).unsqueeze(1) + self.epsilon)
        if self.learnable:
            x = self.alpha.expand_as(x) * x + self.beta.expand_as(x)
        return x.view(size)
