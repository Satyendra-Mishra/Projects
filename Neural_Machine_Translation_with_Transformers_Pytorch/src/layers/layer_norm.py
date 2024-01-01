import torch
from torch import nn

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
