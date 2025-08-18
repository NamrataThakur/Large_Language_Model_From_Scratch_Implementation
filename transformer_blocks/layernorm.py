import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.epsilon = 1e-5
        self.shift = nn.Parameter(torch.zeros(config['embedding_dimension']))
        self.scale = nn.Parameter(torch.ones(config['embedding_dimension']))

    def forward(self,input):
        mean = input.mean(dim=-1, keepdim = True)
        var = input.var(dim=-1, keepdim = True, unbiased=False)
        normalized_input = (input - mean)/ torch.sqrt(var + self.epsilon)
        scaled_shifted_input = normalized_input * self.scale + self.shift

        return scaled_shifted_input
