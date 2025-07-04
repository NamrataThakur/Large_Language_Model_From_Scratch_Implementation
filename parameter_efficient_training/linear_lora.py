import torch
from .lora import LORA

class LinearWithLORA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        #features_in, features_out = linear.weight.shape
        self.LORA = LORA(feature_in=linear.in_features, feature_out=linear.out_features, 
                         rank= rank, alpha=alpha)

    def forward(self, input):

        return self.linear(input) + self.LORA(input)