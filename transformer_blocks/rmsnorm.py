import torch
import torch.nn as nn

class RMSNormalization(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.eps = config["rms_eps"]
        self.rms_bias = config['rms_bias']
        self.scale = nn.Parameter(torch.ones(config["embedding_dimension"]))
        self.shift = nn.Parameter(torch.zeros(config["embedding_dimension"])) if self.rms_bias else None


    def forward(self, input):

        input_dtype = input.dtype

        mean = input.pow(2).mean(dim=-1, keepdim = True)
        rms_norm = torch.rsqrt(mean + self.eps)
        normalized_input = input * rms_norm
        scaled_norm_input = normalized_input * self.scale

        if self.shift is not None:
            scaled_shifted_norm_input = scaled_norm_input + self.shift

        return scaled_shifted_norm_input.to(input_dtype)
