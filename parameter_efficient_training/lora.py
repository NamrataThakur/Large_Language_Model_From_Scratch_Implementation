import torch
import math

class LORA(torch.nn.Module):
    def __init__(self, feature_in, feature_out, rank, alpha):
        super().__init__()
        
        self.A = torch.nn.Parameter(torch.empty(feature_in, rank))
        
        #Distribution that A matrix will follow:
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        #torch.nn.init.normal_(self.lora_A, mean=0, std=1)

        self.B = torch.nn.Parameter(torch.zeros(rank, feature_out))
        self.alpha = alpha

    def forward(self, input):

        input = self.alpha * (input @ self.A @ self.B)

        return input