import torch
import math

class LORA(torch.nn.Module):
    def __init__(self, feature_in, feature_out, rank, alpha):
        super().__init__()
        
        self.rank = rank
        self.A = torch.nn.Parameter(torch.empty(feature_in, self.rank))
        
        #Distribution that A matrix will follow:
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        #torch.nn.init.normal_(self.lora_A, mean=0, std=1)

        self.B = torch.nn.Parameter(torch.zeros(self.rank, feature_out))
        self.alpha = alpha

    def forward(self, input):

        input = self.alpha * (input @ self.A @ self.B)

        return input
    

    #NEW FEAT: Saving and Loading LoRA matrices. To be used when training class-wise LoRA modules:
    def save_state_dict_lora(self):
        state_dict = {
            'A' : self.A.detach().cpu(),
            'B' : self.B.detach().cpu(),
            'rank' : self.rank,
            'alpha' : self.alpha
        }

        return state_dict
    
    def load_state_dict_lora(self, state_dict):

        self.alpha = state_dict.get("alpha", self.alpha)
        self.rank = state_dict.get("rank", self.rank)
        self.A.data.copy_(state_dict.get("A").to(self.A.device))
        self.B.data.copy_(state_dict.get("B").to(self.B.device))
        
