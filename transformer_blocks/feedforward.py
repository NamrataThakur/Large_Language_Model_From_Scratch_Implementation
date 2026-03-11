import torch
import torch.nn as nn
from .gelu import GELU

class FeedForwardBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(config['embedding_dimension'], 4*config['embedding_dimension']),
            GELU(),
            nn.Linear( 4*config['embedding_dimension'], config['embedding_dimension'])
        )
    
    def forward(self,input, lora_gates = None):
        return self.block(input, gates = lora_gates) if lora_gates is not None else self.block(input) 
    

