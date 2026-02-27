import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAGateBlock(torch.nn.Module):
    def __init__(self, config, adapter_names): 
        super().__init__()

        self.adapter_names = adapter_names
        self.lora_gate = nn.Linear(in_features=config['embedding_dimension'], out_features=len(adapter_names), bias= False)


    def forward(self, input):
        
        lora_gate_logits = self.lora_gate(input)
        lora_probs = F.softmax(lora_gate_logits, dim=-1)

        probs_dict = {
            name : lora_probs[:, i:i+1] for i, name in enumerate(self.adapter_names)
        }

        return probs_dict
