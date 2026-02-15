import torch
from .lora import LORA
import torch.nn as nn
import os

class ClassLinearWithLORA(torch.nn.Module):
    def __init__(self, linear, classLoraAdapters):
        super().__init__()

        self.linear = linear
        self.classLoraAdapters = nn.ModuleDict(classLoraAdapters) #{'global_lora': LORA(), 'class_1':LORA(), 'class_2':LORA(), ......}


    #--"gates" is the output of the 'LoRAGateBlock' class
    #Format: {'global_lora': probs_0, 'class_1':probs_1, 'class_2':probs_2, ......}
    def forward(self, input, gates):

        linear_out = self.linear(input)

        for name, adapters in self.classLoraAdapters.items():
            gate_probs = gates[name]
            out_lora = linear_out + gate_probs * adapters(input)

        return out_lora
    
    
    #NEW FEAT: Saving and Loading LoRA matrices. To be used when training class-wise LoRA modules:
    def save_loras(self, save_path):
        os.makedirs(save_path, exist_ok=True )

        for name, lora_module in self.classLoraAdapters.items():
            torch.save(lora_module.save_state_dict_lora(), os.path.join(save_path, f"lora_{name}.pth"))

    
    def load_loras(self, save_path):

        for name, lora_module in self.classLoraAdapters.items():
            lora_path = os.path.join(save_path, f"lora_{name}.pth")
            if os.path.exists(lora_path):
                state_dict = torch.load(lora_path)
                lora_module.load_state_dict_lora(state_dict)