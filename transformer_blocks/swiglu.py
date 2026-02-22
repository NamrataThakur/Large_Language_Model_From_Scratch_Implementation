import torch
import torch.nn as nn
from .silu import SiLU

class SwigluExpertBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=config['embedding_dimension'], out_features=config['ff_hidden_dim'], bias=False)
        self.layer_2 = nn.Linear(in_features=config['embedding_dimension'], out_features=config['ff_hidden_dim'], bias=False)
        self.layer_3 = nn.Linear(in_features=config['ff_hidden_dim'], out_features=config['embedding_dimension'], bias=False)
        self.silu = SiLU()
        self.dropout = nn.Dropout(config['ffn_dropout'])

    def forward(self, input):
        
        output_layer_1 = self.layer_1(input)
        output_layer_2 = self.layer_2(input)

        activation_output = self.silu(output_layer_1)

        input = activation_output * output_layer_2

        input = self.dropout(input)
        
        output = self.layer_3(input)

        return output
