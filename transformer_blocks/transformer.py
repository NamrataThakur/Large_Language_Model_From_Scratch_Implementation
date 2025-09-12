import torch
import torch.nn as nn

#import sys
#sys.path.insert(0, r'D:\LLM_Deeplearning.ai\SEBASTIAN_RASCHKA\Large_Language_Model_From_Scratch_Implementation\attention_implementation')

import os
os.pardir

from attention_implementation.causal_multi_head_attention import MultiHead_Attention
from .feedforward import FeedForwardBlock
from .gelu import GELU
from .layernorm import LayerNormalization

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention_block = MultiHead_Attention(
            dim_in = config['embedding_dimension'],
            dim_out = config['embedding_dimension'],
            context_length = config['context_length'],
            heads = config['num_heads'],
            dropout = config['dropout'],
            qkv_bias = config['qkv_bias']
        )
        self.layer_norm_attention = LayerNormalization(config)
        self.layer_norm_feedforward = LayerNormalization(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.feedForward = FeedForwardBlock(config)

    def forward(self,input, cache = False):

        #Attention Block Computation:
        residual_conn = input
        input = self.layer_norm_attention(input)

        #NEW FEATURE: KV_CACHE
        input = self.attention_block(input, cache = cache)
        input = self.dropout(input)

        input = input + residual_conn

        #Feed Forward Block Computation:
        residual_conn = input
        input = self.layer_norm_feedforward(input)
        input = self.feedForward(input)
        input = self.dropout(input)

        input = input + residual_conn


        return input

        
    
if __name__ == '__main__':
    x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    GPT2_CONFIG = {
    'vocab_size':50257,
    'embedding_dimension':768,
    'num_heads':12,
    'context_length':1024,
    'dropout':0.1,
    'qkv_bias':False,
    'num_layers':12,
    }
    block = TransformerBlock(GPT2_CONFIG)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device Available: ', device)
