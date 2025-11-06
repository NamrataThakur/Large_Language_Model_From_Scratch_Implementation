import torch
import torch.nn as nn

import os
os.pardir

from .rmsnorm import RMSNormalization
from .moe import MixtureofExperts
from attention_implementation.group_query_attention import GroupQueryAttention


class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention_block = GroupQueryAttention(config=config)
        self.rms_norm_attention = RMSNormalization(config=config)
        self.rms_norm_moe = RMSNormalization(config=config)
        self.moe_block = MixtureofExperts(config=config)

    
    def forward(self, input_tensor, rope, mask, offset = 0, cache = None):

        #Attention Block:
        residual_conn = input_tensor
        input_tensor = self.rms_norm_attention(input_tensor)

        #K_V Cache:
        input_tensor, new_cache = self.attention_block(input_tensor, rope, mask=mask, offset=offset, cache=cache)

        # Shape: (batch, seq_length, embedding_dim) + (batch, seq_length, embedding_dim) --> (batch, seq_length, embedding_dim)
        input_tensor = residual_conn + input_tensor

        #FeedForward / MoE Block:
        residual_conn = input_tensor
        input_tensor = self.rms_norm_moe(input_tensor)
        input_tensor = self.moe_block(input_tensor)

        # Shape: (batch, seq_length, embedding_dim) + (batch, seq_length, embedding_dim) --> (batch, seq_length, embedding_dim)
        input_tensor = residual_conn + input_tensor

        return input_tensor, new_cache

