import torch
import torch.nn as nn
from .rmsnorm import RMSNormalization
from .RoPE import RoPE
from .transformer_gqa import GQATransformerBlock

class GQAGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.token_embedding = nn.Embedding(self.config['vocab_size'],self.config['embedding_dimension'])

        #NEW FEATURE: KV_CACHE
        # Sequential takes only 1 Parameter (input tensor), we need to send cache flag too. So, changed to ModuleList
        self.transformerGQA_block = nn.ModuleList(
            [GQATransformerBlock(config=self.config) for _ in range(self.config['num_layers'])]
        )

        self.final_rmsNorm = RMSNormalization(self.config)
        self.final_projection = nn.Linear(self.config['embedding_dimension'],self.config['vocab_size'],bias=self.config['qkv_bias'])
        self.current_pos = 0
        
        self.rope_angles = RoPE(self.config) 

    def forward(self, input_tensor, cache = None):

        batch_size, context_length = input_tensor.shape

        #Get the embeddings for the list of tokens:
        token_embed = self.token_embedding(input_tensor)

        input = token_embed

        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + input.shape[1]

        else:
            pos_start = 0

        #NEW FEATURE: KV_CACHE
        #Pass the input through the transformer blocks
        for i, block in self.transformerGQA_block:

            existing_cache = cache.get(i) if cache else None 
            input, new_cache = block(input, rope=self.rope_angles, cache=existing_cache)

            if cache is not None:
                cache.update(i, new_cache)

        #Pass the output through the final layer normalization block:
        input = self.final_rmsNorm(input)

        #Pass the output through the final projection/linear layer:
        logits = self.final_projection(input)

        return logits
    
    def clear_cache(self):

        self.current_pos = 0



