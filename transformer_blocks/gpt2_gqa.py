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
        self.token_dropout = nn.Dropout(self.config['token_dropout'])

        #NEW FEATURE: KV_CACHE
        # Sequential takes only 1 Parameter (input tensor), we need to send cache flag too. So, changed to ModuleList
        self.transformerGQA_block = nn.ModuleList(
            [GQATransformerBlock(config=self.config) for _ in range(self.config['num_layers'])]
        )

        self.final_rmsNorm = RMSNormalization(self.config)
        self.final_projection = nn.Linear(self.config['embedding_dimension'],self.config['vocab_size'],bias=self.config['qkv_bias'])
        self.current_pos = 0
        
        self.rope_angles = RoPE(self.config) 
        self.final_projection.weight = self.token_embedding.weight

    
    #cache --> object of KVCache class:
    def forward(self, input_tensor, cache = None):

        batch_size, context_length = input_tensor.shape

        #Get the embeddings for the list of tokens:
        token_embed = self.token_embedding(input_tensor)

        input = token_embed

        if cache is not None:
            start_pos = self.current_pos
            end_pos = start_pos + input.shape[1]
            self.current_pos = end_pos

            #Since KV Cache is being used, so we have to create mask only for the new tokens, to compute attention scores only for new tokens.
            #For old tokens, the masked attention scores are extracted from the cache:
            mask = torch.triu(
                                torch.ones(end_pos,end_pos, device=input_tensor.device, dtype=torch.bool),
                                diagonal=1
                            )[start_pos : end_pos, :end_pos]

        else:
            start_pos = 0

            #KV_cache is not being used, so position embedding and mask needs to be created for the entire sequence:
            mask = torch.triu(
                                torch.ones(context_length, context_length, device=input_tensor.device, dtype=torch.bool),
                                diagonal= 1
                            )
            
        #Explicitely broadcast the mask:
        #Shape : (context_length, context_length) --> (batch, dim_head, context_length, context_length)
        mask = mask[None, None, :, :]

        #Pass the input through the dropout layer:
        input = self.token_dropout(input)

        #NEW FEATURE: KV_CACHE
        #Pass the input through the transformer blocks
        for i, block in enumerate(self.transformerGQA_block):

            existing_cache = cache.get(i) if cache else None 
            input, new_cache = block(input, rope=self.rope_angles, mask=mask, offset=start_pos, cache=existing_cache)

            if cache is not None:
                cache.update(i, new_cache)

        #Pass the output through the final layer normalization block:
        input = self.final_rmsNorm(input)

        #Pass the output through the final projection/linear layer:
        logits = self.final_projection(input)

        return logits
    
    def clear_cache(self):

        self.current_pos = 0



