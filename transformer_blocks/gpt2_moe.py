import torch
import torch.nn as nn

from .RoPE import RoPE
from .rmsnorm import RMSNormalization
from .transformer_moe import MoETransformerBlock

class MoEGPT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.token_embedding = nn.Embedding(num_embeddings=self.config['vocab_size'], embedding_dim=self.config['embedding_dimension'])

        self.transformer_block = nn.ModuleList(
                                    [MoETransformerBlock(config=self.config) for _ in range(self.config['num_layers'])]
                                )
        
        self.final_rmsNorm = RMSNormalization(config=config)
        self.final_projection = nn.Linear(in_features=self.config['embedding_dimension'], 
                                          out_features=self.config['vocab_size'], 
                                          bias=False)
        

        self.current_pos = 0

        self.rope_angles = RoPE(config=self.config)
        self.register_buffer("cosine", self.rope_angles.cosine, persistent=False)
        self.register_buffer("sine", self.rope_angles.sine, persistent=False)

    
    def forward(self, input_tensor, cache = None):

        batch, seq_len = input_tensor.shape

        #Pass the input through token embedding layer
        token_embed = self.token_embedding(input_tensor)

        input = token_embed

        if cache is not None:
            start_pos = self.current_pos
            end_pos = start_pos + seq_len
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
                                torch.ones(seq_len, seq_len, device=input_tensor.device, dtype=bool),
                                diagonal= 1
                            )

        #Explicitely broadcast the mask:
        #Shape : (seq_len, seq_len) --> (batch, dim_head, seq_len, seq_len)
        mask = mask[None, None, :, :]


        #New FEAT: KV Cache Implemented
        #Pass the tokenized input through the transformer blocks:
        for i, t_blk in enumerate(self.transformer_block):

            existing_cache = cache.get(i) if cache else None
            input, new_cache = t_blk(input, self.rope_angles, mask=mask, offset=start_pos, cache=existing_cache)

            if cache is not None:
                cache.update(i, new_cache)

        #Pass transformer output through last RMSNorm Layer
        input = self.final_rmsNorm(input)

        #Pass normalized output through last linear Layer to project to vocab space from embedding space
        logits = self.final_projection(input)

        return logits


    def clear_cache(self):

        self.current_pos = 0




        
