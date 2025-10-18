import torch
import torch.nn as nn
from .router import Router
from .swiglu import SwigluExpertBlock


class MixtureofExperts(nn.Module):
    def __init__(self, config, noise=False):
        super().__init__()

        self.num_experts = config['num_experts']
        self.experts = nn.ModuleList([SwigluExpertBlock(config) for _ in self.num_experts])
        
        self.router = Router(config=config, noise=noise)

    
    def forward(self, input):

        #Tensor to contain final MoE output:
        #Shape: (b, num_tokens, dim_in)
        final_moe_output = torch.zeros_like(input=input)
        
        #Shape: (b, num_tokens, num_experts)
        moe_weights, topK_indices = self.router(input)

        #Flatten out input: Flattening all dimension except the last
        #Shape: (b, num_tokens, dim_in) --> (b * num_tokens, dim_in)
        input_flatten = input.view(-1, input.size(-1))

        #Flatten out the moe_weights:
        #Shape: (b, num_tokens, num_experts) --> (b * num_tokens, num_experts)
        moe_weights_flatten = moe_weights.view(-1, moe_weights.size(-1))


        for expert_idx, expert in enumerate(self.experts):

            #Get the mask at indices matching the expert index:
            #This tells which input tokens goes to the current expert
            moe_mask = (topK_indices == expert_idx).any(dim=-1)

            moe_mask_flatten = moe_mask.view(-1)

            if moe_mask_flatten.any():

                #Selects only those tokens that are supposed to go to the current expert:
                moe_input = input_flatten[moe_mask_flatten]

                #Pass those tokens to the current expert:
                moe_output = expert(moe_input)

                moe_scores = moe_weights_flatten[moe_mask, expert_idx].unsqueeze(1)
                moe_weighted_output = moe_output * moe_scores

                final_moe_output[moe_mask] += moe_weighted_output.unsqueeze(1)

        return final_moe_output
