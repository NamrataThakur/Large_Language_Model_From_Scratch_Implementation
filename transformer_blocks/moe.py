import torch
import torch.nn as nn
from .router import Router
from .swiglu import SwigluExpertBlock


class MixtureofExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config['num_experts']
        self.experts = nn.ModuleList(
                                        [SwigluExpertBlock(config=config) for _ in range(self.num_experts)]
                                    )
        
        self.router = Router(config=config)

    
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
            #Shape : (b, num_tokens)
            moe_mask = (topK_indices == expert_idx).any(dim=-1)

            #Shape: (b * num_tokens)
            moe_mask_flatten = moe_mask.view(-1)

            if moe_mask_flatten.any():

                #Selects only those tokens that are supposed to go to the current expert:
                #Shape: (selected_tokens, dim_in)
                #selected_tokens = moe_mask_flatten.sum() (number of True entries).
                moe_input = input_flatten[moe_mask_flatten]

                #Pass those tokens to the current expert:
                #Shape: (selected_tokens, dim_in)
                moe_output = expert(moe_input)

                # Extract router scores:
                # It contains routing weights (probabilities or gating scalars) for each token and every expert. Indexing with the same mask selects the scalar weight for each selected token for this expert.
                #Shape: (selected_tokens, 1)
                moe_scores = moe_weights_flatten[moe_mask_flatten, expert_idx].unsqueeze(1)

                # Apply router scores to get weighted outputs:
                #Shape: (selected_tokens, dim_in) * (selected_tokens, 1) --> (selected_tokens, dim_in)
                moe_weighted_output = moe_output * moe_scores

                # Update final output additively by indexing and adding
                # Note: moe_mask (2D) indexes into the original (b, num_tokens, dim_in) shape; boolean indexing will flatten it to select the same selected_tokens rows in the flattened view — shapes align.
                #Shape: moe_weighted_output.squeeze(1) --> (selected_tokens, dim_in)
                final_moe_output[moe_mask] += moe_weighted_output.squeeze(1)


        return final_moe_output


#Note : 
# Assume:

# b=2, num_tokens=3, dim_in=4, num_experts=3.

# Flatten length = 6.

# If topK_indices indicates token0 → expert0, token1 → expert2, token2 → expert0 & expert1 (top-2), then:

# Loop expert 0: picks tokens [0,2], runs expert on 2 inputs, multiplies by their weights, adds to final positions [0,2].

# Loop expert 1: picks token [2], adds contribution to final pos 2 (now sum of expert0 + expert1).

# Loop expert 2: picks token [1], etc.