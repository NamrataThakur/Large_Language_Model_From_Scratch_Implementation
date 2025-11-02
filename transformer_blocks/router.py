import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.router = nn.Linear(in_features=config['embedding_dimension'], out_features=config['num_experts'], bias= False)
        self.num_active_experts = config['num_active_experts']

        self.noise = config['moe_noise']

        if self.noise:
            self.noisy_router = nn.Linear(in_features=config['embedding_dimension'], out_features=config['num_experts'], bias=False)
            

    
    def forward(self, input):

        #Shape : b, num_tokens, dim_in
        batch, seq_len, dim_in = input.shape

        #Step 1: Get the indices and weights for the active experts:

        #Shape: (b, num_tokens, dim_in) --> (b, num_tokens, num_experts)
        router_logits = self.router(input)

        if self.noise:

            #Shape: (b, num_tokens, dim_in) --> (b, num_tokens, num_experts)
            noisy_router_logits = self.noisy_router(input)

            #Add Gaussian Noise in the logits:
            noisy_logits = torch.randn_like(noisy_router_logits) * F.softplus(noisy_router_logits)

            #Element wise add the noisy_logits with the router logits:
            router_logits = noisy_logits + router_logits


        #Shape: (b, num_tokens, num_active_experts), (b, num_tokens, num_active_experts)
        topK_logits, topK_indices = torch.topk(router_logits, k=self.num_active_experts)

        #Expand the topK_logits to make the same shape as the number of experts:
        #Shape:(b, num_tokens, num_experts)
        zero_fill = torch.full_like(router_logits,fill_value=float('-inf'))

        #Replace the value with topK_logits at topK_indices:
        #Shape: (b, num_tokens, num_experts)
        sparse_router_logits = zero_fill.scatter(dim=-1, index=topK_indices, src=topK_logits)


        #Additional way to get the sparse logits:
        # sparse_router_logits = torch.where(condition= (router_logits == topK_logits), input= topK_logits, 
        #                                    other= torch.tensor(torch.float('-inf')).to(router_logits.device) 
        #                                    )

        #Apply softmax on the sparse topK logits to get the moe weights:
        #Shape: (b, num_tokens, num_experts)
        moe_weights = F.softmax(input= sparse_router_logits, dim=-1, dtype=torch.float).to(input.dtype)

        return moe_weights, topK_indices

