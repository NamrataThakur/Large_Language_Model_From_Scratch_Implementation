import torch.nn as nn
import torch

class Causal_Attention(nn.Module):
    def __init__(self, dim_in, dim_out,context_length,
                 dropout,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_key = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_value = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,input_tensor):

        batch, token_length, dim_in = input_tensor.shape
        Vec_query = self.W_query(input_tensor)
        Vec_key = self.W_key(input_tensor)
        Vec_value = self.W_value(input_tensor)

        att_score = Vec_query @ Vec_key.transpose(1,2)

        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        masked_scores = att_score.masked_fill(self.mask.bool()[:token_length, :token_length], -torch.inf)

        dim_k = Vec_key.shape[-1]
        weights = torch.softmax(masked_scores /dim_k**0.5, dim=-1)

        sparse_maskedWeights = self.dropout(weights)
        context = sparse_maskedWeights @ Vec_value

        return context



if __name__ == '__main__':
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
    )

    batch = torch.stack((inputs, inputs), dim=0)
    
    d_in = batch[0].shape[-1] 
    d_out = 5 
    context = batch.shape[1]

    torch.manual_seed(789)
    context_vectors = Causal_Attention(d_in, d_out,context,dropout=0.0,qkv_bias=False)
    print(context_vectors(batch))
    





