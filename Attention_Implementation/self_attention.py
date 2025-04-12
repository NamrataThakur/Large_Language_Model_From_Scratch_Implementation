import torch
import torch.nn as nn

class Self_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_key = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_value = nn.Linear(dim_in,dim_out,bias=qkv_bias)

    def forward(self,input_tensor):
        Vec_query = self.W_query(input_tensor)
        Vec_key = self.W_key(input_tensor)
        Vec_value = self.W_value(input_tensor)

        att_score = Vec_query @ Vec_key.T

        dim_k = Vec_key.shape[-1]
        weights = torch.softmax(att_score /dim_k**0.5, dim=-1)

        context = weights @ Vec_value

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

    torch.manual_seed(789)
    d_in = inputs[0].shape[-1] 
    d_out = 2 
    context_vectors = Self_Attention(d_in, d_out)
    print(context_vectors(inputs))
