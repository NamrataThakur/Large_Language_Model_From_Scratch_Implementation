import torch.nn as nn
import torch

class MultiHead_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length,heads,
                  dropout, qkv_bias=False):
        super().__init__()

        assert (dim_out % heads) == 0, \
            "d_out must be divisible by heads"
        
        self.dim_head = dim_out // heads
        self.dim_out = dim_out
        self.heads = heads

        self.W_query = nn.Linear(dim_in,dim_out,bias=qkv_bias) #Shape : 3x4
        self.W_key = nn.Linear(dim_in,dim_out,bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(dim_out, dim_out)  # Linear layer to combine head outputs
        
        #New Feat: When KV_cache is being used, we dont allocate mask at this stage:
        #self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length), diagonal=1))

        #NEW FEATURE: KV_CACHE
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)

    
    def forward(self,input_tensor, mask, cache = False): # Flag to use KV cache during inference or not

        batch, num_tokens, dim_in = input_tensor.shape

        #Pass the mask in the forward prop layer:
        self.mask = mask

        Vec_query = self.W_query(input_tensor) # Shape: (b, context_length, dim_in)
        Vec_key = self.W_key(input_tensor)
        Vec_value = self.W_value(input_tensor)

        #Divide the original Q,K,V projections into smaller projections (each projection for each head). Attention will be computed on each of these smaller projections.
        Vec_query = Vec_query.view(batch,num_tokens, self.heads, self.dim_head) #Shape: b,6,2,2
        Vec_key_new = Vec_key.view(batch,num_tokens, self.heads, self.dim_head)
        Vec_value_new = Vec_value.view(batch,num_tokens, self.heads, self.dim_head)

        #NEW FEATURE: KV_CACHE
        if cache:
            if self.k_cache is not None :               
                #dim=1 → sequence dimension (what grows with new tokens):
                self.k_cache = torch.cat([self.k_cache, Vec_key_new], dim=1)
                self.v_cache = torch.cat([self.v_cache, Vec_value_new], dim=1)

            else:
                self.k_cache = Vec_key_new
                self.v_cache = Vec_key_new

            Vec_key = self.k_cache
            Vec_value = self.v_cache
            #print(f'key_cache.shape: {self.k_cache.shape}, Vec_key.shape: {Vec_key.shape}, mask.shape : {self.mask.shape} ')
        
        else:
            Vec_key = Vec_key_new
            Vec_value = Vec_value_new

            #print(f'Vec_key.shape: {Vec_key.shape}, mask.shape : {self.mask.shape} ')


        #Transform or Shuffle the dimensions of the smaller projections to make the tensors situable for attention.
        Vec_query = Vec_query.transpose(1,2) #Shape: b, heads, context_length, dim_head #Shape: b,2,6,2
        Vec_key = Vec_key.transpose(1,2)
        Vec_value = Vec_value.transpose(1,2)

        #Compute un-normalized self-attention score [The matrix multiplication is carried out between the 2 last dimensions (context_length, dim_head) 
                                                        # and then repeated for the individual heads ]
        attention_score = Vec_query @ Vec_key.transpose(2,3)

        # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        boolean_mask = self.mask.bool()[:num_tokens, :num_tokens]
        # masked_scores = attention_score.masked_fill_(boolean_mask, -torch.inf)
        masked_attention_score = attention_score.masked_fill_(boolean_mask, -torch.inf)

        #Compute normalized and scaled attention weights
        dim_k = Vec_key.shape[-1]
        attention_weight = torch.softmax(masked_attention_score / dim_k**0.5, dim=-1)

        attention_weight = self.dropout(attention_weight)

        #Compute the context vectors using the sparse attention weights and value vector
        context_vector = attention_weight @ Vec_value

        #Perform re-transpositon of the context vector to make the tensor situable for rolling the last 2 dimension into 1 dimension (dim_out)
        context_vector = context_vector.transpose(1,2)

        #Rolling the last two dimension back into 1 to make the tensor situable for final output: dim_out = heads * dim_head
        context_vector = context_vector.contiguous().view(batch,num_tokens,self.dim_out)
        #context_vector = context_vector.reshape(batch, num_tokens, self.dim_out)

        #Perform the final projection to get the FINAL Context Vector
        context_vector = self.out_projection(context_vector)

        assert context_vector.shape[-1] == self.heads * self.dim_head
        
        return context_vector
    
    
    #NEW FEATURE: KV_CACHE
    def clear_cache(self):

        self.k_cache = None
        self.v_cache = None




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

    batch_size, context_length, dim_in = batch.shape
    dim_out = 4

    torch.manual_seed(789)
    context_vectors = MultiHead_Attention(dim_in, dim_out,context_length,dropout=0.0,heads=2, qkv_bias=False)
    print(context_vectors(batch))
    






        




