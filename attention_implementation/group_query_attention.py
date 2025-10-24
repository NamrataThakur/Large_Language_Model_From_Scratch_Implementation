import torch.nn as nn
import torch


class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_in = config['embedding_dimension']
        self.dim_out = config['embedding_dimension']
        self.num_heads = config['num_heads']
        self.context_length = config['context_length']
        self.dropout = config['dropout']

        self.kv_groups = self.num_heads if config['num_kv_groups'] == 0 else config['num_kv_groups'] #If the given KV group is 0, then GQA == MHA
        assert (self.num_heads % self.kv_groups == 0), 'Number of heads needs to be divisible by selected KV groups'
        self.group_length = self.num_heads // self.kv_groups #How many query heads will be grouped for each k-v head 

        assert (self.dim_out % self.num_heads == 0), 'dim_out must be divisible by heads '
        self.dim_head = self.dim_out // self.num_heads #32
        
        
        self.W_query = nn.Linear(self.dim_in, self.dim_out, bias=config['qkv_bias'])

        self.W_key = nn.Linear(self.dim_in, self.kv_groups * self.dim_head, bias=config['qkv_bias'])
        self.W_value = nn.Linear(self.dim_in, self.kv_groups * self.dim_head, bias=config['qkv_bias'])

        self.out_projection = nn.Linear(self.dim_out, self.dim_in, bias=config['qkv_bias'])

        #New Feat: When KV_cache is being used, we dont allocate mask at this stage:
        #self.register_buffer("mask", torch.triu(torch.ones(self.context_length, self.context_length),diagonal=1))


    def forward(self, input_tensor, rope, mask, offset = 0, cache = None):

        # Shape: b, 1, dim_in (Example: 16,269, 256)
        batch, seq_length, dim_in = input_tensor.shape

        #Pass the mask in the forward prop layer:
        self.mask = mask

        # Shape: (b, 1, dim_in) --> (b, 1, dim_out)
        Vec_query = self.W_query(input_tensor)
        Vec_key = self.W_key(input_tensor)
        Vec_value = self.W_value(input_tensor)


        #Divide the original Q,K,V projections into smaller projections (each projection for each head). Attention will be computed on each of these smaller projections.
        # Shape: (b, 1, dim_out) --> (b, 1, num_heads, dim_head)
        Vec_query = Vec_query.view(batch, seq_length, self.num_heads, self.dim_head)

        # Shape: (b, 1, dim_out) --> (b, 1, kv_groups, dim_head)
        Vec_key = Vec_key.view(batch, seq_length, self.kv_groups, self.dim_head)
        Vec_value = Vec_value.view(batch, seq_length, self.kv_groups, self.dim_head)

        #Transform or Shuffle the dimensions of the smaller projections to make the tensors situable for RoPE.
        # RoPE expects shape as (batch, num_heads, seq_len, dim_head)
        # Shape : (b, 1, num_heads, dim_head) --> (b, num_heads, 1, dim_head) { 1 here is seq_length}
        Vec_query = Vec_query.transpose(1,2)
        Vec_key = Vec_key.transpose(1,2)
        Vec_value = Vec_value.transpose(1,2)

        #Apply Rotary Transformation to get positional embeddings
        Vec_query = rope(Vec_query, offset = offset)
        Vec_key = rope(Vec_key, offset = offset)

        #KV Cache:
        if cache is not None:
            key_cache, value_cache = cache
            Vec_key = torch.cat([key_cache, Vec_key], dim=-1)
            Vec_value = torch.cat([value_cache, Vec_value], dim=-1)
            
            #Update the cache with the new key and value vector:
            new_cache = (Vec_key, Vec_value)

        else:

            #NO cache is used, so reseting the current position for RoPE computation:
            offset = 0
            #Insert cache with the new key and value vector:
            new_cache = (Vec_key, Vec_value)

        # Each group of Query vector share the same Key and Value vector, so repeat K and V vectors for each Q vector in a group:
        # Shape : (b, kv_groups, 1, dim_head) --> (b, num_heads, 1, dim_head)
        Vec_key = Vec_key.repeat_interleave(self.group_length, dim=1)
        Vec_value = Vec_value.repeat_interleave(self.group_length, dim=1)

        #Compute un-normalized self-attention score [The matrix multiplication is carried out between the 2 last dimensions (context_length, dim_head) 
                                                        # and then repeated for the individual heads ]
        attention_score = Vec_query @ Vec_key.transpose(2,3)

        # `:seq_length` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        boolean_mask = self.mask.bool()[:seq_length, :seq_length]
        # masked_scores = attention_score.masked_fill_(boolean_mask, -torch.inf)
        masked_attention_score = attention_score.masked_fill_(boolean_mask, -torch.inf)

        #Compute normalized and scaled attention weights
        dim_k = Vec_key.shape[-1]
        attention_weight = torch.softmax(masked_attention_score / dim_k**0.5, dim=-1)

        #Compute the context vectors using the sparse attention weights and value vector
        context_vector = attention_weight @ Vec_value

        #Perform re-transpositon of the context vector to make the tensor situable for rolling the last 2 dimension into 1 dimension (dim_out)
        # Shape : (b, num_heads, 1, dim_head) --> (b, 1, num_heads, dim_head)
        context_vector = context_vector.transpose(1,2)

        #Rolling the last two dimension back into 1 to make the tensor situable for final output: dim_out = num_heads * dim_head
        context_vector = context_vector.contiguous().view(batch,seq_length,self.dim_out)

        #Perform the final projection to get the FINAL Context Vector
        context_vector = self.out_projection(context_vector)

        assert context_vector.shape[-1] == self.num_heads * self.dim_head
        
        return context_vector, new_cache


        
