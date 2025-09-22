import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, config, dtype=torch.float32):
        super().__init__()

        dim_out = config['embedding_dimension']
        heads = config["num_heads"]
        theta_base = config['theta_base']

        assert (dim_out % heads == 0), ("dim_out must be divisible by heads")

        self.dim_head = dim_out // heads

        assert (self.dim_head % 2 == 0), "Head dimension must be even"

        self.context_length = config['context_length']

        #Step 1: Postion of the token in the sequence --> Name corresponding to the RoFormer paper
        m = torch.arange(self.context_length, dtype=dtype)

        #Step 2: Build the theta parameter:
        i_range = torch.arange(0, self.dim_head, 2, dtype=dtype)[ : (self.dim_head / 2)].float()
        theta = theta_base ** (i_range / self.dim_head)
        inv_theta = 1.0 / theta

        #Step 3: Multiple m and theta to get the angles:
        angles = m.unsqueeze(1) * inv_theta.unsqueeze(0) #Shape : [context_length, dim_head/2 ]

        #Step 4: Expand the 'angles' to match head dimension:
        angles = torch.cat([angles, angles], dim=1) #Shape : [context_length, dim_head ]

        #Step 5: Get the cosine and sine values of these angles:
        self.cosine = torch.cos(angles) #Shape : [context_length, dim_head ]
        self.sine = torch.sin(angles) #Shape : [context_length, dim_head ]


    def forward(self, input):

        batch, num_heads, seq_length, dim_head = input.shape
        assert (dim_head % 2 == 0), "Head dimension must be even"

        #Truncating input tensor into 2 parts
        x1 = input[..., : dim_head // 2] #Shape : [b, heads, seq_len, dim_head/2 ] 
        x2 = input[..., dim_head // 2 :] #Shape : [b, heads, seq_len, dim_head/2 ]

        #Concatenating the parts after rotating it.
        input_rotated = torch.cat([-x2, x1], dim=-1) #Shape : [b, heads, seq_len, dim_head ]

        #Truncating the tensors to take values till 'seq_length' position
        cos = self.cosine[ : seq_length, :] #Shape : [context_length, dim_head ] (original) --> #Shape : [seq_len, dim_head ]
        sin = self.sine[ : seq_length, :]

        #Adding the batch and num_heads dimensions to match the input tensor shape:
        cos = cos.unsqueeze(0).unsqueeze(0) #Shape : [1, 1, seq_len, dim_head ]
        sin = sin.unsqueeze(0).unsqueeze(0)

        #Apply rotary transformation according to Pic:34 of the Roformer paper
        #(input * cos) Shape: [b, heads, seq_len, dim_head] * [1, 1, seq_len, dim_head ] --> [b, heads, seq_len, dim_head]
        # Same for (input_rotated * sin) --> rope_output Shape : [b, heads, seq_len, dim_head]
        rope_output = (input * cos) + (input_rotated * sin)

        # Output with lower-precision
        return rope_output.to(dtype=input.dtype)