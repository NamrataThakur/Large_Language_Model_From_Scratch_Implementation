import torch 
import torch.nn as nn
from .transformer import TransformerBlock
from .layernorm import LayerNormalization

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'],config['embedding_dimension'])
        self.pos_embedding = nn.Embedding(config['context_length'],config['embedding_dimension'])
        self.token_dropout = nn.Dropout(config['dropout'])

        self.transformer_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['num_layers'])]
        )

        self.final_layerNorm = LayerNormalization(config)

        self.final_projection = nn.Linear(config['embedding_dimension'],config['vocab_size'],bias=False)

    def forward(self,token_list):

        batch_size, context_length = token_list.shape

        #Get the embeddings for the list of tokens:
        token_embed = self.token_embedding(token_list)

        #Get the postional embeddings for the list of tokens:
        pos_embed = self.pos_embedding(torch.arange(context_length, device=token_list.device))

        #Final Embeddings:
        input = token_embed + pos_embed

        #Pass the input through the dropout layer:
        input = self.token_dropout(input)

        #Pass the dropped out input through the transformer blocks:
        input = self.transformer_block(input)

        #Pass the output through the final layer normalization block:
        input = self.final_layerNorm(input)

        #Pass the output through the final projection/linear layer:
        logits = self.final_projection(input)

        return logits
    


