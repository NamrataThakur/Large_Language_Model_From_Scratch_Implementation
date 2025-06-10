import torch
import numpy as np

def assign_weights(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def gpt2_loadedWeights(gpt2, params):

    #Load the weights for the token embedding and postitional embedding layers:
    gpt2.token_embedding.weight = assign_weights(gpt2.token_embedding.weight, params['wte'])
    gpt2.pos_embedding.weight = assign_weights(gpt2.pos_embedding.weight, params['wpe'])

    #Load the weights for the transformer blocks:
    for block in range(len(params['blocks'])):

        #-------------------Split an array into multiple sub-arrays as views : Splitting the weights matrix -----------------------
        q_weight, k_weight, v_weight = np.split(params['blocks'][block]['attn']['c_attn']['w'],3, axis = -1)

        #Load the weights into the Query matrix of each blocks:
        gpt2.transformer_block[block].attention_block.W_query.weight = assign_weights(gpt2.transformer_block[block].attention_block.W_query.weight, 
                                                                                      q_weight.T)
        
        #Load the weights into the Key matrix of each blocks:
        gpt2.transformer_block[block].attention_block.W_key.weight = assign_weights(gpt2.transformer_block[block].attention_block.W_key.weight, 
                                                                                      k_weight.T)
        
        #Load the weights into the Value matrix of each blocks:
        gpt2.transformer_block[block].attention_block.W_value.weight = assign_weights(gpt2.transformer_block[block].attention_block.W_value.weight, 
                                                                                      v_weight.T)
        

        #----------------- Splitting the bias matrix --------------------------------------------------------------------------------
        q_bias, k_bias, v_bias = np.split(params['blocks'][block]['attn']['c_attn']['b'],3, axis = -1)

        #Load the bias into the Query matrix of each blocks:
        gpt2.transformer_block[block].attention_block.W_query.bias = assign_weights(gpt2.transformer_block[block].attention_block.W_query.bias, 
                                                                                      q_bias)
        
        #Load the bias into the Key matrix of each blocks:
        gpt2.transformer_block[block].attention_block.W_key.bias = assign_weights(gpt2.transformer_block[block].attention_block.W_key.bias, 
                                                                                      k_bias)
        
        #Load the bias into the Value matrix of each blocks:
        gpt2.transformer_block[block].attention_block.W_value.bias = assign_weights(gpt2.transformer_block[block].attention_block.W_value.bias, 
                                                                                      v_bias)
        


        #----------------- Loading the projection layer weights present inside each attention block ---------------------------------
        #Load the weights into the Output Projection layer of each blocks:
        gpt2.transformer_block[block].attention_block.out_projection.weight = assign_weights(gpt2.transformer_block[block].attention_block.out_projection.weight, 
                                                                                      params['blocks'][block]['attn']['c_proj']['w'].T)
        
        #Load the bias into the Output Projection layer of each blocks:
        gpt2.transformer_block[block].attention_block.out_projection.bias = assign_weights(gpt2.transformer_block[block].attention_block.out_projection.bias, 
                                                                                      params['blocks'][block]['attn']['c_proj']['b'])
        

        #----------------- Loading the FeedForward Block weights and bias present inside each Transformer block ---------------------------------
        #Load the weights into the 1st FC layer:
        gpt2.transformer_block[block].feedForward.block[0].weight = assign_weights(gpt2.transformer_block[block].feedForward.block[0].weight, 
                                                                                      params['blocks'][block]['mlp']['c_fc']['w'].T)
        
        #Load the bias into the 1st FC layer:
        gpt2.transformer_block[block].feedForward.block[0].bias = assign_weights(gpt2.transformer_block[block].feedForward.block[0].bias, 
                                                                                      params['blocks'][block]['mlp']['c_fc']['b'])
        
        #Load the weights into the 2st FC layer:
        gpt2.transformer_block[block].feedForward.block[2].weight = assign_weights(gpt2.transformer_block[block].feedForward.block[2].weight, 
                                                                                      params['blocks'][block]['mlp']['c_proj']['w'].T)
        
        #Load the bias into the 2st FC layer:
        gpt2.transformer_block[block].feedForward.block[2].bias = assign_weights(gpt2.transformer_block[block].feedForward.block[2].bias, 
                                                                                      params['blocks'][block]['mlp']['c_proj']['b'])
        

        #----------------- Loading the LayerNormalization Block SCALE and SHIFT values present inside each Transformer block ---------------------------------
        #Load the weights into the 1st LayerNorm layer:
        gpt2.transformer_block[block].layer_norm_attention.scale = assign_weights(gpt2.transformer_block[block].layer_norm_attention.scale, 
                                                                                      params['blocks'][block]['ln_1']['g'])
        
        #Load the bias into the 1st LayerNorm layer:
        gpt2.transformer_block[block].layer_norm_attention.shift = assign_weights(gpt2.transformer_block[block].layer_norm_attention.shift, 
                                                                                      params['blocks'][block]['ln_1']['b'])
        
        #Load the weights into the 2nd LayerNorm layer:
        gpt2.transformer_block[block].layer_norm_feedforward.scale = assign_weights(gpt2.transformer_block[block].layer_norm_feedforward.scale, 
                                                                                      params['blocks'][block]['ln_2']['g'])
        
        #Load the bias into the 2nd LayerNorm layer:
        gpt2.transformer_block[block].layer_norm_feedforward.shift = assign_weights(gpt2.transformer_block[block].layer_norm_feedforward.shift, 
                                                                                      params['blocks'][block]['ln_2']['b'])
        

    #----------------- Loading the FINAL LayerNormalization Block SCALE and SHIFT values  ---------------------------------------------------------------------
    gpt2.final_layerNorm.scale = assign_weights(gpt2.final_layerNorm.scale, params['g'])
    gpt2.final_layerNorm.shift = assign_weights(gpt2.final_layerNorm.scale, params['b'])

    #----------------- Loading the FINAL Projection Block weights  ---------------------------------------------------------------------
    gpt2.final_projection.weight = assign_weights(gpt2.final_projection.weight, params['wte'])
