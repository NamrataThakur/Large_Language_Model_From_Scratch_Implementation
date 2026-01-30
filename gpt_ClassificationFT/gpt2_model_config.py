class GPT2_ModelConfig:
    def __init__(self):
        
        self.base_config = {
                            'vocab_size':50257,
                            'embedding_dimension':768,
                            'num_heads':12,
                            'context_length':256, #We used a smaller context length till now to do quick training of the model to make sure the code is working properly.
                            'dropout':0.1,
                            'qkv_bias':False,
                            'num_layers':12
                            }
        
        self.config_dict =  {
                                "gpt2_124M" : {'embedding_dimension':768, 'num_heads':12, 'num_layers':12},
                                "gpt2_355M" : {'embedding_dimension':1024, 'num_heads':16, 'num_layers':24},
                                "gpt2_774M" : {'embedding_dimension':1280, 'num_heads':20, 'num_layers':36},
                                "gpt2_1558M" : {'embedding_dimension':1600, 'num_heads':25, 'num_layers':48},
                            }

    def load_model_config(self, model_name, context_length = 1024, drop_rate = 0.0, causal_mask = True):

        updated_config = self.base_config.copy()
        updated_config.update(self.config_dict[model_name])
        updated_config.update({'qkv_bias':True, 'context_length':context_length, 'dropout': drop_rate, 'causal_mask' : causal_mask})

        return updated_config
