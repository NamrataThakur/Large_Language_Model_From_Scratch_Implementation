from dataclasses import dataclass

@dataclass
class GPT2_CustomConfig:
    def __init__(self, arg_config):
        
        self.config = {
            'vocab_size' :  50257,
            'embedding_dimension':arg_config['embedding_dimension'],
            'num_heads':arg_config['num_heads'],
            'context_length':arg_config['context_length'],
            'dropout':arg_config['dropout'],
            'qkv_bias':arg_config['qkv_bias'],
            'num_layers':arg_config['num_layers'],
            'rms_eps':arg_config['rms_eps'],
            'num_kv_groups': arg_config['num_kv_groups']
        }

        self.optim_config = {
            'weight_decay' : arg_config['weight_decay'],
            'beta1' : arg_config['beta1'],
            'beta2' : arg_config['beta2']
        }