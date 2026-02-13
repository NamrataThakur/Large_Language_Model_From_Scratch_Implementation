from dataclasses import dataclass

@dataclass
class GPT2_CustomConfig:
    def __init__(self, arg_config):
        
        # Model Configs:
        self.config = {
            'vocab_size' :  50257,
            'embedding_dimension':arg_config['embedding_dimension'],
            'num_heads':arg_config['num_heads'],
            'context_length':arg_config['context_length'],
            'token_dropout':arg_config['token_dropout'],
            'attn_dropout':arg_config['attn_dropout'],
            'ffn_dropout':arg_config['ffn_dropout'],
            'qkv_bias':arg_config['qkv_bias'],
            'num_layers':arg_config['num_layers'],
            'ff_hidden_dim':arg_config['ff_hidden_dim'],

            # RMS Configs:
            'rms_eps':arg_config['rms_eps'],
            'rms_bias':arg_config['rms_bias'],

            # RoPE Config:
            'theta_base':arg_config['theta_base'],

            # Group Query Attention Config:
            'num_kv_groups': arg_config['num_kv_groups'],

            #Mixture of Experts Config:
            'num_experts': arg_config['num_experts'],
            'num_active_experts': arg_config['num_active_experts'],
            'moe_noise': arg_config['moe_noise']
        }
        
        # Optimization Configs:
        self.optim_config = {
            'weight_decay' : arg_config['weight_decay'],
            'beta1' : arg_config['beta1'],
            'beta2' : arg_config['beta2']
        }