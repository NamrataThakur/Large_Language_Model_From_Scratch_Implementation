import torch 
import torch.nn as nn
from .transformer import TransformerBlock
from .layernorm import LayerNormalization
from huggingface_hub import PyTorchModelHubMixin
import os
import configparser
from parameter_efficient_training.apply_lora import *
from gpt_ClassificationFT.gpt2_model_config import GPT2_ModelConfig
import requests
import time

path_config = configparser.ConfigParser()
path_config.read("config.ini")
MODEL_ROOT_FOLDER = path_config['PATHS']['MODEL_ROOT_FOLDER']


class GPT2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config['vocab_size'],config['embedding_dimension'])
        self.pos_embedding = nn.Embedding(config['context_length'],config['embedding_dimension'])
        self.token_dropout = nn.Dropout(config['dropout'])

        self.transformer_block = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['num_layers'])]
        )

        self.final_layerNorm = LayerNormalization(config)

        self.final_projection = nn.Linear(config['embedding_dimension'],config['vocab_size'],bias=config['qkv_bias'])

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
    
    @classmethod
    def from_pretrained(self, base_modelName, model_name, device='cuda', num_classes = None, classify = False, lora = False):

        try:
           
            device = torch.device(device)

            m_config = GPT2_ModelConfig()
            model_path = os.path.join(MODEL_ROOT_FOLDER,model_name)
            print('Model Path : ', model_path)

            m_config = GPT2_ModelConfig()
            gpt2_config = m_config.load_model_config(model_name=base_modelName)

            if os.path.exists(model_path):
                gpt2_baseInst = GPT2(gpt2_config)
                
                if classify:
                    print('*********************** Loading the Classification Supervised Fine-Tune Model ***********************')
                    in_features = gpt2_config['embedding_dimension']
                    out_features = num_classes

                    #Add the classification layer:
                    gpt2_baseInst.final_projection = torch.nn.Linear(in_features=in_features, out_features= out_features)

                    if lora:
                        params_orig = freeze_model(gpt2_baseInst)
                        lora_parameterization(model=gpt2_baseInst, rank = 16, alpha = 16)
                else:
                    print('*********************** Loading the Preference Fine-Tune Model ***********************')

                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                print('Checkpoint :: ', checkpoint.keys())
                gpt2_baseInst.load_state_dict(checkpoint['model'] )

                gpt2_baseInst.eval()
                gpt2_baseInst.to(device)

                return gpt2_baseInst

            else:
                print('Model weights not found in local. Loading the weights from HuggingFace Hub..!')

                if not os.path.exists(MODEL_ROOT_FOLDER):
                    os.mkdir(MODEL_ROOT_FOLDER)
                
                if classify:
                    url = "https://huggingface.co/NamrataThakur/GPT2_124M_SFT_Spam/resolve/main/gpt2_124M_SFT_Spam_v2_LoRA_noGC.pth"
                    
                else:
                    url = "https://huggingface.co/NamrataThakur/GPT2_355M_Perference-Fine-Tune_DPO/resolve/main/gpt2_355M_MaskedInstruct_PFT_v2.pth"
                    

                start_time = time.time()
                print('Downloading and Saving in the local ..!')
                
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(model_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=100000000):
                            print('Chunk Done..!')
                            f.write(chunk)

                end_time = time.time()
                execution_time_minutes = (end_time - start_time) / 60
                print(f"Model downloaded in {execution_time_minutes:.2f} minutes.")
                print('Model Downloaded and Saved in local ..!')
                self.from_pretrained(base_modelName,model_name, device,num_classes, classify, lora)


        except Exception as e:

            print("Exception while loading the model weights : ", e)









    


