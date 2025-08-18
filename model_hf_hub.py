import torch
from gpt_ClassificationFT.gpt2_model_config import GPT2_ModelConfig
from transformer_blocks.gpt2 import GPT2
import os
import configparser
from parameter_efficient_training.apply_lora import *

config = configparser.ConfigParser()
config.read("config.ini")
MODEL_ROOT_FOLDER = config['PATHS']['MODEL_ROOT_FOLDER']

def push_model_hf_hub(model_name, base_modelName,drop_rate, context_length, hf_model_name, classify=False,num_classes=None ):

    model_path = os.path.join(MODEL_ROOT_FOLDER,model_name)
    print('Model Path : ', model_path)
    m_config = GPT2_ModelConfig()
    gpt2_config = m_config.load_model_config(model_name=base_modelName, drop_rate=drop_rate,
                                                context_length=context_length)
    gpt2_baseInst = GPT2(gpt2_config)

    if classify:
        in_features = gpt2_config['embedding_dimension']
        out_features = num_classes

        #Add the classification layer:
        gpt2_baseInst.final_projection = torch.nn.Linear(in_features=in_features, out_features= out_features)
    
        params_orig = freeze_model(gpt2_baseInst)
        lora_parameterization(model=gpt2_baseInst, rank = 16, alpha = 16)

    checkpoint = torch.load(model_path, map_location=torch.device("cuda"), weights_only=True)
    print('Checkpoint :: ', checkpoint.keys())
    gpt2_baseInst.load_state_dict(checkpoint['model'] )

    gpt2_baseInst.push_to_hub(f"NamrataThakur/{hf_model_name}", token = os.environ["HF_TOKEN"]) #
    #https://huggingface.co/NamrataThakur/GPT2_124M_SFT_Spam/raw/main/gpt2_124M_SFT_Spam_v2_LoRA_noGC.pth


if __name__ == '__main__':

    push_model_hf_hub(model_name="gpt2_124M_SFT_Spam_v2_LoRA_noGC.pth",
                      base_modelName="gpt2_124M",
                      drop_rate=0.0,
                      context_length=1024,
                      hf_model_name="gpt2-sft-spam",
                      classify=True,
                      num_classes=2)

