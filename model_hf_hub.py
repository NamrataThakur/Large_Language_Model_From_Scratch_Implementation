import torch
from gpt_ClassificationFT.gpt2_model_config import GPT2_ModelConfig
from transformer_blocks.gpt2 import GPT2
import os
import configparser
from parameter_efficient_training.apply_lora import *
from dotenv import load_dotenv
load_dotenv()
import huggingface_hub
import huggingface_hub.hub_mixin
import json
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))
from huggingface_hub import HfApi
from transformer_blocks.gpt2_gqa import GQAGPT2
from transformer_blocks.gpt2_moe import MoEGPT2


config = configparser.ConfigParser()
config.read("config.ini")
MODEL_ROOT_FOLDER = config['PATHS']['MODEL_ROOT_FOLDER']

def prep_model_folder(model_name, base_modelName,drop_rate, context_length, classify=False,num_classes=None,
                        pretrain = False, arch_type = 'original' ):

    model_path = os.path.join(MODEL_ROOT_FOLDER,model_name)
    print('Model Path : ', model_path)

    if pretrain:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False) 
        m_config = checkpoint['config']
        gpt2_config = m_config.config

        if arch_type == 'original':
            gpt2_baseInst = GPT2(gpt2_config)
        elif arch_type == 'GQA':
            gpt2_baseInst = GQAGPT2(gpt2_config)
        else:
            gpt2_baseInst = MoEGPT2(gpt2_config)

        gpt2_baseInst.load_state_dict(checkpoint['model'] )

        print('config :: ', gpt2_config)
    else:
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

    
    
    hf_path = os.path.join(MODEL_ROOT_FOLDER,'customGPT_Pretrain')

    # 2. Save the weights in the standard PyTorch format
    torch.save(gpt2_baseInst.state_dict(), os.path.join(hf_path, "pytorch_model.bin"))

    gpt2_config['architectures'] = [gpt2_baseInst.__class__.__name__]
    gpt2_config['model_type'] = "customGPT_pretrain" if pretrain else "gpt2_finetune"

    with open(os.path.join(hf_path, "config.json"), "w") as f:
        json.dump(gpt2_config, f, indent=4)

    return f"Config and model.bin created at the {hf_path}. Ready for Huggingface Upload..!"


def push_model_hf_hub(model_name, base_modelName,drop_rate, context_length, hf_model_name, classify=False,num_classes=None,
                            pretrain = False, arch_type = 'original' ):

    #Step 1: Create a folder containing the 'config.json' and '"pytorch_model.bin"
    x = prep_model_folder(model_name, base_modelName,drop_rate, context_length, classify,num_classes,
                            pretrain, arch_type )

    print(x)
    
    #Step 2: Initialize the API
    api = HfApi()

    #Step 3: Create the path containing the newly created folder with model files:
    hf_path = os.path.join(MODEL_ROOT_FOLDER, 'customGPT_Pretrain')

    #Step 4: Define the HuggingFace repo and local path
    repo_id = f"NamrataThakur/{hf_model_name}"  #Format : <account_name>/<repo_name>
    print(f"Huggingface Repo ID : {repo_id}")

    local_folder = hf_path  # The absolute path we used earlier

    #Step 5: Create the repo (skip if it already exists)
    api.create_repo(repo_id=repo_id, exist_ok=True)

    #Step 6: Upload everything in the folder
    api.upload_folder(
        folder_path=local_folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial upload of custom GPT model"
    )

    print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")


if __name__ == '__main__':

    push_model_hf_hub(model_name="gpt2_GQA_preTrain_S_V1.pth",
                      base_modelName="gpt2_124M",
                      drop_rate=0.0,
                      context_length=1024,
                      hf_model_name="Small_Language_Model_GQA_48M_Pretrained",
                      classify=False,
                      num_classes=None,
                      pretrain=True,
                      arch_type='GQA')
