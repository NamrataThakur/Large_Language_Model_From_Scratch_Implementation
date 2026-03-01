import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
import torch
import time
from pathlib import Path
import sys
import os

import tiktoken
import configparser
import logging
from datetime import datetime

#Load the class for model config
from gpt_ClassificationFT.gpt2_model_config import GPT2_ModelConfig

#Load the Model Architecture
from transformer_blocks.gpt2 import GPT2
from transformer_blocks.gpt2_gqa import GQAGPT2
from transformer_blocks.gpt2_moe import MoEGPT2

#Load the text generation class
from gpt_Pretraining.text_generation import Text_Generation

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def load_model_tokenizer():

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        
        customGPT_pretrain_gqa =GQAGPT2.from_pretrained("NamrataThakur/Small_Language_Model_GQA_48M_Pretrained")
        print('Text Generation - Advanced Model is loaded..!')
        customGPT_pretrain_gqa.eval()
        customGPT_pretrain_gqa.to(device)
        print(f'Text Generation - Advanced Model is sent to {device} and ready to use..!')

        customGPT_pretrain_mha =GPT2.from_pretrained("NamrataThakur/Small_Language_Model_MHA_53M_Pretrained")
        print('Text Generation Model is loaded..!')
        customGPT_pretrain_gqa.eval()
        customGPT_pretrain_gqa.to(device)
        print(f'Text Generation - Model is sent to {device} and ready to use..!')

        return customGPT_pretrain_gqa, customGPT_pretrain_mha, tokenizer

    except Exception as e:
         print("Exception while loading the model weights : ", str(e))
    


try:
        config = configparser.ConfigParser()
        config.read("config.ini")

        MODEL_ROOT_FOLDER = config['PATHS']['MODEL_ROOT_FOLDER']
        LOG_FOLDER = config["PATHS"]["LOG_FOLDER"]

        #Add the logging functionality:
        if not os.path.exists(LOG_FOLDER):
            os.mkdir(LOG_FOLDER)

        logging_path = os.path.join(LOG_FOLDER,'chainlit_run'+'_'+str(datetime.now().timestamp())+'.log')
        print(logging_path)
        
        logging.basicConfig(
            filename=logging_path,
            level=logging.INFO,
            format='[%(asctime)s.%(msecs)03d] %(message)s',
            filemode='w'
        )

        logger = logging.getLogger()

        customGPT_pretrain_gqa, customGPT_pretrain_mha, tokenizer = load_model_tokenizer()
       
        logger.info('*********************** ALL MODELS LOADED SUCCESSFULL ***********************')

        pretrain_gqa = Text_Generation(model=customGPT_pretrain_gqa, device=device, tokenizer_model='gpt2', arch_type='GQA')

        pretrain_mha = Text_Generation(model=customGPT_pretrain_mha, device=device, tokenizer_model='gpt2', arch_type='original')
    
except Exception as e:
    print("Exception while reading config file : ", e)


@cl.set_chat_profiles
async def chat_profiles():
    """Chat profile setter."""

    return [
        cl.ChatProfile(name="stories-SLM", 
                       markdown_description="This is LLM can generate short stories",
                       icon="https://picsum.photos/200"),

        cl.ChatProfile(name="stories-SLM Advanced",
                       markdown_description="This advanced LLM can generate better short stories",
                       icon="https://picsum.photos/250"),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Handler for chat start events. Sets session variables. """

    chat_prof = cl.user_session.get("chat_profile")
    #await cl.Message(content=f"Starting the session with {chat_prof}.").send()
    if chat_prof == 'stories-SLM':
        cl.user_session.set("llm", pretrain_mha)
        cl.user_session.set("m_type",chat_prof)
    else:
        cl.user_session.set("llm", pretrain_gqa)
        cl.user_session.set("m_type",chat_prof)

    
    settings = await cl.ChatSettings([
            # Select(id="LLM", label="Models to use", initial_index=0, values=['Classification Model', 'Chat Model']),
            Slider(id="Temperature", label='Temperature of the LLM', initial=0, min=0, max=2, step=0.1),
            Slider(id="max_new_tokens", label='Max new tokens', initial=10, min=10, max=500, step=10),
            Slider(id="top_k", label='Top K', initial=0, min=0, max=100, step=1),
        ]
    ).send()

@cl.on_settings_update
async def update_settings(settings):
    """Handler to manage settings updates"""

    chat_prof = cl.user_session.get("chat_profile")
    cl.user_session.set("temp", settings["Temperature"])
    cl.user_session.set("max_new_tokens", settings["max_new_tokens"])
    cl.user_session.set("top_k", settings["top_k"])
    

    if chat_prof == "stories-SLM":
        cl.user_session.set("llm", pretrain_mha)
        cl.user_session.set("m_type",chat_prof)
        
    else:
        cl.user_session.set("llm", pretrain_gqa)
        cl.user_session.set("m_type",chat_prof)
        

    logger.info(f"New settings received. LLM: {chat_prof}.")


@cl.step
@cl.on_message
async def main(message : cl.Message):
     
    input = message.content

    m_type = cl.user_session.get('m_type')
    print('Model : ', m_type)
    torch.manual_seed(123)

    temp = cl.user_session.get('temp')
    max_new_tokens = cl.user_session.get('max_new_tokens')
    
    if max_new_tokens is None:
        max_new_tokens = 160
    
    if temp is None:
        temp = 0.1

    top_k = cl.user_session.get('top_k')

    print(message.content)

    if m_type == "stories-SLM":

        output_text = pretrain_mha.text_generation(input_text = message.content, max_new_tokens=max_new_tokens, 
                                                            temp=temp, top_k= top_k, eos_id=50256, kv_cache=True)

    else:
        print('Model : ', m_type)
        output_text = pretrain_gqa.text_generation(input_text = message.content, max_new_tokens=max_new_tokens, 
                                                            temp=temp, top_k= top_k, eos_id=50256, kv_cache=True)
        
    print(output_text)
    await cl.Message(content=f'{output_text}').send()
        
