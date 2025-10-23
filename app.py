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

#Load the text generation class
from gpt_Pretraining.text_generation import Text_Generation

#Dataloaders for loading function for input formatting
from dataloader.Instruction_finetuning.gpt2_instructDataFormat import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_tokenizer():

    try:
        tokenizer = tiktoken.get_encoding("gpt2")

        gpt2_sft_Inst = GPT2.from_pretrained(base_modelName='gpt2_124M', model_name='gpt2_124M_SFT_Spam_v2_LoRA_noGC.pth',
                                            device='cuda', num_classes=2, classify=True, lora=True)
        
        gpt2_pft_Inst = GPT2.from_pretrained(base_modelName='gpt2_355M', model_name='gpt2_355M_MaskedInstruct_PFT_v2.pth',
                                            device='cuda', classify=False, lora=False)
        
        return gpt2_sft_Inst, gpt2_pft_Inst, tokenizer

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

        gpt2_sft_Inst, gpt2_pft_Inst, tokenizer = load_model_tokenizer()
        logger.info('*********************** ALL MODELS LOADED SUCCESSFULL ***********************')

        sft_classify = Text_Generation(model=gpt2_sft_Inst, device=device, tokenizer_model='gpt2')

        pft_generate = Text_Generation(model=gpt2_pft_Inst, device=device, tokenizer_model='gpt2')
    
except Exception as e:
    print("Exception while reading config file : ", e)


@cl.set_chat_profiles
async def chat_profiles():
    """Chat profile setter."""

    return [
        cl.ChatProfile(name="Classification Model", 
                       markdown_description="This LLM can classify spam vs ham messages",
                       icon="https://picsum.photos/200"),

        cl.ChatProfile(name="Chat Model",
                       markdown_description="This LLM is a personal assistant",
                       icon="https://picsum.photos/250"),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Handler for chat start events. Sets session variables. """

    chat_prof = cl.user_session.get("chat_profile")
    #await cl.Message(content=f"Starting the session with {chat_prof}.").send()
    if chat_prof == 'Chat Model':
        cl.user_session.set("llm", gpt2_pft_Inst)
        cl.user_session.set("m_type",chat_prof)
    else:
        cl.user_session.set("llm", gpt2_sft_Inst)
        cl.user_session.set("m_type",chat_prof)

    if chat_prof == 'Chat Model':
        settings = await cl.ChatSettings([
            # Select(id="LLM", label="Models to use", initial_index=0, values=['Classification Model', 'Chat Model']),
            Slider(id="Temperature", label='Temperature of the LLM', initial=0, min=0, max=2, step=0.1),
            Slider(id="max_new_tokens", label='Max new tokens', initial=1, min=1, max=1024, step=1),
            Slider(id="top_k", label='Top K', initial=0, min=0, max=100, step=1),

        ]
    ).send()

@cl.on_settings_update
async def update_settings(settings):
    """Handler to manage settings updates"""

    chat_prof = cl.user_session.get("chat_profile")

    if chat_prof == "Chat Model":
        cl.user_session.set("temp", settings["Temperature"])
        cl.user_session.set("max_new_tokens", settings["max_new_tokens"])
        cl.user_session.set("top_k", settings["top_k"])
        cl.user_session.set("llm", gpt2_pft_Inst)
        cl.user_session.set("m_type",chat_prof)
        
    else:
        cl.user_session.set("llm", gpt2_sft_Inst)
        cl.user_session.set("m_type",chat_prof)
        

    logger.info(f"New settings received. LLM: {chat_prof}.")


@cl.step
@cl.on_message
async def main(message : cl.Message):
     
    input = message.content

    m_type = cl.user_session.get('m_type')

    if m_type == "Chat Model":

        print('Model : ', m_type)
        torch.manual_seed(123)

        prompt = f"""Below is an instruction that describes a task. Write a response
        that appropriately completes the request.

        ### Instruction:
        {message.content}"""

        input_dict = {}
        input_dict['instruction'] = message.content
        input_dict['input'] = ''
        input_dict['output'] = ''
        _, input_text = format_input_response(input_dict, inference=True)

        print('----------')
        logger.info(input_text)
        print('----------')
        temp = cl.user_session.get('temp')
        max_new_tokens = cl.user_session.get('max_new_tokens')
        
        if max_new_tokens is None:
            max_new_tokens = 100
        
        if temp is None:
            temp = 0.0

        top_k = cl.user_session.get('top_k')

        output_text = pft_generate.text_generation(input_text = input_text, max_new_tokens=max_new_tokens, 
                                                            temp=temp, top_k= top_k, eos_id=50256, kv_cache=True)
        
        response = (output_text[len(input_text)-1:]).replace("### Response:", " ").replace('Response:', '').strip()

        print(response)

        await cl.Message(content=f'{response}').send()


    else:
        print('Model : ', m_type)
        pred_label = sft_classify.classify_text(input, max_length=120)
        text_label = 'spam' if pred_label == 1 else 'ham'

        print('Response Label : ', text_label)
        await cl.Message(content=f'{text_label}').send()
