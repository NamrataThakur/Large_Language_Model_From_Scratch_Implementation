import zipfile
import os 
from pathlib import Path
import configparser
import tiktoken 
from transformer_blocks.gpt2 import GPT2 
from transformer_blocks.gpt2_gqa import GQAGPT2
from transformer_blocks.gpt2_moe import MoEGPT2
from gpt_Pretraining.text_generation import Text_Generation

# config = configparser.ConfigParser()
# config.read("config.ini")

# DATA_FOLDER = config["PATHS"]["DATA_FOLDER"]
# data_path = os.path.join(DATA_FOLDER,'sms_spam_collection.zip')

#Initializer the tokenizer
# tokenizer = tiktoken.get_encoding("gpt2")

# #Get the last token id of the tokenizer selected:
# pad_token = tokenizer.encode('<|endoftext|>', allowed_special='all')[0]
# print(pad_token)
model = GQAGPT2.from_pretrained("NamrataThakur/Small_Language_Model_GQA_48M_Pretrained")

model.eval()
print("Model loaded and ready!")

#---------------------------- Checking the generation to make everything is okay ---------------------------

generation = Text_Generation(model=model, device='cpu', tokenizer_model='gpt2', 
                                          arch_type='GQA')

print('----------------------- EXAMPLE 1 --------------------------------')
start_context = "Bob and Billy went to the "
print('PROMPT : ', start_context)
response = generation.text_generation(input_text=start_context, max_new_tokens = 160, temp = 0.5, top_k=10, kv_cache=False)
print(response)
