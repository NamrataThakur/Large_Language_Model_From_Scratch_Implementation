from huggingface_hub import hf_hub_download
import torch
from transformer_blocks.gpt2 import GPT2 
from transformer_blocks.gpt2_gqa import GQAGPT2
from transformer_blocks.gpt2_moe import MoEGPT2
from gpt_Pretraining.text_generation import Text_Generation

#------------------------------ TEST 1 ---------------------------------
# This triggers a "Download" event in the stats
# file_path = hf_hub_download(repo_id="NamrataThakur/Small_Language_Model_MHA_53M_Pretrained", filename="pytorch_model.bin", local_dir="model")
# print(f"Verified! Model downloaded to: {file_path}")


#------------------------------ TEST 2 ----------------------------------
# This downloads the config.json and weights, then instantiates the model
model = GQAGPT2.from_pretrained("NamrataThakur/Small_Language_Model_GQA_48M_Pretrained")

model.eval()
print("Model loaded and ready!")

#---------------------------- Checking the generation to make everything is okay ---------------------------

generation = Text_Generation(model=model, device='cpu', tokenizer_model='gpt2', 
                                          arch_type='GQA')

print('----------------------- EXAMPLE 1 --------------------------------')
start_context = "Bob and Billy went to the "
print('PROMPT : ', start_context)
response = generation.text_generation(input_text=start_context, max_new_tokens = 560, temp = 0.5, top_k=10, kv_cache=False)
print(response)

print('----------------------- EXAMPLE 2 --------------------------------')
start_context = "Once upon a time, there was a little bird named Pip. Pip wanted to fly to the moon "
print('PROMPT : ', start_context)
response = generation.text_generation(input_text=start_context, max_new_tokens = 160, temp = 0.5, top_k=10, kv_cache=False)
print(response)

print('----------------------- EXAMPLE 3 --------------------------------')
start_context = "One day, a "
print('PROMPT : ', start_context)
response = generation.text_generation(input_text=start_context, max_new_tokens = 160, temp = 0.5, top_k=10, kv_cache=False)
print(response)


