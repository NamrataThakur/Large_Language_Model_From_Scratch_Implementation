from torch.utils.data import DataLoader
import tiktoken
import torch
from .gpt2_preferenceDataset import GPTCustomPreferenceDataset
from functools import partial

#Create the custom dataloader function that will call the GPTCustomPreferenceDataset class to create the dataset from the given text:
def GPTCustomPreferenceDataloader(data_file, device, pad_token = None, max_seq_length = None, batch_size=8, prompt_style = 'alpaca', tokenizer = 'tiktoken', 
                                  shuffle=True, drop_last=True,num_workers=0, seed= 123, mask_instruction = False):
    
    #Initializer the tokenizer
    if tokenizer == 'tiktoken':
        tokenizer = tiktoken.get_encoding("gpt2")

    #Get the last token id of the tokenizer selected:
    if pad_token is None:
        pad_token = tokenizer.encode('<|endoftext|>', allowed_special='all')[0]

    torch.manual_seed(seed)

    c_collate_preference = partial(custom_collate_preference, device=device, pad_token = pad_token,
                                    max_seq_length = max_seq_length, mask_instruction = mask_instruction)
    
    #Create the dataset with the tokenizer and the input file:
    dataset = GPTCustomPreferenceDataset(data = data_file, tokenizer=tokenizer, prompt_style = prompt_style)

    #Create the dataloader with the dataset
    custom_dataloader = DataLoader(dataset,batch_size=batch_size, collate_fn=c_collate_preference, 
                                   shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return custom_dataloader


def custom_collate_preference(batch, pad_token = 50256, device = 'cpu', 
                              max_seq_length = None, mask_instruction = False):
    

    batch_data = {
        'input': [],
        'correct_response': [],
        'wrong_response' : [],
        'correct_response_mask': [],
        'wrong_response_mask': []
    }

    max_length_batch = 0
    
    for key in ['correct_response', 'wrong_response']:
        key_length = max(len(row[key])+1 for row in batch)
        max_length_batch = max(max_length_batch, key_length)

    #For each entry in the current batch:
    for row in batch:
        prompt = torch.tensor(row['input'])
        batch_data['input'].append(prompt)

        for key in ['correct_response', 'wrong_response']:

            #Pad the sequence according to the max batch length across all the keys in that batch:
            input_response = row[key]
            input_response_padded = input_response + [pad_token] * (max_length_batch - len(input_response))

            #Create the mask for the padded tokens:
            mask = torch.ones(len(input_response_padded)).bool()

            #Make the mask values for all padded tokens to False:
            mask[len(input_response) : ] = False

            #Mask the input/prompt tokens:
            if mask_instruction:
                mask[ : prompt.shape[0]+2] = False

            batch_data[key+'_mask'].append(mask)
            batch_data[key].append(torch.tensor(input_response_padded))

    #Final Processing:
    for key in ['correct_response', 'wrong_response', 'correct_response_mask', 'wrong_response_mask']:

        #Stacking all tensors for a key in a batch together:
        tensor_stack = torch.stack(batch_data[key])

        #Truncate the tensor to the max_seq_length provided (Generally max_seq_length == context length of the model)
        if max_seq_length is not None:

            #Truncating the non batch dimension:
            tensor_stack = tensor_stack[ :, :max_seq_length]

            #Sending the batch tensor stack for this key to the device provided:
            batch_data[key] = tensor_stack.to(device)

    return batch_data