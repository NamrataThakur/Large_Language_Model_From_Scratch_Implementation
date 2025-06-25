from torch.utils.data import DataLoader
import tiktoken
import torch
from .gpt2_instructDataset import GPTCustomInstructionDataset
from functools import partial



#Create the custom dataloader function that will call the GPTCustomInstructionDataset class to create the dataset from the given text:
def GPTCustomInstructDataloader(data_file, device, pad_token = None, max_seq_length = None, batch_size=8, prompt_style = 'alpaca', 
                        shuffle=True, drop_last=True,num_workers=0, mask_instruction = False):
    
    #Initializer the tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    #Get the last token id of the tokenizer selected:
    if pad_token is None:
        pad_token = tokenizer.encode('<|endoftext|>', allowed_special='all')[0]

    torch.manual_seed(123)
    c_collate_instruct = partial(custom_collate_instruct, device=device, pad_token = pad_token, 
                                 max_seq_length=max_seq_length, mask_instruction=mask_instruction )

    #Create the dataset with the tokenizer and the input file:
    dataset = GPTCustomInstructionDataset(input_data = data_file, tokenizer=tokenizer, prompt_style=prompt_style)

    #Create the dataloader with the dataset
    custom_dataloader = DataLoader(dataset,batch_size=batch_size, collate_fn=c_collate_instruct, 
                                   shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return custom_dataloader


def custom_collate_instruct(batch, pad_token = 50256, ignore_index = -100, device = 'cpu', 
                            max_seq_length = None, mask_instruction = False):
    
    batch_max_length = max([len(item)+1 for instruction_length, item in batch])
    input_list, target_list = [], []

    for instruction_length, item in batch:
        tokens = item.copy()
        tokens = tokens + [pad_token]

        padded_tokens = tokens + [pad_token] * (batch_max_length - len(tokens))

        inputs = torch.tensor(padded_tokens[:-1])
        targets = torch.tensor(padded_tokens[1:])

        mask = targets == pad_token
        idx = torch.nonzero(mask).squeeze()
        if idx.numel() > 1:
            targets[idx[1:]] = ignore_index

        if mask_instruction:
            targets[:instruction_length] = ignore_index

        if max_seq_length is not None:
            inputs = inputs[:max_seq_length]
            targets = targets[:max_seq_length]

        input_list.append(inputs)
        target_list.append(targets)


    input_tensor = torch.stack(input_list).to(device)
    target_tensor = torch.stack(target_list).to(device)

    return input_tensor, target_tensor