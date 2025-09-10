import torch
import tiktoken
from functools import partial
from .customGPT2_pretrainDataset import GPTCustomPretrainDataset
from torch.utils.data import DataLoader

#Create the custom dataloader function that will call the GPTCustomPretrainDataset class to create the dataset from the given text:
def GPTCustomPretrainDataloader(data, tokenizer='tiktoken', pad_token=None, device = 'cpu', max_seq_length = None, 
                                batch_size=8,shuffle=True, drop_last=True,num_workers=0):

    if tokenizer == 'tiktoken':
        tokenizer = tiktoken.get_encoding('gpt2')

    #Get the last token id of the tokenizer selected:
    if pad_token is None:
        pad_token = tokenizer.encode('<|endoftext|>', allowed_special='all')[0]

    torch.manual_seed(123)

    c_collate_pretrain = partial(custom_collate_pretrain, device=device, max_seq_length=max_seq_length , pad_token = pad_token)

    dataset = GPTCustomPretrainDataset(tokenizer=tokenizer, input_data=data)

    total_tokens = dataset.get_total_tokens()

    #Create the dataloader with the dataset
    custom_dataloader = DataLoader(dataset,batch_size=batch_size, collate_fn=c_collate_pretrain, 
                                   shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return custom_dataloader, total_tokens


# Custom collate function 
def custom_collate_pretrain(batch, pad_token = 50256, device = 'cpu', max_seq_length = None):

    input_list, target_list = [], []
    batch_max_length = max([len(item)+1 for item in batch])

    for row in batch:

        tokens = row.copy()
        tokens = tokens + [pad_token]

        padded_tokens = tokens + [pad_token] * (batch_max_length - len(tokens))

        inputs = torch.tensor(padded_tokens[:-1])
        targets = torch.tensor(padded_tokens[1:])

        if max_seq_length is not None:
            inputs = inputs[:max_seq_length]
            targets = targets[:max_seq_length]


        input_list.append(inputs)
        target_list.append(targets)

    inputs_tensor = torch.stack(input_list).to(device)
    targets_tensor = torch.stack(target_list).to(device)

    return inputs_tensor, targets_tensor