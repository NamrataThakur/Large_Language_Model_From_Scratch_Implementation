from torch.utils.data import DataLoader
import tiktoken
from .gpt2_pretrainDataset import GPTCustomDataset

#Create the custom dataloader function that will call the GPTCustomDataset class to create the dataset from the given text:
def GPTCustomDataloader(text, context_length=256, stride=128, batch_size=4, tokenizer = 'tiktoken',
                        shuffle=True, drop_last=True,num_workers=0):

    print(f'Context Length : {context_length}, Stride : {stride}, Batch_Size : {batch_size}')
    #Initializer the tokenizer
    if tokenizer == 'tiktoken':
        tokenizer = tiktoken.get_encoding("gpt2")

    #Create the dataset with the tokenizer
    custom_dataset = GPTCustomDataset(text,tokenizer, context_length, stride)

    #Create the dataloader with the dataset
    custom_dataloader = DataLoader(custom_dataset,batch_size=batch_size,
                                   shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    
    return custom_dataloader