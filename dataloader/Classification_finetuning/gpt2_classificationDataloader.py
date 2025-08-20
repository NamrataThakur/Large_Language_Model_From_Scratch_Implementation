from torch.utils.data import DataLoader
import tiktoken
from .gpt2_classficationDataset import GPTCustomSFTDataset

#Create the custom dataloader function that will call the GPTCustomSFTDataset class to create the dataset from the given text:
def GPTCustomSFTDataloader(file_path,last_token_id = None, max_seq_length = None, batch_size=8, tokenizer = 'tiktoken',
                        shuffle=True, drop_last=True,num_workers=0):

    #Initializer the tokenizer
    if tokenizer == 'tiktoken':
        tokenizer = tiktoken.get_encoding("gpt2")

    #Get the last token id of the tokenizer selected:
    if last_token_id is None:
        last_token_id = tokenizer.encode('<|endoftext|>', allowed_special='all')[0]

    #Create the dataset with the tokenizer and the input file:
    dataset = GPTCustomSFTDataset(text = file_path, tokenizer=tokenizer, last_token_id = last_token_id, max_seq_length= max_seq_length)

    #Create the dataloader with the dataset
    custom_dataloader = DataLoader(dataset,batch_size=batch_size,
                                   shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


    return custom_dataloader