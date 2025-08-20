import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

#Preparing the dataset for training:
# X : [i:i+context_length], Y : [i+1: i+1+context_length]
class GPTCustomDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        self.text = text
        self.encoded_id = []
        self.target_id = []

        #Use the tokenizer to encode the raw text
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"} )

        #Use the sliding window based chunking to create the X and Y part of the dataset:
        for i in range(0, len(tokens) - context_length, stride):
            x_chunk = tokens[i : i+context_length]
            y_chunk = tokens[i+1 : i+1+context_length ]
            self.encoded_id.append(torch.tensor(x_chunk))
            self.target_id.append(torch.tensor(y_chunk))

    def __len__(self):
        return len(self.encoded_id)
    
    def __getitem__(self,index):
        return self.encoded_id[index], self.target_id[index]
    


    