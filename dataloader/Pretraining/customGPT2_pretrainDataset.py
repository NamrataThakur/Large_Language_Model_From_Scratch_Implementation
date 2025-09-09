import torch
from torch.utils.data import Dataset
import tiktoken

#Preparing the dataset for pre-training:
class GPTCustomPretrainDataset(Dataset):
    def __init__(self, tokenizer, input_data):
        self.data = input_data
        self.encoded_data = []
        self.total_tokens = 0

        for row in input_data:
            raw_text = row['text']
            encoded_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})

            row_length = len(encoded_text)
            self.total_tokens = self.total_tokens + row_length
            self.encoded_data.append(encoded_text)

            self.encoded_data.append(encoded_text)

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, index):
        return self.encoded_data[index]
    
    def get_total_tokens(self):
        return self.total_tokens