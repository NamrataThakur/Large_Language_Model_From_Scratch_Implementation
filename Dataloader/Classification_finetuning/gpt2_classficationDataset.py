import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class GPTCustomSFTDataset(Dataset):
    
    def __init__(self, text,tokenizer, last_token_id = None, max_seq_length = None):
        super().__init__()

        #Read the data:
        self.data = pd.read_csv(text)

        #Get the encoded data:
        self.encoded_data = [tokenizer.encode(row, allowed_special='all') 
                             for row in self.data['Text'] 
                            ]
        
        #Truncate the text according to the max sequence length given in the input:
        if max_seq_length is not None:
            self.max_length = max_seq_length

            #Truncate all sequences to the max length:
            self.encoded_data = [row[:self.max_length] 
                                for row in self.encoded_data]
        
        else:
            self.max_length = self._max_length()
            
        #Pad the texts to the maximum length if the length of the sequence is less that the maximum length, else use the sequence as is:
        #Final form of the data that will be used for training
        self.encoded_data = [
                            row + [last_token_id] * (self.max_length - len(row)) 
                            for row in self.encoded_data
                            ]
        

    def __getitem__(self, index):
        X = self.encoded_data[index]
        Y = self.data.iloc[index]['Label']

        X_tensor = torch.tensor(X,dtype=torch.long)
        Y_tensor = torch.tensor(Y,dtype=torch.long)

        #Return the X and Y Tensors for the dataloader
        return X_tensor, Y_tensor

    def __len__(self):
        return len(self.data)
    
    def _max_length(self):
        max_length = max([len(row) for row in self.encoded_data])
        return max_length