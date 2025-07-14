import torch
from torch.utils.data import Dataset

import os
os.pardir

from dataloader.Instruction_finetuning.gpt2_instructDataFormat import format_input_response

class GPTCustomPreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, prompt_style = 'alpaca'):
        super().__init__()
        self.data = data

        self.encoded_data = []

        for row in data:
            ins_length, input = format_input_response(row,prompt_style, inference=True) #inference = True --> Gives output as Instruction + Input.
            correct_response = row['chosen']
            wrong_response = row['rejected']

            encoded_input = tokenizer.encode(input)
            input_correctRes = f"{input}\n\n### Response:\n{correct_response}"
            input_wrongRes = f"{input}\n\n### Response:\n{wrong_response}"

            encoded_input_correctRes = tokenizer.encode(input_correctRes)
            encoded_input_wrongRes = tokenizer.encode(input_wrongRes)

            self.encoded_data.append({
                'input' : encoded_input,
                'correct_response': encoded_input_correctRes,
                'wrong_response' : encoded_input_wrongRes
            })

    def __getitem__(self, index):
        return self.encoded_data[index]
        
    def  __len__(self):
        return len(self.data)